# DeepFilterNet3.c.wasm — Optimization Notes

Optimizations applied to this C/WASM engine compared to the official Rust implementation ([libDF](https://github.com/Rikorose/DeepFilterNet/tree/main/libDF) + [Tract](https://github.com/sonos/tract) ONNX runtime).

---

## Architecture Differences

| | Official Rust | C/WASM (this project) |
|---|---|---|
| NN execution | [Tract](https://github.com/sonos/tract) ONNX runtime (interpreter) | Hand-written C (direct implementation) |
| Weight format | ONNX protobuf (enc/erb_dec/df_dec.onnx) | Single binary blob (dfn3_weights.bin) |
| FFT | [`realfft`](https://crates.io/crates/realfft) crate ([RustFFT](https://crates.io/crates/rustfft)-based) | [KissFFT](https://github.com/mborgerding/kissfft) (mixed-radix, BSD-3) |
| SIMD | None (relies on Tract's internal auto-vectorization) | Explicit [WASM SIMD128](https://github.com/WebAssembly/simd/blob/main/proposals/simd/SIMD.md) intrinsics |
| Memory | Tract Tensor alloc/clone/permute | Fixed-size struct, zero-alloc streaming |
| Complex numbers | [`num_complex::Complex32`](https://docs.rs/num-complex) type | Split re/im arrays (SIMD-friendly) |

The official Rust implementation runs ONNX models through the Tract framework, where all NN operations (GRU, Conv, etc.) are handled by the Tract interpreter. There is no explicit SIMD optimization.

---

## Applied Optimizations

### 1. Global WASM SIMD128 Vectorization

The official Rust implementation has no explicit SIMD code. This C/WASM engine uses [`wasm_simd128.h`](https://clang.llvm.org/docs/LanguageExtensions.html#webassembly-simd-builtins) intrinsics for the following operations:

| Operation | SIMD approach |
|-----------|---------------|
| `dfn3_vdot` (dot product) | f32x4 FMA + horizontal sum |
| `dfn3_vadd` (vector add) | f32x4 add |
| `dfn3_vscale` (scalar mul) | f32x4 mul (broadcast) |
| `dfn3_vmul` (element-wise mul) | f32x4 mul |
| `dfn3_relu_vec` (ReLU) | f32x4 max(v, 0) |
| `dfn3_matvec` (matrix-vector) | vdot per row |
| `dfn3_matvec_t` (transposed matrix-vector) | f32x4 broadcast + FMA |
| `dfn3_extract_spec` (feature extraction) | sqrt, div, mul all SIMD |
| `dfn3_apply_erb_mask` (ERB mask) | f32x4 broadcast multiply |
| `dfn3_apply_df` (deep filtering) | Complex MAC (gather + SIMD) |
| `dfn3_post_filter` (post-filter) | sqrt, div, clamp SIMD |
| `dfn3_frame_synthesis` (OLA) | f32x4 add |
| GRU gate bias addition | f32x4 add (4 vectors at once) |
| GRU h_new update | f32x4 (1-z)*h_hat + z*h |

---

### 2. Pointwise Conv2d — Input Transpose to Eliminate Scatter-Gather

**Problem**: Pointwise Conv2d computes dot products along the channel axis (`ci`) for a fixed frequency `w` from `[C_in, W]` layout input. This requires stride-W (up to 96) memory access, forcing SIMD `wasm_f32x4_make` to individually load 4 scalars. Each of the 4 lanes hits a different cache line, degrading SIMD efficiency to 25-40%.

**Solution**: Transpose `[C_in, W]` to `[W, C_in]` inside the function, then use `dfn3_vdot` (contiguous SIMD dot product).

```c
// Before: stride-W gather (each lane hits a different cache line)
v128_t vi = wasm_f32x4_make(in[ci*W+w], in[(ci+1)*W+w], ...);

// After: contiguous access -> full SIMD throughput
out[co*W + w] = b + dfn3_vdot(w_row, in_t + w*C_in, C_in);
```

**Scope**: 26.5% of total computation (1.92M FLOPs, 10 call sites)

**vs Rust**: Rust/Tract executes the ONNX Conv operator via its interpreter with no memory layout optimization at this level.

---

### 3. GRU Fused Gate Computation

**Problem**: A GRU cell computes 3 gates (z, r, h_hat) with separate matvec calls for `W*x` and `R*h` each. Input vector `x` and hidden state `h` are each read 3 times, preventing cache reuse.

**Solution**: Fuse into a single `dfn3_matvec` call over W[3H, input] and R[3H, H]. `x` is read once to produce all 3 gate results in contiguous memory.

```c
// Before: 6 separate matvecs (x read 3 times, h read 3 times)
dfn3_matvec(z, Wz, x, H, input_size);
dfn3_matvec(r, Wr, x, H, input_size);
dfn3_matvec(h_hat, Wh, x, H, input_size);

// After: 2 fused matvecs (x read once, h read once)
dfn3_matvec(tmp, W, x, 3*H, input_size);   // z,r,h_hat at once
dfn3_matvec(gates_h, R, h, 3*H, H);         // Rz*h,Rr*h,Rh*h at once
```

**Scope**: 54.9% of total computation (3.98M FLOPs, 5 GRU cells: 1 encoder + 2 ERB decoder + 2 DF decoder)

**vs Rust**: Tract ONNX runtime handles the GRU operator internally. Whether it fuses gates depends on Tract's implementation; no explicit fusion optimization has been confirmed.

---

### 4. df_convp Grouped Conv — Frequency-Axis SIMD Vectorization

**Problem**: The DF decoder's `df_convp` is a Grouped Conv2d(64->10, kernel=5x1, groups=2, C_in_per_group=32, C_out_per_group=5) with a 5-level nested scalar loop (groups x co x ci x kh x ow). 614K FLOPs ran without SIMD.

**Solution**: Restructure the loop so the frequency axis (ow, 96 elements) is processed SIMD 4-wide. Weights are frequency-invariant, so `wasm_f32x4_splat` broadcasts the scalar weight. Input is stride-1 along the frequency axis, enabling `wasm_v128_load` for contiguous loads.

```c
// Before: scalar inner loop
for (int ow = 0; ow < W_freq; ow++) {
    sum += src[ow] * wt[...];
}

// After: SIMD accumulate (frequency axis 4-wide)
v128_t vw = wasm_f32x4_splat(wval);
for (ow = 0; ow < 96; ow += 4) {
    v128_t vs = wasm_v128_load(src + ow);
    v128_t vd = wasm_v128_load(dst + ow);
    wasm_v128_store(dst + ow, wasm_f32x4_add(vd, wasm_f32x4_mul(vw, vs)));
}
```

**Scope**: 8.5% of total computation (614K FLOPs)

**vs Rust**: Rust/Tract runs this Conv2d as a generic ONNX operator with no kernel-structure-specific SIMD optimization.

---

### 5. Split re/im Complex Layout

**Rust**: Uses [`num_complex::Complex32`](https://docs.rs/num-complex) with `[re, im, re, im, ...]` interleaved layout.

**C/WASM**: Stores as `float spec_re[481]`, `float spec_im[481]` (split layout). This enables:
- ERB energy: `vdot(re, re) + vdot(im, im)` with full SIMD efficiency
- Mask application: independent scalar multiply on re/im
- Deep filtering complex MAC: contiguous load for re/im separately

---

### 6. Zero-Allocation Streaming

**Rust**: Every frame incurs heap allocations via `Tensor::clone()`, `VecDeque::push_back(tensor)`, `permute_axes`, etc.

**C/WASM**: All buffers are pre-allocated at fixed sizes in the `DFN3State` struct. `dfn3_process()` performs **zero** heap allocations. Rolling buffers shift via `memmove`, and all intermediate results reuse stack or scratch buffers within the struct.

---

### 7. Tract Interpreter Overhead Elimination

The official Rust implementation runs 3 ONNX models (encoder, erb_decoder, df_decoder) through the [Tract](https://github.com/sonos/tract) runtime. Tract is a general-purpose tensor computation graph interpreter, incurring per-operation overhead:
- Operator dispatch
- Tensor shape validation
- Dynamic memory allocation (tensor creation)
- Computation graph traversal

This C implementation executes all operations as inline C code, eliminating this overhead entirely.

---

### 8. KissFFT Mixed-Radix (N=960)

960 = 2^6 x 3 x 5. [KissFFT](https://github.com/mborgerding/kissfft) natively supports mixed-radix, performing FFT directly at N=960 without zero-padding. BSD-3 licensed, included as source with no external dependency.

Rust's [`realfft`](https://crates.io/crates/realfft)/[`rustfft`](https://crates.io/crates/rustfft) also supports mixed-radix, so the FFT performance difference itself is minimal.

---

## FLOP Profile (per frame)

| Operation | FLOPs | Share |
|-----------|-------|-------|
| GRU cells (x5) | 3,983K | 54.9% |
| Pointwise Conv (total) | 1,920K | 26.5% |
| df_convp Grouped Conv | 614K | 8.5% |
| Grouped Linear (total) | 235K | 3.2% |
| FFT + IFFT | 98K | 1.4% |
| DW Conv (erb/df) | 43K | 0.6% |
| Other | 361K | 4.9% |
| **Total** | **~7.25M** | **100%** |

---

## Build Output

- WASM binary: **75 KB**
- Weights: **8.2 MB** (dfn3_weights.bin)
- AudioWorklet (WASM + weights base64 embedded): **~11 MB**

---

## Measured Performance (WASM, not native)

| Device | CPU | Time per frame | vs audio budget |
|--------|-----|----------------|-----------------|
| MacBook (M2) | Apple M2 | **~1 ms** | 10% |
| Samsung Galaxy S23 | Snapdragon 8 Gen 2 (Cortex-X3) | **~4 ms** | 40% |
| Samsung Galaxy Note10+ 5G | Snapdragon 855 (Kryo 485) | **~6 ms** | 60% |

- Frame = 480 samples @ 48 kHz = **10 ms** audio
- Both platforms well within real-time budget (RTF < 0.5)
