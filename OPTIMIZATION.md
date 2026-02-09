# DeepFilterNet3.c.wasm — Optimization Notes

공식 Rust 구현(libDF + Tract ONNX runtime) 대비 이 C/WASM 엔진에 적용된 최적화 목록.

---

## 아키텍처 차이

| | Rust 공식 | C/WASM (이 프로젝트) |
|---|---|---|
| NN 실행 | Tract ONNX 런타임 (인터프리터) | 수작업 C 코드 (직접 구현) |
| 가중치 형식 | ONNX protobuf (enc/erb_dec/df_dec.onnx) | 단일 바이너리 blob (dfn3_weights.bin) |
| FFT | `realfft` crate (RustFFT 기반) | KissFFT (mixed-radix, BSD-3) |
| SIMD | 없음 (Tract 내부 자동벡터화에 의존) | WASM SIMD128 명시적 intrinsics |
| 메모리 | Tract Tensor 할당/복제/permute | 고정 크기 구조체, zero-alloc 스트리밍 |
| 복소수 | `num_complex::Complex32` 타입 | split re/im 배열 (SIMD 친화적) |

Rust 공식 구현은 ONNX 모델을 Tract 프레임워크로 실행하며, GRU/Conv 등 모든 NN 연산을 Tract 인터프리터가 처리한다. 명시적 SIMD 최적화는 없다.

---

## 적용된 최적화

### 1. Pointwise Conv2d — 입력 전치로 scatter-gather 제거

**문제**: Pointwise Conv2d는 `[C_in, W]` 레이아웃 입력에서 고정 주파수 `w`에 대해 채널 방향(`ci`)으로 내적을 계산해야 한다. 이 접근은 stride-W(최대 96)로 메모리를 건너뛰며 읽으므로, SIMD `wasm_f32x4_make`로 4개 스칼라를 개별 로드해야 한다. 4개 레인이 각각 다른 캐시라인을 히트하여 SIMD 효율이 25-40%로 저하된다.

**해결**: 함수 내부에서 `[C_in, W]` → `[W, C_in]` 전치 후, `dfn3_vdot` (contiguous SIMD dot product)를 사용한다.

```c
// Before: stride-W gather (각 레인이 다른 캐시라인)
v128_t vi = wasm_f32x4_make(in[ci*W+w], in[(ci+1)*W+w], ...);

// After: contiguous access → full SIMD throughput
out[co*W + w] = b + dfn3_vdot(w_row, in_t + w*C_in, C_in);
```

**영향 범위**: 전체 연산의 26.5% (1.92M FLOPs, 10개 호출 지점)

**Rust 대비**: Rust/Tract는 ONNX Conv 연산자를 인터프리터로 실행하며, 이 수준의 메모리 레이아웃 최적화는 없다.

---

### 2. GRU 게이트 통합 연산 (Fused Gate Computation)

**문제**: GRU 셀은 z, r, h_hat 3개 게이트에 대해 각각 `W*x`와 `R*h`를 별도 matvec으로 계산한다. 입력 벡터 `x`와 은닉 상태 `h`를 각각 3번씩 읽게 되어 캐시 재사용이 안 된다.

**해결**: W[3H, input]과 R[3H, H]에 대해 단일 `dfn3_matvec` 호출로 통합한다. x를 1번만 읽으면서 3개 게이트 결과를 연속 메모리에 한번에 생성한다.

```c
// Before: 6번의 별도 matvec (x 3번 읽기, h 3번 읽기)
dfn3_matvec(z, Wz, x, H, input_size);
dfn3_matvec(r, Wr, x, H, input_size);
dfn3_matvec(h_hat, Wh, x, H, input_size);

// After: 2번의 통합 matvec (x 1번, h 1번)
dfn3_matvec(tmp, W, x, 3*H, input_size);   // z,r,h_hat 한번에
dfn3_matvec(gates_h, R, h, 3*H, H);         // Rz*h,Rr*h,Rh*h 한번에
```

**영향 범위**: 전체 연산의 54.9% (3.98M FLOPs, GRU 5개: encoder 1 + ERB decoder 2 + DF decoder 2)

**Rust 대비**: Tract ONNX 런타임이 GRU 연산자를 처리하며, 게이트 통합 여부는 Tract 내부 구현에 의존한다. 명시적 통합 최적화는 확인되지 않음.

---

### 3. df_convp DW Conv — 주파수 축 SIMD 벡터화

**문제**: DF 디코더의 `df_convp`는 Conv2d(64→10, kernel=5×1, groups=2)로, 5중 중첩 스칼라 루프(groups × co × ci × kh × ow)였다. 614K FLOPs가 SIMD 없이 실행되었다.

**해결**: 루프 구조를 재배치하여 주파수 축(ow, 96개)을 SIMD 4-wide로 처리한다. 가중치는 주파수에 무관하므로 `wasm_f32x4_splat`으로 broadcast하고, 입력은 주파수 축에서 stride-1이므로 `wasm_v128_load`로 contiguous 로드한다.

```c
// Before: 스칼라 내부 루프
for (int ow = 0; ow < W_freq; ow++) {
    sum += src[ow] * wt[...];
}

// After: SIMD accumulate (주파수 축 4-wide)
v128_t vw = wasm_f32x4_splat(wval);
for (ow = 0; ow < 96; ow += 4) {
    v128_t vs = wasm_v128_load(src + ow);
    v128_t vd = wasm_v128_load(dst + ow);
    wasm_v128_store(dst + ow, wasm_f32x4_add(vd, wasm_f32x4_mul(vw, vs)));
}
```

**영향 범위**: 전체 연산의 8.5% (614K FLOPs)

**Rust 대비**: Rust/Tract는 이 Conv2d를 일반 ONNX 연산자로 실행하며, 커널 구조에 특화된 SIMD 최적화는 없다.

---

### 4. 전역 WASM SIMD128 벡터화

Rust 공식 구현에는 명시적 SIMD 코드가 전혀 없다. 이 C/WASM 엔진은 다음 연산에 `wasm_simd128.h` intrinsics를 사용한다:

| 연산 | SIMD 적용 |
|------|----------|
| `dfn3_vdot` (내적) | f32x4 FMA + horizontal sum |
| `dfn3_vadd` (벡터 덧셈) | f32x4 add |
| `dfn3_vscale` (스칼라 곱) | f32x4 mul (broadcast) |
| `dfn3_vmul` (원소곱) | f32x4 mul |
| `dfn3_relu_vec` (ReLU) | f32x4 max(v, 0) |
| `dfn3_matvec` (행렬-벡터) | vdot per row |
| `dfn3_matvec_t` (전치 행렬-벡터) | f32x4 broadcast + FMA |
| `dfn3_extract_spec` (특징 추출) | sqrt, div, mul 전부 SIMD |
| `dfn3_apply_erb_mask` (ERB 마스크) | f32x4 broadcast multiply |
| `dfn3_apply_df` (딥 필터링) | 복소 MAC (gather + SIMD) |
| `dfn3_post_filter` (포스트 필터) | sqrt, div, clamp SIMD |
| `dfn3_frame_synthesis` (OLA) | f32x4 add |
| GRU 게이트 bias 합산 | f32x4 add (4개 벡터 동시) |
| GRU h_new 업데이트 | f32x4 (1-z)*h_hat + z*h |

---

### 5. 복소수 split re/im 레이아웃

**Rust**: `num_complex::Complex32`를 사용하여 `[re, im, re, im, ...]` interleaved 레이아웃으로 저장한다.

**C/WASM**: `float spec_re[481]`, `float spec_im[481]`로 split 저장한다. 이 레이아웃은:
- ERB 에너지 계산 시 `vdot(re, re) + vdot(im, im)`으로 SIMD 효율적
- 마스크 적용 시 re/im 독립적 스칼라 곱으로 단순화
- 딥 필터링의 복소 MAC에서 re/im를 각각 contiguous 로드 가능

---

### 6. Zero-allocation 스트리밍

**Rust**: 매 프레임마다 `Tensor::clone()`, `VecDeque::push_back(tensor)`, `permute_axes` 등으로 힙 할당이 발생한다.

**C/WASM**: `DFN3State` 구조체에 모든 버퍼를 고정 크기로 사전 할당하고, `dfn3_process()` 내에서 힙 할당이 **0번** 발생한다. 롤링 버퍼는 `memmove`로 시프트하며, 모든 중간 결과는 스택 또는 구조체 내 scratch 버퍼를 재사용한다.

---

### 7. Tract 인터프리터 오버헤드 제거

Rust 공식은 3개 ONNX 모델(encoder, erb_decoder, df_decoder)을 Tract 런타임으로 실행한다. Tract는 범용 텐서 연산 그래프 인터프리터로, 각 연산마다:
- 연산자 디스패치 오버헤드
- 텐서 shape 검증
- 동적 메모리 할당 (텐서 생성)
- 연산 그래프 순회

이 C 구현은 모든 연산을 인라인 C 코드로 직접 실행하여 이 오버헤드가 없다.

---

### 8. KissFFT mixed-radix (N=960)

960 = 2⁶ × 3 × 5. KissFFT는 mixed-radix를 네이티브로 지원하여 zero-padding 없이 N=960에서 직접 FFT를 수행한다. BSD-3 라이선스로 별도 의존성 없이 소스 포함 가능.

Rust의 `realfft`/`rustfft`도 mixed-radix를 지원하므로 FFT 자체의 성능 차이는 크지 않다.

---

## 연산량 프로파일 (프레임당)

| 연산 | FLOPs | 비율 | SIMD |
|------|-------|------|------|
| GRU 셀 (×5) | 3,983K | 54.9% | ✅ matvec + bias합산 + h_new |
| Pointwise Conv (전체) | 1,920K | 26.5% | ✅ 전치 후 contiguous vdot |
| df_convp DW Conv | 614K | 8.5% | ✅ 주파수 축 4-wide |
| Grouped Linear (전체) | 235K | 3.2% | ✅ matvec_t SIMD |
| FFT + IFFT | 98K | 1.4% | — (KissFFT 내부) |
| DW Conv (erb/df) | 43K | 0.6% | — (소규모) |
| 기타 | 361K | 4.9% | 부분적 |
| **합계** | **~7.25M** | **100%** | |

---

## 빌드 결과

- WASM 바이너리: **75 KB**
- 가중치: **8.2 MB** (dfn3_weights.bin)
- AudioWorklet (WASM + 가중치 base64 임베딩): **~11 MB**
- 컴파일: `emcc -O3 -msimd128`
