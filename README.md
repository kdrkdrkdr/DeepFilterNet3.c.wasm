# DeepFilterNet3.c.wasm

C/WASM reimplementation of the [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) inference engine. Performs real-time speech noise suppression in a browser [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet).

## Why rewrite from scratch

The official implementation is based on Rust + [Tract](https://github.com/sonos/tract) ONNX runtime. This works well as a native binary, but poses problems for browser deployment:

- **Tract cannot be compiled to WASM.** Tract is a large general-purpose ONNX graph interpreter. Building it for the `wasm32-unknown-unknown` target is not officially supported.
- **ONNX Runtime Web is too heavy.** The [ONNX Runtime](https://onnxruntime.ai/) WASM build adds several MB of runtime, exceeding the memory and loading constraints of AudioWorklet threads on mobile browsers.
- **Real-time constraints.** An AudioWorklet must finish processing within 10ms per frame (480 samples @ 48kHz). The interpreter overhead and dynamic memory allocation of a general-purpose runtime is unnecessary cost within this budget.

The most reliable approach was to implement all NN operations directly in C, optimize with [WASM SIMD128](https://github.com/WebAssembly/simd/blob/main/proposals/simd/SIMD.md), and use a fixed-size streaming structure with zero heap allocations.

## Structure

```
dfn3.h              Main engine (single-header, #define DFN3_IMPLEMENTATION)
dfn3_math.h         NN operations (GRU, Conv, matvec, SIMD vector ops)
dfn3_weights.h      Weight layout definitions + model config constants
dfn3_wasm.c         Emscripten entry point (WASM exported functions)
dfn3_weights.bin    Pretrained weight binary (8.2 MB, 2.1M params)
extract_weights.py  Weight extraction script from official ONNX models
build.sh            Emscripten build + AudioWorklet single-file generation
kiss_fft.*          KissFFT (BSD-3, mixed-radix FFT for N=960)
```

## Build

Requires [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).

```bash
bash build.sh
```

Output:
- `dfn3.js` + `dfn3.wasm` -- WASM module
- `../static/js/webclient/denoiser/dfn3-worklet.js` -- Single-file AudioWorklet with WASM + weights base64-embedded

## WASM API

```c
// Initialize: create engine with weights pointer
int dfn3_wasm_create(const void* weights_ptr, int weights_len);

// I/O buffer pointers (480 floats each, 48kHz mono)
float* dfn3_wasm_get_input_ptr(void);
float* dfn3_wasm_get_output_ptr(void);
int dfn3_wasm_get_input_size(void);
int dfn3_wasm_get_output_size(void);

// Process one frame (480 samples in -> 480 samples out)
void dfn3_wasm_process(void);

// Query estimated SNR
float dfn3_wasm_get_lsnr(void);

// Runtime parameters
void dfn3_wasm_set_atten_lim(float db);           // Max attenuation (dB)
void dfn3_wasm_set_post_filter_beta(float beta);   // Post-filter strength
void dfn3_wasm_set_min_db_thresh(float val);        // LSNR lower threshold
void dfn3_wasm_set_max_db_erb_thresh(float val);    // ERB decoder upper bound
void dfn3_wasm_set_max_db_df_thresh(float val);     // DF decoder upper bound

// Cleanup
void dfn3_wasm_destroy(void);
```

## Performance (WASM)

| Device | CPU | Time per frame |
|--------|-----|----------------|
| MacBook (M2) | Apple M2 | ~1 ms |
| Galaxy S23 | Snapdragon 8 Gen 2 | ~4 ms |
| Galaxy Note10+ 5G | Snapdragon 855 | ~6 ms |

Frame = 480 samples @ 48kHz = 10ms audio. Both platforms well within half of the real-time budget.

See [OPTIMIZATION.md](OPTIMIZATION.md) for details.

## License

- [KissFFT](https://github.com/mborgerding/kissfft): BSD-3-Clause
- DeepFilterNet model weights: subject to the [original author's license](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE)
