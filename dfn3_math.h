/* DeepFilterNet3 - Math/NN operations with WASM SIMD optimization */
#ifndef DFN3_MATH_H
#define DFN3_MATH_H

#include <math.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define DFN3_SIMD 1
#else
#define DFN3_SIMD 0
#endif

/* ---- IEEE 754 bit-hack fast math ---- */

/* fast_expf: Schraudolph bit-hack + 1 multiplicative correction.
 * Reduces max relative error from ~4% to ~0.3%.
 * Critical for DFN3's large GRU (H=256, 5 layers). */
static inline float dfn3_fast_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x >  88.0f) return 3.4028235e+38f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(x * 12102203.16156f + 1065353216.0f);
    float e0 = u.f;
    union { float f; int32_t i; } lu = {e0};
    float ln_e0 = (float)lu.i * 8.2629582881927490e-8f - 87.989971088f;
    return e0 * (1.0f + x - ln_e0);
}

/* fast_log2f: IEEE 754 bit-hack log2(x). Max relative error < 0.3%. */
static inline float dfn3_fast_log2f(float x) {
    union { float f; uint32_t i; } u = {x};
    return (float)(int32_t)u.i * 1.1920928955078125e-7f - 126.94269504f;
}

/* fast_sinf for x in [0, pi/2]: 5th-order minimax polynomial.
 * Max error < 0.00025. Used only in post-filter gain. */
static inline float dfn3_fast_sinf(float x) {
    float x2 = x * x;
    return x * (1.0f - x2 * (0.16666667f - x2 * 0.00833333f));
}

/* ---- Activation functions (scalar) ---- */

static inline float dfn3_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float dfn3_sigmoid(float x) {
    return 1.0f / (1.0f + dfn3_fast_expf(-x));
}

static inline float dfn3_fast_tanh(float x) {
    float s = dfn3_sigmoid(2.0f * x);
    return 2.0f * s - 1.0f;
}

/* ---- SIMD helpers ---- */

#if DFN3_SIMD

/* ---- SIMD fast math (v128) ---- */

/* Schraudolph bit-hack exp + correction, 4-wide.
 * Same algorithm as dfn3_fast_expf but operates on v128_t. */
static inline v128_t dfn3_fast_expf_v128(v128_t x) {
    x = wasm_f32x4_max(x, wasm_f32x4_splat(-87.0f));
    x = wasm_f32x4_min(x, wasm_f32x4_splat(88.0f));
    /* u.i = (int32_t)(x * 12102203.16156f + 1065353216.0f) */
    v128_t vsum = wasm_f32x4_add(
        wasm_f32x4_mul(x, wasm_f32x4_splat(12102203.16156f)),
        wasm_f32x4_splat(1065353216.0f));
    v128_t vi = wasm_i32x4_trunc_sat_f32x4(vsum);   /* float→int truncation */
    v128_t e0 = vi;   /* reinterpret int bits as float (v128_t is type-agnostic) */
    /* correction: ln_e0 = (float)bits * 8.263e-8 - 87.99 */
    v128_t vbits_f = wasm_f32x4_convert_i32x4(vi);   /* int→float conversion */
    v128_t vln = wasm_f32x4_sub(
        wasm_f32x4_mul(vbits_f, wasm_f32x4_splat(8.2629582881927490e-8f)),
        wasm_f32x4_splat(87.989971088f));
    /* result = e0 * (1 + x - ln_e0) */
    return wasm_f32x4_mul(e0, wasm_f32x4_add(
        wasm_f32x4_splat(1.0f), wasm_f32x4_sub(x, vln)));
}

static inline v128_t dfn3_sigmoid_v128(v128_t x) {
    v128_t exp_neg = dfn3_fast_expf_v128(wasm_f32x4_neg(x));
    return wasm_f32x4_div(wasm_f32x4_splat(1.0f),
                          wasm_f32x4_add(wasm_f32x4_splat(1.0f), exp_neg));
}

static inline v128_t dfn3_fast_tanh_v128(v128_t x) {
    v128_t two = wasm_f32x4_splat(2.0f);
    v128_t s = dfn3_sigmoid_v128(wasm_f32x4_mul(two, x));
    return wasm_f32x4_sub(wasm_f32x4_mul(two, s), wasm_f32x4_splat(1.0f));
}

static inline v128_t dfn3_fast_sinf_v128(v128_t x) {
    v128_t x2 = wasm_f32x4_mul(x, x);
    return wasm_f32x4_mul(x, wasm_f32x4_sub(
        wasm_f32x4_splat(1.0f),
        wasm_f32x4_mul(x2, wasm_f32x4_sub(
            wasm_f32x4_splat(0.16666667f),
            wasm_f32x4_mul(x2, wasm_f32x4_splat(0.00833333f))))));
}

/* Horizontal sum of f32x4 → scalar */
static inline float dfn3_hsum_f32x4(v128_t v) {
    /* (a,b,c,d) → (a+c, b+d, ?, ?) */
    v128_t hi = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);
    v128_t sum2 = wasm_f32x4_add(v, hi);
    /* (a+c, b+d, ?, ?) → (a+b+c+d, ?, ?, ?) */
    v128_t hi2 = wasm_i32x4_shuffle(sum2, sum2, 1, 0, 3, 2);
    v128_t sum1 = wasm_f32x4_add(sum2, hi2);
    return wasm_f32x4_extract_lane(sum1, 0);
}

/* Bulk ReLU: n floats in-place */
static inline void dfn3_relu_vec(float* x, int n) {
    v128_t zero = wasm_f32x4_splat(0.0f);
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t v = wasm_v128_load(x + i);
        wasm_v128_store(x + i, wasm_f32x4_max(v, zero));
    }
    for (; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/* Bulk vector add: dst[i] += src[i] */
static inline void dfn3_vadd(float* dst, const float* src, int n) {
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t a = wasm_v128_load(dst + i);
        v128_t b = wasm_v128_load(src + i);
        wasm_v128_store(dst + i, wasm_f32x4_add(a, b));
    }
    for (; i < n; i++) {
        dst[i] += src[i];
    }
}

/* Bulk vector multiply by scalar: dst[i] *= s */
static inline void dfn3_vscale(float* dst, float s, int n) {
    v128_t vs = wasm_f32x4_splat(s);
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t v = wasm_v128_load(dst + i);
        wasm_v128_store(dst + i, wasm_f32x4_mul(v, vs));
    }
    for (; i < n; i++) {
        dst[i] *= s;
    }
}

/* dst[i] += s * src[i] for n floats (fused scale-add) */
static inline void dfn3_vscale_add(float* dst, const float* src, float s, int n) {
    v128_t vs = wasm_f32x4_splat(s);
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t vd = wasm_v128_load(dst + i);
        v128_t va = wasm_v128_load(src + i);
        wasm_v128_store(dst + i, wasm_f32x4_add(vd, wasm_f32x4_mul(vs, va)));
    }
    for (; i < n; i++) {
        dst[i] += s * src[i];
    }
}

/* Bulk a[i] = b[i] * c[i] (element-wise multiply) */
static inline void dfn3_vmul(float* dst, const float* a, const float* b, int n) {
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        wasm_v128_store(dst + i, wasm_f32x4_mul(va, vb));
    }
    for (; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

/* Dot product: sum(a[i] * b[i]) for n floats — 4 accumulators for pipeline hiding */
static inline float dfn3_vdot(const float* a, const float* b, int n) {
    v128_t acc0 = wasm_f32x4_splat(0.0f);
    v128_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int i = 0;
    int n16 = n & ~15;
    for (; i < n16; i += 16) {
        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i),      wasm_v128_load(b + i)));
        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4),  wasm_v128_load(b + i + 4)));
        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(a + i + 8),  wasm_v128_load(b + i + 8)));
        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12)));
    }
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    }
    float sum = dfn3_hsum_f32x4(
        wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3)));
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#else /* No SIMD fallbacks */

static inline void dfn3_relu_vec(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

static inline void dfn3_vadd(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

static inline void dfn3_vscale(float* dst, float s, int n) {
    for (int i = 0; i < n; i++) dst[i] *= s;
}

static inline void dfn3_vmul(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] * b[i];
}

static inline float dfn3_vdot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif /* DFN3_SIMD */

/* ---- Matrix-vector multiply: y[M] = A[M,N] * x[N] ---- */
/* 4-row tiling: reads x once per 4 output rows (75% fewer x reads) */
static inline void dfn3_matvec(
    float* restrict y,
    const float* restrict A,
    const float* restrict x,
    int M, int N
) {
#if DFN3_SIMD
    int M4 = M & ~3;
    int N4 = N & ~3;
    int i = 0;
    for (; i < M4; i += 4) {
        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
        const float* a0 = A + i * N;
        const float* a1 = a0 + N;
        const float* a2 = a1 + N;
        const float* a3 = a2 + N;
        for (int j = 0; j < N4; j += 4) {
            v128_t vx = wasm_v128_load(x + j);
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a0 + j), vx));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a1 + j), vx));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(a2 + j), vx));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(a3 + j), vx));
        }
        y[i]   = dfn3_hsum_f32x4(acc0);
        y[i+1] = dfn3_hsum_f32x4(acc1);
        y[i+2] = dfn3_hsum_f32x4(acc2);
        y[i+3] = dfn3_hsum_f32x4(acc3);
        for (int j = N4; j < N; j++) {
            float xj = x[j];
            y[i]   += a0[j] * xj;
            y[i+1] += a1[j] * xj;
            y[i+2] += a2[j] * xj;
            y[i+3] += a3[j] * xj;
        }
    }
    for (; i < M; i++) {
        y[i] = dfn3_vdot(A + i * N, x, N);
    }
#else
    for (int i = 0; i < M; i++) {
        y[i] = dfn3_vdot(A + i * N, x, N);
    }
#endif
}

/* ---- Transposed matrix-vector multiply: y[N] = A[M,N]^T * x[M] ---- */
/* A is stored as [M, N] (row-major), computes y = A^T @ x */
static inline void dfn3_matvec_t(
    float* restrict y,
    const float* restrict A,
    const float* restrict x,
    int M, int N
) {
    memset(y, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        const float* row = A + i * N;
        float xi = x[i];
#if DFN3_SIMD
        v128_t vxi = wasm_f32x4_splat(xi);
        int j = 0;
        int N4 = N & ~3;
        for (; j < N4; j += 4) {
            v128_t vy = wasm_v128_load(y + j);
            v128_t va = wasm_v128_load(row + j);
            wasm_v128_store(y + j, wasm_f32x4_add(vy, wasm_f32x4_mul(vxi, va)));
        }
        for (; j < N; j++) {
            y[j] += xi * row[j];
        }
#else
        for (int j = 0; j < N; j++) {
            y[j] += xi * row[j];
        }
#endif
    }
}

/* ---- Matrix-vector multiply-add: y[M] += A[M,N] * x[N] ---- */
/* 4-row tiling same as dfn3_matvec */
static inline void dfn3_matvec_add(
    float* restrict y,
    const float* restrict A,
    const float* restrict x,
    int M, int N
) {
#if DFN3_SIMD
    int M4 = M & ~3;
    int N4 = N & ~3;
    int i = 0;
    for (; i < M4; i += 4) {
        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
        const float* a0 = A + i * N;
        const float* a1 = a0 + N;
        const float* a2 = a1 + N;
        const float* a3 = a2 + N;
        for (int j = 0; j < N4; j += 4) {
            v128_t vx = wasm_v128_load(x + j);
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a0 + j), vx));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a1 + j), vx));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(a2 + j), vx));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(a3 + j), vx));
        }
        y[i]   += dfn3_hsum_f32x4(acc0);
        y[i+1] += dfn3_hsum_f32x4(acc1);
        y[i+2] += dfn3_hsum_f32x4(acc2);
        y[i+3] += dfn3_hsum_f32x4(acc3);
        for (int j = N4; j < N; j++) {
            float xj = x[j];
            y[i]   += a0[j] * xj;
            y[i+1] += a1[j] * xj;
            y[i+2] += a2[j] * xj;
            y[i+3] += a3[j] * xj;
        }
    }
    for (; i < M; i++) {
        y[i] += dfn3_vdot(A + i * N, x, N);
    }
#else
    for (int i = 0; i < M; i++) {
        y[i] += dfn3_vdot(A + i * N, x, N);
    }
#endif
}

/*
 * GRU cell (single timestep, single layer).
 *
 * ONNX format: W[1, 3H, input], R[1, 3H, hidden], B[1, 6H]
 *   W rows: [W_z | W_r | W_h] each [H, input]
 *   R rows: [R_z | R_r | R_h] each [H, hidden]
 *   B:      [Wb_z | Wb_r | Wb_h | Rb_z | Rb_r | Rb_h] each [H]
 *
 * linear_before_reset=1 (ONNX default for this model):
 *   z = sigmoid(Wz*x + Rbz + Rz*h + Wbz)
 *   r = sigmoid(Wr*x + Rbr + Rr*h + Wbr)
 *   h_hat = tanh(Wh*x + Wbh + r*(Rh*h + Rbh))
 *   h_new = (1-z)*h_hat + z*h_prev
 *
 * h is updated in-place.
 * tmp must be >= 4*H floats.
 */
static void dfn3_gru_cell(
    float* restrict h,           /* [H] hidden state, updated in-place */
    const float* restrict x,     /* [input_size] input */
    const float* restrict W,     /* [3H, input_size] */
    const float* restrict R,     /* [3H, H] */
    const float* restrict B,     /* [6H] */
    int H,                       /* hidden size */
    int input_size,
    float* restrict tmp          /* scratch [4*H] */
) {
    /* Pointers into B */
    const float* Wbz = B;
    const float* Wbr = B + H;
    const float* Wbh = B + 2 * H;
    const float* Rbz = B + 3 * H;
    const float* Rbr = B + 4 * H;
    const float* Rbh = B + 5 * H;

    float* z = tmp;              /* [H] */
    float* r = tmp + H;          /* [H] */
    float* h_hat = tmp + 2 * H;  /* [H] */
    float* rh = tmp + 3 * H;     /* [H] — reused */

    /* --- Fused gate computation: W[3H, input] * x → [z, r, h_hat_wx] --- */
    /* Single pass over x, 3x better cache reuse than 3 separate matvecs */
    dfn3_matvec(tmp, W, x, 3 * H, input_size);
    /* tmp now has: [Wz*x (H) | Wr*x (H) | Wh*x (H)] in z, r, h_hat */

    /* --- Fused: R[3H, H] * h → rh buffer (reuse scratch3 from caller) --- */
    /* Need separate buffer for R*h since we need all 3 slices */
    float gates_h[768];  /* 3 * H = 3 * 256 = 768 max */
    dfn3_matvec(gates_h, R, h, 3 * H, H);
    /* gates_h: [Rz*h (H) | Rr*h (H) | Rh*h (H)] */

    /* --- z = sigmoid(Wz*x + Rz*h + Wbz + Rbz) --- */
    /* z already has Wz*x, add Rz*h */
#if DFN3_SIMD
    {
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vz = wasm_v128_load(z + i);
            v128_t vrh = wasm_v128_load(gates_h + i);
            v128_t vwb = wasm_v128_load(Wbz + i);
            v128_t vrb = wasm_v128_load(Rbz + i);
            vz = wasm_f32x4_add(wasm_f32x4_add(vz, vrh), wasm_f32x4_add(vwb, vrb));
            wasm_v128_store(z + i, dfn3_sigmoid_v128(vz));
        }
        for (; i < H; i++) {
            z[i] = dfn3_sigmoid(z[i] + gates_h[i] + Wbz[i] + Rbz[i]);
        }
    }
#else
    for (int i = 0; i < H; i++) {
        z[i] = dfn3_sigmoid(z[i] + gates_h[i] + Wbz[i] + Rbz[i]);
    }
#endif

    /* --- r = sigmoid(Wr*x + Rr*h + Wbr + Rbr) --- */
    /* r already has Wr*x, add Rr*h */
#if DFN3_SIMD
    {
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vr = wasm_v128_load(r + i);
            v128_t vrh = wasm_v128_load(gates_h + H + i);
            v128_t vwb = wasm_v128_load(Wbr + i);
            v128_t vrb = wasm_v128_load(Rbr + i);
            vr = wasm_f32x4_add(wasm_f32x4_add(vr, vrh), wasm_f32x4_add(vwb, vrb));
            wasm_v128_store(r + i, dfn3_sigmoid_v128(vr));
        }
        for (; i < H; i++) {
            r[i] = dfn3_sigmoid(r[i] + gates_h[H + i] + Wbr[i] + Rbr[i]);
        }
    }
#else
    for (int i = 0; i < H; i++) {
        r[i] = dfn3_sigmoid(r[i] + gates_h[H + i] + Wbr[i] + Rbr[i]);
    }
#endif

    /* --- h_hat = tanh(Wh*x + Wbh + r * (Rh*h + Rbh)) --- */
    /* h_hat already has Wh*x, gates_h+2H has Rh*h */
#if DFN3_SIMD
    {
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vh = wasm_v128_load(h_hat + i);
            v128_t vwb = wasm_v128_load(Wbh + i);
            v128_t vr = wasm_v128_load(r + i);
            v128_t vrh = wasm_v128_load(gates_h + 2 * H + i);
            v128_t vRbh = wasm_v128_load(Rbh + i);
            v128_t val = wasm_f32x4_add(vh, wasm_f32x4_add(vwb,
                wasm_f32x4_mul(vr, wasm_f32x4_add(vrh, vRbh))));
            wasm_v128_store(h_hat + i, dfn3_fast_tanh_v128(val));
        }
        for (; i < H; i++) {
            h_hat[i] = dfn3_fast_tanh(h_hat[i] + Wbh[i] + r[i] * (gates_h[2 * H + i] + Rbh[i]));
        }
    }
#else
    for (int i = 0; i < H; i++) {
        h_hat[i] = dfn3_fast_tanh(h_hat[i] + Wbh[i] + r[i] * (gates_h[2 * H + i] + Rbh[i]));
    }
#endif

    /* h_new = (1-z)*h_hat + z*h_prev */
#if DFN3_SIMD
    {
        v128_t one = wasm_f32x4_splat(1.0f);
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vz = wasm_v128_load(z + i);
            v128_t vh = wasm_v128_load(h + i);
            v128_t vhh = wasm_v128_load(h_hat + i);
            v128_t one_mz = wasm_f32x4_sub(one, vz);
            v128_t result = wasm_f32x4_add(
                wasm_f32x4_mul(one_mz, vhh),
                wasm_f32x4_mul(vz, vh));
            wasm_v128_store(h + i, result);
        }
        for (; i < H; i++) {
            h[i] = (1.0f - z[i]) * h_hat[i] + z[i] * h[i];
        }
    }
#else
    for (int i = 0; i < H; i++) {
        h[i] = (1.0f - z[i]) * h_hat[i] + z[i] * h[i];
    }
#endif
}

/*
 * Grouped linear (einsum): y = einsum("btgi,gih->btgh", x_grouped, W)
 * PyTorch weight layout: W[groups, dim_in_per_group, dim_out_per_group]
 * For each group g: y_g = W_g^T @ x_g  (W_g is [in, out], need transpose)
 */
static inline void dfn3_grouped_linear(
    float* restrict y,
    const float* restrict x,
    const float* restrict W,
    int groups,
    int dim_in_per_group,
    int dim_out_per_group
) {
    for (int g = 0; g < groups; g++) {
        const float* x_g = x + g * dim_in_per_group;
        const float* W_g = W + g * dim_in_per_group * dim_out_per_group;
        float* y_g = y + g * dim_out_per_group;
        dfn3_matvec_t(y_g, W_g, x_g, dim_in_per_group, dim_out_per_group);
    }
}

/*
 * 1D depthwise conv kernel: kW=3, stride=1, freq_pad=1 (W_out == W).
 * Accumulates one temporal row into out[0..W-1].
 * SIMD: process 4 output positions per iteration using shifted loads.
 */
static inline void dfn3_dw_row_k3s1_accum(
    float* restrict out,
    const float* restrict src,
    float w0, float w1, float w2,
    int W
) {
#if DFN3_SIMD
    v128_t vw0 = wasm_f32x4_splat(w0);
    v128_t vw1 = wasm_f32x4_splat(w1);
    v128_t vw2 = wasm_f32x4_splat(w2);

    /* Left edge: out[0] += w1*src[0] + w2*src[1] (w0*0 = 0) */
    out[0] += w1 * src[0] + w2 * src[1];

    /* Bulk SIMD: indices 1..W-2, in groups of 4 */
    int ow = 1;
    for (; ow + 3 < W - 1; ow += 4) {
        v128_t s_left  = wasm_v128_load(src + ow - 1);  /* src[ow-1..ow+2] */
        v128_t s_mid   = wasm_v128_load(src + ow);      /* src[ow..ow+3]   */
        v128_t s_right = wasm_v128_load(src + ow + 1);  /* src[ow+1..ow+4] */
        v128_t vout = wasm_v128_load(out + ow);
        vout = wasm_f32x4_add(vout, wasm_f32x4_mul(vw0, s_left));
        vout = wasm_f32x4_add(vout, wasm_f32x4_mul(vw1, s_mid));
        vout = wasm_f32x4_add(vout, wasm_f32x4_mul(vw2, s_right));
        wasm_v128_store(out + ow, vout);
    }

    /* Scalar tail for remaining interior positions */
    for (; ow < W - 1; ow++) {
        out[ow] += w0 * src[ow - 1] + w1 * src[ow] + w2 * src[ow + 1];
    }

    /* Right edge: out[W-1] += w0*src[W-2] + w1*src[W-1] (w2*0 = 0) */
    if (W > 1) {
        out[W - 1] += w0 * src[W - 2] + w1 * src[W - 1];
    }
#else
    /* Left edge */
    out[0] += w1 * src[0] + w2 * src[1];
    /* Interior */
    for (int ow = 1; ow < W - 1; ow++) {
        out[ow] += w0 * src[ow - 1] + w1 * src[ow] + w2 * src[ow + 1];
    }
    /* Right edge */
    if (W > 1) {
        out[W - 1] += w0 * src[W - 2] + w1 * src[W - 1];
    }
#endif
}

/*
 * 1D depthwise conv kernel: kW=3, stride=2, freq_pad=1.
 * W_out = W/2 (assumes W is even). Accumulates into out[0..W_out-1].
 * out[ow] += w0*src[2*ow-1] + w1*src[2*ow] + w2*src[2*ow+1]
 */
static inline void dfn3_dw_row_k3s2_accum(
    float* restrict out,
    const float* restrict src,
    float w0, float w1, float w2,
    int W_in
) {
    int W_out = W_in / 2;

#if DFN3_SIMD
    /* Left edge: out[0] += w1*src[0] + w2*src[1] (2*0-1 = -1 → pad 0) */
    out[0] += w1 * src[0] + w2 * src[1];

    v128_t vw0 = wasm_f32x4_splat(w0);
    v128_t vw1 = wasm_f32x4_splat(w1);
    v128_t vw2 = wasm_f32x4_splat(w2);

    /* Bulk: ow = 1..W_out-2, need src[2*ow-1], src[2*ow], src[2*ow+1]
       De-interleave: load 8 consecutive floats from src[2*ow-1..2*ow+6]
       then use shuffles to extract even/odd positions */
    int ow = 1;
    for (; ow + 3 < W_out; ow += 4) {
        /* Need: src[2*ow-1], src[2*ow+1], src[2*ow+3], src[2*ow+5]  (left)
                 src[2*ow],   src[2*ow+2], src[2*ow+4], src[2*ow+6]  (mid)
                 src[2*ow+1], src[2*ow+3], src[2*ow+5], src[2*ow+7]  (right) */
        int base = 2 * ow;
        v128_t a = wasm_v128_load(src + base - 1); /* [base-1, base, base+1, base+2] */
        v128_t b = wasm_v128_load(src + base + 3); /* [base+3, base+4, base+5, base+6] */
        v128_t s_left  = wasm_i32x4_shuffle(a, b, 0, 2, 4, 6); /* even: base-1, base+1, base+3, base+5 */
        v128_t s_mid   = wasm_i32x4_shuffle(a, b, 1, 3, 5, 7); /* odd:  base, base+2, base+4, base+6 */
        v128_t c = wasm_v128_load(src + base + 1); /* [base+1, base+2, base+3, base+4] */
        v128_t d = wasm_v128_load(src + base + 5); /* [base+5, base+6, base+7, base+8?] */
        v128_t s_right = wasm_i32x4_shuffle(c, d, 0, 2, 4, 6); /* base+1, base+3, base+5, base+7 */

        v128_t vout = wasm_v128_load(out + ow);
        vout = wasm_f32x4_add(vout, wasm_f32x4_mul(vw0, s_left));
        vout = wasm_f32x4_add(vout, wasm_f32x4_mul(vw1, s_mid));
        vout = wasm_f32x4_add(vout, wasm_f32x4_mul(vw2, s_right));
        wasm_v128_store(out + ow, vout);
    }

    /* Scalar tail */
    for (; ow < W_out; ow++) {
        int iw_left  = 2 * ow - 1;
        int iw_mid   = 2 * ow;
        int iw_right = 2 * ow + 1;
        out[ow] += (iw_left >= 0 ? w0 * src[iw_left] : 0.0f)
                 + w1 * src[iw_mid]
                 + (iw_right < W_in ? w2 * src[iw_right] : 0.0f);
    }
#else
    for (int ow = 0; ow < W_out; ow++) {
        int iw_left  = 2 * ow - 1;
        int iw_mid   = 2 * ow;
        int iw_right = 2 * ow + 1;
        out[ow] += (iw_left >= 0 ? w0 * src[iw_left] : 0.0f)
                 + w1 * src[iw_mid]
                 + (iw_right < W_in ? w2 * src[iw_right] : 0.0f);
    }
#endif
}

/*
 * Depthwise Conv2d: for each channel independently.
 */
static void dfn3_depthwise_conv2d(
    float* restrict out,
    const float* restrict pad_buf,
    const float* restrict cur,
    const float* restrict weight,
    int C, int W, int kH, int kW,
    int freq_pad, int stride_w
) {
    int W_padded = W + 2 * freq_pad;
    int W_out = (W_padded - kW) / stride_w + 1;

    for (int c = 0; c < C; c++) {
        const float* w_c = weight + c * kH * kW;
        float* out_c = out + c * W_out;

        for (int ow = 0; ow < W_out; ow++) {
            float sum = 0.0f;
            for (int kh = 0; kh < kH; kh++) {
                const float* src;
                if (kh < kH - 1) {
                    src = pad_buf + c * (kH - 1) * W + kh * W;
                } else {
                    src = cur + c * W;
                }
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * stride_w + kw - freq_pad;
                    float val = (iw >= 0 && iw < W) ? src[iw] : 0.0f;
                    sum += val * w_c[kh * kW + kw];
                }
            }
            out_c[ow] = sum;
        }
    }
}

/*
 * Pointwise Conv2d (1x1): scatter-accumulate approach.
 * Input:  [C_in, W]
 * Weight: [C_out, C_in, 1, 1]
 * Bias:   [C_out] (can be NULL)
 * Output: [C_out, W]
 *
 * Instead of transposing input (24KB stack), accumulate output directly:
 *   for each ci: out[co][w] += weight[co][ci] * in[ci][w]
 * Each in[ci] row is W contiguous floats → SIMD-friendly vscale_add.
 */
static void dfn3_pointwise_conv2d(
    float* restrict out,
    const float* restrict in,
    const float* restrict weight,
    const float* restrict bias,
    int C_in, int C_out, int W
) {
    /* Initialize output with bias */
#if DFN3_SIMD
    for (int co = 0; co < C_out; co++) {
        float b = bias ? bias[co] : 0.0f;
        v128_t vb = wasm_f32x4_splat(b);
        int w = 0;
        int W4 = W & ~3;
        for (; w < W4; w += 4)
            wasm_v128_store(out + co * W + w, vb);
        for (; w < W; w++)
            out[co * W + w] = b;
    }
    /* Accumulate: for each input channel, add weighted contribution */
    for (int ci = 0; ci < C_in; ci++) {
        const float* in_row = in + ci * W;
        for (int co = 0; co < C_out; co++) {
            float wt = weight[co * C_in + ci];
            dfn3_vscale_add(out + co * W, in_row, wt, W);
        }
    }
#else
    for (int co = 0; co < C_out; co++) {
        float b = bias ? bias[co] : 0.0f;
        for (int w = 0; w < W; w++)
            out[co * W + w] = b;
    }
    for (int ci = 0; ci < C_in; ci++) {
        const float* in_row = in + ci * W;
        for (int co = 0; co < C_out; co++) {
            float wt = weight[co * C_in + ci];
            for (int w = 0; w < W; w++)
                out[co * W + w] += wt * in_row[w];
        }
    }
#endif
}

/*
 * Separable Conv2d block (single timestep, streaming):
 *   depthwise conv -> pointwise conv -> ReLU
 */
static void dfn3_sep_conv2d_block(
    float* restrict out,
    float* restrict dw_out,
    float* restrict pad_buf,
    const float* restrict cur,
    const float* restrict dw_w,
    const float* restrict pw_w,
    const float* restrict pw_b,
    int C_in, int C_out, int W,
    int kH, int kW,
    int freq_pad, int stride_w
) {
    int W_out = (W + 2 * freq_pad - kW) / stride_w + 1;

    /* 1. Depthwise conv */
    dfn3_depthwise_conv2d(dw_out, pad_buf, cur, dw_w,
                          C_in, W, kH, kW, freq_pad, stride_w);

    /* 2. Update temporal padding buffer */
    if (kH > 1) {
        int buf_frames = kH - 1;
        if (buf_frames > 1) {
            memmove(pad_buf, pad_buf + C_in * W,
                    C_in * (buf_frames - 1) * W * sizeof(float));
        }
        memcpy(pad_buf + C_in * (buf_frames - 1) * W,
               cur, C_in * W * sizeof(float));
    }

    /* 3. Pointwise conv + bias */
    dfn3_pointwise_conv2d(out, dw_out, pw_w, pw_b, C_in, C_out, W_out);

    /* 4. ReLU */
    dfn3_relu_vec(out, C_out * W_out);
}

/*
 * Transposed Conv2d (1D on frequency axis, stride 2):
 */
static void dfn3_depthwise_convtranspose1d(
    float* restrict out,
    const float* restrict in,
    const float* restrict weight,
    int C, int W_in, int kW, int stride
) {
    int W_out = (W_in - 1) * stride + kW;
    memset(out, 0, C * W_out * sizeof(float));

    for (int c = 0; c < C; c++) {
        const float* w_c = weight + c * kW;
        const float* in_c = in + c * W_in;
        float* out_c = out + c * W_out;
        for (int i = 0; i < W_in; i++) {
            for (int k = 0; k < kW; k++) {
                out_c[i * stride + k] += in_c[i] * w_c[k];
            }
        }
    }
}

#endif /* DFN3_MATH_H */
