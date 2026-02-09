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

/* ---- Activation functions (scalar) ---- */

static inline float dfn3_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float dfn3_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ---- SIMD helpers ---- */

#if DFN3_SIMD

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

/* Dot product: sum(a[i] * b[i]) for n floats */
static inline float dfn3_vdot(const float* a, const float* b, int n) {
    v128_t acc = wasm_f32x4_splat(0.0f);
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
    }
    float sum = dfn3_hsum_f32x4(acc);
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
static inline void dfn3_matvec(
    float* restrict y,
    const float* restrict A,
    const float* restrict x,
    int M, int N
) {
    for (int i = 0; i < M; i++) {
        y[i] = dfn3_vdot(A + i * N, x, N);
    }
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
static inline void dfn3_matvec_add(
    float* restrict y,
    const float* restrict A,
    const float* restrict x,
    int M, int N
) {
    for (int i = 0; i < M; i++) {
        y[i] += dfn3_vdot(A + i * N, x, N);
    }
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
    /* Pointers into W, R, B */
    const float* Wz = W;
    const float* Wr = W + H * input_size;
    const float* Wh = W + 2 * H * input_size;

    const float* Rz = R;
    const float* Rr = R + H * H;
    const float* Rh = R + 2 * H * H;

    const float* Wbz = B;
    const float* Wbr = B + H;
    const float* Wbh = B + 2 * H;
    const float* Rbz = B + 3 * H;
    const float* Rbr = B + 4 * H;
    const float* Rbh = B + 5 * H;

    float* z = tmp;
    float* r = tmp + H;
    float* h_hat = tmp + 2 * H;
    float* rh = tmp + 3 * H;

    /* z = Wz*x */
    dfn3_matvec(z, Wz, x, H, input_size);
    /* z += Rz*h */
    dfn3_matvec_add(z, Rz, h, H, H);
    /* z[i] = sigmoid(z[i] + Wbz[i] + Rbz[i]) */
#if DFN3_SIMD
    {
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vz = wasm_v128_load(z + i);
            v128_t vwb = wasm_v128_load(Wbz + i);
            v128_t vrb = wasm_v128_load(Rbz + i);
            vz = wasm_f32x4_add(vz, wasm_f32x4_add(vwb, vrb));
            /* Scalar sigmoid — SIMD add saves 2 loads + adds per 4 elements */
            for (int k = 0; k < 4; k++) {
                float v = wasm_f32x4_extract_lane(vz, 0);
                z[i + k] = dfn3_sigmoid(v);
                vz = wasm_i32x4_shuffle(vz, vz, 1, 2, 3, 0);
            }
        }
        for (; i < H; i++) {
            z[i] = dfn3_sigmoid(z[i] + Wbz[i] + Rbz[i]);
        }
    }
#else
    for (int i = 0; i < H; i++) {
        z[i] = dfn3_sigmoid(z[i] + Wbz[i] + Rbz[i]);
    }
#endif

    /* r = Wr*x */
    dfn3_matvec(r, Wr, x, H, input_size);
    dfn3_matvec_add(r, Rr, h, H, H);
    /* r[i] = sigmoid(r[i] + Wbr[i] + Rbr[i]) */
#if DFN3_SIMD
    {
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vr = wasm_v128_load(r + i);
            v128_t vwb = wasm_v128_load(Wbr + i);
            v128_t vrb = wasm_v128_load(Rbr + i);
            vr = wasm_f32x4_add(vr, wasm_f32x4_add(vwb, vrb));
            for (int k = 0; k < 4; k++) {
                float v = wasm_f32x4_extract_lane(vr, 0);
                r[i + k] = dfn3_sigmoid(v);
                vr = wasm_i32x4_shuffle(vr, vr, 1, 2, 3, 0);
            }
        }
        for (; i < H; i++) {
            r[i] = dfn3_sigmoid(r[i] + Wbr[i] + Rbr[i]);
        }
    }
#else
    for (int i = 0; i < H; i++) {
        r[i] = dfn3_sigmoid(r[i] + Wbr[i] + Rbr[i]);
    }
#endif

    /* linear_before_reset=1: rh = Rh*h, h_hat = Wh*x */
    dfn3_matvec(rh, Rh, h, H, H);
    dfn3_matvec(h_hat, Wh, x, H, input_size);
    /* h_hat[i] = tanh(h_hat[i] + Wbh[i] + r[i] * (rh[i] + Rbh[i])) */
#if DFN3_SIMD
    {
        int i = 0;
        int H4 = H & ~3;
        for (; i < H4; i += 4) {
            v128_t vh = wasm_v128_load(h_hat + i);
            v128_t vwb = wasm_v128_load(Wbh + i);
            v128_t vr = wasm_v128_load(r + i);
            v128_t vrh = wasm_v128_load(rh + i);
            v128_t vRbh = wasm_v128_load(Rbh + i);
            /* h_hat + Wbh + r * (rh + Rbh) */
            v128_t val = wasm_f32x4_add(vh, wasm_f32x4_add(vwb,
                wasm_f32x4_mul(vr, wasm_f32x4_add(vrh, vRbh))));
            for (int k = 0; k < 4; k++) {
                float v = wasm_f32x4_extract_lane(val, 0);
                h_hat[i + k] = tanhf(v);
                val = wasm_i32x4_shuffle(val, val, 1, 2, 3, 0);
            }
        }
        for (; i < H; i++) {
            h_hat[i] = tanhf(h_hat[i] + Wbh[i] + r[i] * (rh[i] + Rbh[i]));
        }
    }
#else
    for (int i = 0; i < H; i++) {
        h_hat[i] = tanhf(h_hat[i] + Wbh[i] + r[i] * (rh[i] + Rbh[i]));
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
 * Pointwise Conv2d (1x1): for each spatial position.
 * Input:  [C_in, W]
 * Weight: [C_out, C_in, 1, 1]
 * Bias:   [C_out] (can be NULL)
 * Output: [C_out, W]
 *
 * SIMD: inner ci loop uses 4-wide dot product.
 */
static void dfn3_pointwise_conv2d(
    float* restrict out,
    const float* restrict in,
    const float* restrict weight,
    const float* restrict bias,
    int C_in, int C_out, int W
) {
#if DFN3_SIMD
    int C4 = C_in & ~3;
    for (int co = 0; co < C_out; co++) {
        const float* w_row = weight + co * C_in;
        float b = bias ? bias[co] : 0.0f;
        for (int w = 0; w < W; w++) {
            v128_t acc = wasm_f32x4_splat(0.0f);
            int ci = 0;
            for (; ci < C4; ci += 4) {
                v128_t vw = wasm_v128_load(w_row + ci);
                /* Gather in[ci*W+w] .. in[(ci+3)*W+w] — not contiguous */
                v128_t vi = wasm_f32x4_make(
                    in[ci * W + w],
                    in[(ci + 1) * W + w],
                    in[(ci + 2) * W + w],
                    in[(ci + 3) * W + w]);
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(vw, vi));
            }
            float sum = b + dfn3_hsum_f32x4(acc);
            for (; ci < C_in; ci++) {
                sum += w_row[ci] * in[ci * W + w];
            }
            out[co * W + w] = sum;
        }
    }
#else
    for (int co = 0; co < C_out; co++) {
        const float* w_row = weight + co * C_in;
        float b = bias ? bias[co] : 0.0f;
        for (int w = 0; w < W; w++) {
            float sum = b;
            for (int ci = 0; ci < C_in; ci++) {
                sum += w_row[ci] * in[ci * W + w];
            }
            out[co * W + w] = sum;
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
