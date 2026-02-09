/*
 * DeepFilterNet3 — Pure C streaming inference engine.
 * Complete 1:1 reproduction of the Rust libDF implementation.
 *
 * Single-header library.  Include in exactly one .c file with:
 *   #define DFN3_IMPLEMENTATION
 *   #include "dfn3.h"
 *
 * Public API:
 *   dfn3_create()   — allocate & initialize state
 *   dfn3_process()  — process one hop (480 samples @ 48 kHz)
 *   dfn3_destroy()  — free state
 */
#ifndef DFN3_H
#define DFN3_H

#include <stddef.h>

typedef struct DFN3State DFN3State;

/* Create a new DFN3 instance.
 * weights_data: pointer to loaded dfn3_weights.bin (must remain valid). */
DFN3State* dfn3_create(const void* weights_data);

/* Process one hop of 480 float samples (48 kHz, mono).
 * Input and output may alias.  Returns 480 enhanced samples. */
void dfn3_process(DFN3State* st, const float* in, float* out);

/* Free all resources. */
void dfn3_destroy(DFN3State* st);

/* ======================================================================== */
#ifdef DFN3_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dfn3_weights.h"
#include "dfn3_math.h"
#include "kiss_fftr.h"

/* ---- constants ---- */
#define FFT_SIZE   DFN3_FFT_SIZE    /* 960 */
#define HOP_SIZE   DFN3_HOP_SIZE    /* 480 */
#define FREQ_BINS  DFN3_FREQ_BINS   /* 481 */
#define NB_ERB     DFN3_NB_ERB      /* 32  */
#define NB_DF      DFN3_NB_DF       /* 96  */
#define DF_ORDER   DFN3_DF_ORDER    /* 5   */
#define CONV_CH    DFN3_CONV_CH     /* 64  */
#define EMB_DIM    DFN3_EMB_DIM     /* 512 */
#define EMB_HIDDEN DFN3_EMB_HIDDEN  /* 256 */

/* Rust constants */
#define MIN_DB_THRESH    (-10.0f)
#define MAX_DB_ERB_THRESH  30.0f
#define MAX_DB_DF_THRESH   20.0f
#define POST_FILTER_BETA   0.02f
#define SILENCE_RMS_THRESH 1e-7f
#define SILENCE_SKIP_MAX   5

/* Rolling buffer sizes — matches Rust tract.rs exactly */
#define CONV_LOOKAHEAD   DFN3_CONV_LOOKAHEAD  /* 2 */
#define DF_LOOKAHEAD     DFN3_DF_LOOKAHEAD    /* 2 (but df_lookahead for DFN3 = 0 in apply) */
#define LOOKAHEAD        CONV_LOOKAHEAD       /* max(conv_lookahead, df_lookahead) = max(2,0) = 2 */
#define SPEC_BUF_Y_SIZE  (DF_ORDER + CONV_LOOKAHEAD)   /* 5 + 2 = 7 */
#define SPEC_BUF_X_SIZE  (DF_ORDER)                    /* max(lookahead, df_order) = max(2,5) = 5 */

/* ERB band widths (sum = 481), from Rust erb_fb() */
static const int ERB_WIDTHS[NB_ERB] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7,
    7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67
};

/* ---- FFT for N=960 using KissFFT ---- */
/* KissFFT provides O(N log N) mixed-radix FFT.
   960 = 2^6 × 3 × 5 is supported natively by KissFFT's mixed-radix engine.
   KissFFT rfft has the same convention as Rust realfft:
   - Forward: DC bin = sum(input), no scaling
   - Round-trip: irfft(rfft(x)) = x × N (un-normalized inverse) */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Real FFT: N real → N/2+1 complex (split re/im arrays)
 * kiss_cpx_buf: scratch of at least N/2+1 kiss_fft_cpx.
 */
static void dfn3_rfft(kiss_fftr_cfg cfg, const float* input,
                      float* out_re, float* out_im, int N,
                      kiss_fft_cpx* kiss_cpx_buf) {
    int N2p1 = N / 2 + 1;
    kiss_fftr(cfg, input, kiss_cpx_buf);
    /* Deinterleave complex → split re/im */
    for (int i = 0; i < N2p1; i++) {
        out_re[i] = kiss_cpx_buf[i].r;
        out_im[i] = kiss_cpx_buf[i].i;
    }
}

/* Inverse real FFT: N/2+1 complex (split re/im) → N real
 * UN-NORMALIZED — round-trip gives input × N.
 * kiss_cpx_buf: scratch of at least N/2+1 kiss_fft_cpx.
 */
static void dfn3_irfft(kiss_fftr_cfg cfg, const float* in_re, const float* in_im,
                       float* output, int N,
                       kiss_fft_cpx* kiss_cpx_buf) {
    int N2p1 = N / 2 + 1;
    /* Pack split re/im into interleaved kiss_fft_cpx */
    for (int i = 0; i < N2p1; i++) {
        kiss_cpx_buf[i].r = in_re[i];
        kiss_cpx_buf[i].i = in_im[i];
    }
    kiss_fftri(cfg, kiss_cpx_buf, output);
}

/* ---- DFN3State ---- */

struct DFN3State {
    DFN3Weights w;

    /* STFT state */
    float window[FFT_SIZE];
    float wnorm;
    /* analysis_mem: overlap buffer of size (fft_size - hop_size) = 480
       Matches Rust: analysis_mem stores the previous frame for overlap. */
    float analysis_mem[HOP_SIZE];
    /* synthesis_mem: overlap-add buffer of size (fft_size - hop_size) = 480 */
    float synthesis_mem[HOP_SIZE];

    /* FFT (KissFFT) */
    float fft_buf[FFT_SIZE];
    kiss_fftr_cfg kiss_fwd;  /* forward rfft config */
    kiss_fftr_cfg kiss_inv;  /* inverse rfft config */
    kiss_fft_cpx kiss_cpx_buf[FREQ_BINS]; /* scratch for N/2+1 complex bins */

    /* Current frame spectrum */
    float spec_re[FREQ_BINS];
    float spec_im[FREQ_BINS];

    /* Rolling spectrum buffers — match Rust tract.rs rolling_spec_buf_y / _x exactly.
       Each frame stores full FREQ_BINS complex spectrum as [re[0..481], im[0..481]].
       rolling_spec_buf_y: enhanced stage 1 (ERB+DF applied to delayed frame)
       rolling_spec_buf_x: noisy spectra for DF FIR filter and post-filter reference */
    float rolling_spec_buf_y[SPEC_BUF_Y_SIZE][2 * FREQ_BINS];  /* [7][962] */
    float rolling_spec_buf_x[SPEC_BUF_X_SIZE][2 * FREQ_BINS];  /* [5][962] */

    /* Feature normalization state */
    float erb_norm_state[NB_ERB];
    float unit_norm_state[NB_DF];   /* Only NB_DF bins, not FREQ_BINS */
    float norm_alpha;

    /* Feature buffers */
    float feat_erb[NB_ERB];
    float feat_spec[2 * NB_DF];   /* [real[96], imag[96]] */

    /* ---- Encoder state ---- */
    /* Conv temporal padding buffers:
       erb_conv0: kH=3, needs 2 past frames of [1 * NB_ERB]
       df_conv0:  kH=3, needs 2 past frames of [2 * NB_DF] */
    float enc_erb_conv0_pad[2 * NB_ERB];       /* [2, 32] */
    float enc_df_conv0_pad[2 * 2 * NB_DF];     /* [2, 2, 96] for 2 input channels */

    /* Encoder intermediate buffers */
    float e0[CONV_CH * NB_ERB];           /* [64, 32] */
    float e1[CONV_CH * (NB_ERB / 2)];     /* [64, 16] */
    float e2[CONV_CH * (NB_ERB / 4)];     /* [64, 8]  */
    float e3[CONV_CH * (NB_ERB / 4)];     /* [64, 8]  */
    float c0[CONV_CH * NB_DF];            /* [64, 96] */

    /* Embedding */
    float emb[EMB_DIM];
    float lsnr;

    /* GRU hidden states */
    float enc_gru_h[EMB_HIDDEN];
    float erb_gru0_h[EMB_HIDDEN];
    float erb_gru1_h[EMB_HIDDEN];
    float df_gru0_h[EMB_HIDDEN];
    float df_gru1_h[EMB_HIDDEN];
    float gru_tmp[4 * EMB_HIDDEN];

    /* ---- ERB Decoder state ---- */
    float erb_mask[NB_ERB];

    /* ---- DF Decoder state ---- */

    /* DF convp temporal padding: c0 [64, 96] needs 4 past frames (kernel=5)
       pad shape: [64, 4, 96] */
    float df_convp_pad[CONV_CH * 4 * NB_DF];

    /* Runtime parameters (settable from JS) */
    float atten_lim;          /* linear attenuation limit: 10^(-dB/20), 0=no limit */
    float post_filter_beta;   /* post-filter strength, 0=disabled */
    float min_db_thresh;
    float max_db_erb_thresh;
    float max_db_df_thresh;

    /* Silence detection */
    int silence_skip_counter;

    /* Scratch buffers */
    float scratch1[CONV_CH * FREQ_BINS];
    float scratch2[CONV_CH * FREQ_BINS];
    float scratch3[EMB_DIM * 4];
};

/* ---- Initialization ---- */

static void dfn3_init_window(DFN3State* st) {
    /* Vorbis window from Rust: sin(PI/2 * sin^2(PI * (i + 0.5) / window_size_h))
       where window_size_h = fft_size / 2 */
    int window_size_h = FFT_SIZE / 2;  /* 480 */
    for (int n = 0; n < FFT_SIZE; n++) {
        double sin_val = sin(M_PI / 2.0 * ((double)n + 0.5) / (double)window_size_h);
        st->window[n] = (float)sin(M_PI / 2.0 * sin_val * sin_val);
    }
    /* wnorm = 1 / (fft_size^2 / (2 * hop_size)) = 2 * hop_size / fft_size^2 */
    st->wnorm = 2.0f * HOP_SIZE / ((float)FFT_SIZE * (float)FFT_SIZE);
}

static void dfn3_init_norm_state(DFN3State* st) {
    /* From Rust: MEAN_NORM_INIT = [-60, -90] linearly interpolated */
    for (int i = 0; i < NB_ERB; i++) {
        st->erb_norm_state[i] = -60.0f + (float)i * (-90.0f - (-60.0f)) / (NB_ERB - 1);
    }
    /* From Rust: UNIT_NORM_INIT = [0.001, 0.0001] linearly interpolated */
    for (int i = 0; i < NB_DF; i++) {
        st->unit_norm_state[i] = 0.001f + (float)i * (0.0001f - 0.001f) / (NB_DF - 1);
    }
    /* alpha = exp(-hop_size / (sr * tau)) */
    st->norm_alpha = expf(-(float)HOP_SIZE / (DFN3_SR * DFN3_NORM_TAU));
}

DFN3State* dfn3_create(const void* weights_data) {
    DFN3State* st = (DFN3State*)calloc(1, sizeof(DFN3State));
    if (!st) return NULL;

    dfn3_weights_init(&st->w, (const float*)weights_data);
    dfn3_init_window(st);
    dfn3_init_norm_state(st);
    st->silence_skip_counter = 0;

    /* Runtime parameter defaults (match Rust defaults) */
    st->atten_lim = 0.0f;           /* 0 = no limit (100dB) */
    st->post_filter_beta = 0.0f;    /* 0 = disabled */
    st->min_db_thresh = -10.0f;
    st->max_db_erb_thresh = 30.0f;
    st->max_db_df_thresh = 20.0f;

    /* Initialize KissFFT configs */
    st->kiss_fwd = kiss_fftr_alloc(FFT_SIZE, 0, NULL, NULL);  /* forward */
    st->kiss_inv = kiss_fftr_alloc(FFT_SIZE, 1, NULL, NULL);  /* inverse */
    if (!st->kiss_fwd || !st->kiss_inv) {
        if (st->kiss_fwd) kiss_fftr_free(st->kiss_fwd);
        if (st->kiss_inv) kiss_fftr_free(st->kiss_inv);
        free(st);
        return NULL;
    }

    return st;
}

void dfn3_destroy(DFN3State* st) {
    if (st) {
        if (st->kiss_fwd) kiss_fftr_free(st->kiss_fwd);
        if (st->kiss_inv) kiss_fftr_free(st->kiss_inv);
        free(st);
    }
}

/* ---- STFT (Rust lib.rs frame_analysis) ---- */

static void dfn3_frame_analysis(DFN3State* st, const float* in) {
    /* Rust: buf[0..fft_size-frame_size] = analysis_mem * window
             buf[fft_size-frame_size..] = input * window[fft_size-frame_size..]
       For fft_size=960, hop_size=480: analysis_mem = 480 samples */

    /* First half: previous overlap (analysis_mem) */
    dfn3_vmul(st->fft_buf, st->analysis_mem, st->window, HOP_SIZE);
    /* Second half: new input */
    dfn3_vmul(st->fft_buf + HOP_SIZE, in, st->window + HOP_SIZE, HOP_SIZE);

    /* Update analysis_mem: store current input for next frame */
    memcpy(st->analysis_mem, in, HOP_SIZE * sizeof(float));

    /* FFT */
    dfn3_rfft(st->kiss_fwd, st->fft_buf, st->spec_re, st->spec_im, FFT_SIZE,
              st->kiss_cpx_buf);

    /* Normalize by wnorm */
    dfn3_vscale(st->spec_re, st->wnorm, FREQ_BINS);
    dfn3_vscale(st->spec_im, st->wnorm, FREQ_BINS);
}

/* ---- ISTFT (Rust lib.rs frame_synthesis) ---- */

static void dfn3_frame_synthesis(DFN3State* st, float* out,
                                  const float* enh_re, const float* enh_im) {
    /* IFFT */
    dfn3_irfft(st->kiss_inv, enh_re, enh_im, st->fft_buf, FFT_SIZE,
               st->kiss_cpx_buf);

    /* Apply window */
    dfn3_vmul(st->fft_buf, st->fft_buf, st->window, FFT_SIZE);

    /* Overlap-add: first HOP_SIZE samples = new + synthesis_mem */
#if DFN3_SIMD
    {
        int i = 0;
        for (; i < (HOP_SIZE & ~3); i += 4) {
            v128_t a = wasm_v128_load(st->fft_buf + i);
            v128_t b = wasm_v128_load(st->synthesis_mem + i);
            wasm_v128_store(out + i, wasm_f32x4_add(a, b));
        }
        for (; i < HOP_SIZE; i++) {
            out[i] = st->fft_buf[i] + st->synthesis_mem[i];
        }
    }
#else
    for (int i = 0; i < HOP_SIZE; i++) {
        out[i] = st->fft_buf[i] + st->synthesis_mem[i];
    }
#endif

    /* Store second half for next frame's overlap-add */
    memcpy(st->synthesis_mem, st->fft_buf + HOP_SIZE, HOP_SIZE * sizeof(float));
}

/* ---- Feature extraction ---- */

static void dfn3_extract_erb(DFN3State* st) {
    /* Rust: band_corr -> dB -> band_mean_norm */
    float alpha = st->norm_alpha;
    int offset = 0;
    for (int b = 0; b < NB_ERB; b++) {
        int w = ERB_WIDTHS[b];
        /* band_corr: mean of |spec|^2 over band
           energy = dot(re,re) + dot(im,im) using SIMD vdot */
        float energy = dfn3_vdot(st->spec_re + offset, st->spec_re + offset, w)
                     + dfn3_vdot(st->spec_im + offset, st->spec_im + offset, w);
        energy /= w;  /* k = 1.0 / band_size */

        /* Convert to dB: 10 * log10(energy + 1e-10) */
        float db = 10.0f * log10f(energy + 1e-10f);

        /* band_mean_norm: state = x*(1-alpha) + state*alpha; x = (x - state)/40 */
        st->erb_norm_state[b] = db * (1.0f - alpha) + st->erb_norm_state[b] * alpha;
        st->feat_erb[b] = (db - st->erb_norm_state[b]) / 40.0f;

        offset += w;
    }
}

static void dfn3_extract_spec(DFN3State* st) {
    /* Rust: band_unit_norm:
       magnitude = |spec[i]|
       state[i] = magnitude * (1 - alpha) + state[i] * alpha
       spec[i] /= sqrt(state[i])  <-- critical: sqrt, not raw state */
    float alpha = st->norm_alpha;
#if DFN3_SIMD
    v128_t valpha = wasm_f32x4_splat(alpha);
    v128_t v1malpha = wasm_f32x4_splat(1.0f - alpha);
    /* NB_DF=96 is divisible by 4 */
    for (int i = 0; i < NB_DF; i += 4) {
        v128_t vre = wasm_v128_load(st->spec_re + i);
        v128_t vim = wasm_v128_load(st->spec_im + i);
        /* mag² = re² + im² */
        v128_t mag2 = wasm_f32x4_add(wasm_f32x4_mul(vre, vre),
                                      wasm_f32x4_mul(vim, vim));
        /* mag = sqrt(mag²) */
        v128_t vmag = wasm_f32x4_sqrt(mag2);
        /* EMA: state = mag * (1-alpha) + state * alpha */
        v128_t vstate = wasm_v128_load(st->unit_norm_state + i);
        vstate = wasm_f32x4_add(wasm_f32x4_mul(vmag, v1malpha),
                                wasm_f32x4_mul(vstate, valpha));
        wasm_v128_store(st->unit_norm_state + i, vstate);
        /* inv = 1/sqrt(state) — use SIMD sqrt + div */
        v128_t vinv = wasm_f32x4_div(wasm_f32x4_splat(1.0f), wasm_f32x4_sqrt(vstate));
        /* feat_spec[i] = spec_re[i] * inv, feat_spec[NB_DF+i] = spec_im[i] * inv */
        wasm_v128_store(st->feat_spec + i, wasm_f32x4_mul(vre, vinv));
        wasm_v128_store(st->feat_spec + NB_DF + i, wasm_f32x4_mul(vim, vinv));
    }
#else
    for (int i = 0; i < NB_DF; i++) {
        float mag = sqrtf(st->spec_re[i] * st->spec_re[i]
                        + st->spec_im[i] * st->spec_im[i]);

        st->unit_norm_state[i] = mag * (1.0f - alpha) + st->unit_norm_state[i] * alpha;
        float inv = 1.0f / sqrtf(st->unit_norm_state[i]);

        st->feat_spec[i]         = st->spec_re[i] * inv;  /* real channel */
        st->feat_spec[NB_DF + i] = st->spec_im[i] * inv;  /* imag channel */
    }
#endif
}

/* ---- Encoder ---- */

static void dfn3_encoder(DFN3State* st) {
    const DFN3Weights* w = &st->w;

    /* ---- ERB path ---- */

    /* erb_conv0: Conv2d(1, 64, 3, 3) with group=1 [ONNX node 17]
       kernel_shape=[3,3], group=1, pads=[0,1,0,1], strides=[1,1]
       Temporal dim: causally padded (2 past frames)
       Frequency dim: pads=[0,1,0,1] = left=1, right=1
       Single standard conv (C_in=1 → C_out=64), no separate pointwise.
       Followed directly by ReLU → e0 */
    {
        int C_in = 1;
        int W_in = NB_ERB;  /* 32 */
        int kH = 3, kW = 3;
        int freq_pad = 1;
        int C_out = CONV_CH;
        float* pad = st->enc_erb_conv0_pad;  /* [2, 32] two previous frames */

        for (int co = 0; co < C_out; co++) {
            const float* wt = w->enc_erb_conv0_dw_w.data + co * C_in * kH * kW;
            float bias = w->enc_erb_conv0_dw_b.data[co];
            for (int ow = 0; ow < W_in; ow++) {
                float sum = bias;
                for (int kh = 0; kh < kH; kh++) {
                    const float* src;
                    if (kh < 2) {
                        src = pad + kh * W_in;
                    } else {
                        src = st->feat_erb;
                    }
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow + kw - freq_pad;
                        float val = (iw >= 0 && iw < W_in) ? src[iw] : 0.0f;
                        sum += val * wt[kh * kW + kw];
                    }
                }
                st->e0[co * W_in + ow] = dfn3_relu(sum);
            }
        }

        /* Update pad: shift left, store current frame */
        memmove(pad, pad + W_in, W_in * sizeof(float));
        memcpy(pad + W_in, st->feat_erb, W_in * sizeof(float));
    }

    /* erb_conv1: DW[64,1,1,3] stride=(1,2) pads=[0,1,0,1] + PW + ReLU → e1 */
    {
        int W_in = NB_ERB;       /* 32 */
        int W_out = NB_ERB / 2;  /* 16 */
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->enc_erb_conv1_dw_w.data + c * 3;
            for (int ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow * 2 + kw - 1;
                    float val = (iw >= 0 && iw < W_in) ? st->e0[c * W_in + iw] : 0.0f;
                    sum += val * wt[kw];
                }
                st->scratch1[c * W_out + ow] = sum;
            }
        }
        dfn3_pointwise_conv2d(st->e1, st->scratch1,
                              w->enc_erb_conv1_pw_w.data,
                              w->enc_erb_conv1_pw_b.data,
                              CONV_CH, CONV_CH, W_out);
        dfn3_relu_vec(st->e1, CONV_CH * W_out);
    }

    /* erb_conv2: DW[64,1,1,3] stride=(1,2) pads=[0,1,0,1] + PW + ReLU → e2 */
    {
        int W_in = NB_ERB / 2;   /* 16 */
        int W_out = NB_ERB / 4;  /* 8 */
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->enc_erb_conv2_dw_w.data + c * 3;
            for (int ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow * 2 + kw - 1;
                    float val = (iw >= 0 && iw < W_in) ? st->e1[c * W_in + iw] : 0.0f;
                    sum += val * wt[kw];
                }
                st->scratch1[c * W_out + ow] = sum;
            }
        }
        dfn3_pointwise_conv2d(st->e2, st->scratch1,
                              w->enc_erb_conv2_pw_w.data,
                              w->enc_erb_conv2_pw_b.data,
                              CONV_CH, CONV_CH, W_out);
        dfn3_relu_vec(st->e2, CONV_CH * W_out);
    }

    /* erb_conv3: DW[64,1,1,3] stride=(1,1) pads=[0,1,0,1] + PW + ReLU → e3 */
    {
        int W_io = NB_ERB / 4;  /* 8 */
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->enc_erb_conv3_dw_w.data + c * 3;
            for (int ow = 0; ow < W_io; ow++) {
                float sum = 0.0f;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow + kw - 1;
                    float val = (iw >= 0 && iw < W_io) ? st->e2[c * W_io + iw] : 0.0f;
                    sum += val * wt[kw];
                }
                st->scratch1[c * W_io + ow] = sum;
            }
        }
        dfn3_pointwise_conv2d(st->e3, st->scratch1,
                              w->enc_erb_conv3_pw_w.data,
                              w->enc_erb_conv3_pw_b.data,
                              CONV_CH, CONV_CH, W_io);
        dfn3_relu_vec(st->e3, CONV_CH * W_io);
    }

    /* ---- DF path ---- */

    /* df_conv0: Conv2d with group=2, kernel_shape=[3,3], pads=[0,1,0,1]
       Input: feat_spec [2, 96]
       Weight: [64, 1, 3, 3], groups=2
       C_in=2, C_out=64, C_out_per_group=32, C_in_per_group=1
       Causally padded (2 past frames) */
    {
        int C_in = 2;
        int W_in = NB_DF;  /* 96 */
        int kH = 3, kW = 3;
        int freq_pad = 1;
        int groups = 2;
        int C_out_per_group = CONV_CH / groups;  /* 32 */
        float* pad = st->enc_df_conv0_pad;

        for (int g = 0; g < groups; g++) {
            for (int co = 0; co < C_out_per_group; co++) {
                int co_abs = g * C_out_per_group + co;
                const float* wt = w->enc_df_conv0_dw_w.data + co_abs * kH * kW;

                for (int ow = 0; ow < W_in; ow++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kH; kh++) {
                        const float* src;
                        if (kh < 2) {
                            src = pad + g * 2 * W_in + kh * W_in;
                        } else {
                            src = st->feat_spec + g * W_in;
                        }
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow + kw - freq_pad;
                            float val = (iw >= 0 && iw < W_in) ? src[iw] : 0.0f;
                            sum += val * wt[kh * kW + kw];
                        }
                    }
                    st->c0[co_abs * W_in + ow] = sum;
                }
            }
        }

        /* Update pad for each input channel */
        for (int g = 0; g < groups; g++) {
            memmove(pad + g * 2 * W_in,
                    pad + g * 2 * W_in + W_in,
                    W_in * sizeof(float));
            memcpy(pad + g * 2 * W_in + W_in,
                   st->feat_spec + g * W_in,
                   W_in * sizeof(float));
        }

        /* PW conv [64,64,1,1] + bias + ReLU → c0 */
        dfn3_pointwise_conv2d(st->scratch1, st->c0,
                              w->enc_df_conv0_pw_w.data,
                              w->enc_df_conv0_pw_b.data,
                              CONV_CH, CONV_CH, W_in);
        memcpy(st->c0, st->scratch1, CONV_CH * W_in * sizeof(float));
        dfn3_relu_vec(st->c0, CONV_CH * W_in);
    }

    /* df_conv1: DW[64,1,1,3] group=64 stride=(1,2) pads=[0,1,0,1] [node 48]
               + PW[64,64,1,1] group=1 + bias [node 49]
               + ReLU [node 50] */
    {
        int W_in = NB_DF;       /* 96 */
        int W_out = NB_DF / 2;  /* 48 */

        /* Depthwise conv */
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->enc_df_conv1_dw_w.data + c * 3;
            for (int ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow * 2 + kw - 1;
                    float val = (iw >= 0 && iw < W_in) ? st->c0[c * W_in + iw] : 0.0f;
                    sum += val * wt[kw];
                }
                st->scratch1[c * W_out + ow] = sum;
            }
        }

        /* Pointwise conv [64,64,1,1] + bias + ReLU */
        dfn3_pointwise_conv2d(st->scratch2, st->scratch1,
                              w->enc_df_conv1_pw_w.data,
                              w->enc_df_conv1_pw_b.data,
                              CONV_CH, CONV_CH, W_out);
        dfn3_relu_vec(st->scratch2, CONV_CH * W_out);
    }

    /* ---- Feature embedding ---- */
    /* ONNX: Transpose(NCHW→NHWC) → Reshape → Einsum('btgi,gih→btgh') → ReLU
       scratch2 = [64, 48] in NCHW (C, W)
       Transpose [0,2,3,1] → NHWC: [48, 64] = [W, C]
       Flatten → [3072]
       Einsum with df_fc_emb_w [32, 96, 16]: groups=32, in=96, out=16
       Output: [512] */
    {
        float flat[3072];
        /* Transpose [C=64, W=48] → [W=48, C=64] then flatten */
        for (int ch = 0; ch < CONV_CH; ch++) {
            for (int f = 0; f < 48; f++) {
                flat[f * CONV_CH + ch] = st->scratch2[ch * 48 + f];
            }
        }

        float cemb[EMB_DIM];
        dfn3_grouped_linear(cemb, flat,
                           w->enc_df_fc_emb_w.data,
                           32, 96, 16);
        dfn3_relu_vec(cemb, EMB_DIM);

        /* e3 embedding: [C=64, W=8] → Transpose NCHW→NHWC → [W=8, C=64] → flatten [512] */
        float erb_flat[EMB_DIM];
        for (int ch = 0; ch < CONV_CH; ch++) {
            for (int f = 0; f < 8; f++) {
                erb_flat[f * CONV_CH + ch] = st->e3[ch * 8 + f];
            }
        }

        /* Combine: emb = erb_flat + cemb (from ONNX Add node [90]) */
        memcpy(st->emb, erb_flat, EMB_DIM * sizeof(float));
        dfn3_vadd(st->emb, cemb, EMB_DIM);
    }

    /* ---- Encoder SqueezedGRU ---- */
    /* linear_in: grouped [16, 32, 16] → [512] → [256] + ReLU
       GRU: hidden=256
       linear_out: grouped [16, 16, 32] → [256] → [512] + ReLU */
    {
        float gru_in[EMB_HIDDEN];
        dfn3_grouped_linear(gru_in, st->emb,
                           w->enc_emb_gru_lin_in_w.data,
                           16, 32, 16);
        for (int i = 0; i < EMB_HIDDEN; i++) {
            gru_in[i] = dfn3_relu(gru_in[i]);
        }

        dfn3_gru_cell(st->enc_gru_h, gru_in,
                     w->enc_emb_gru0_W.data,
                     w->enc_emb_gru0_R.data,
                     w->enc_emb_gru0_B.data,
                     EMB_HIDDEN, EMB_HIDDEN,
                     st->gru_tmp);

        dfn3_grouped_linear(st->emb, st->enc_gru_h,
                           w->enc_emb_gru_lin_out_w.data,
                           16, 16, 32);
        dfn3_relu_vec(st->emb, EMB_DIM);
    }

    /* ---- LSNR ---- */
    /* ONNX: MatMul(emb, lsnr_fc_w) + lsnr_fc_b → Sigmoid → Mul(50) + Add(-15) */
    {
        /* lsnr_fc_w is [512, 1], so it's a dot product */
        float val = w->enc_lsnr_fc_b.data[0] + dfn3_vdot(st->emb, w->enc_lsnr_fc_w.data, EMB_DIM);
        val = dfn3_sigmoid(val);
        st->lsnr = val * (DFN3_LSNR_MAX - DFN3_LSNR_MIN) + DFN3_LSNR_MIN;
    }
}

/* ---- ERB Decoder ---- */

static void dfn3_erb_decoder(DFN3State* st) {
    const DFN3Weights* w = &st->w;

    /* SqueezedGRU with 2 GRU layers:
       linear_in: [16, 32, 16] → [256] + ReLU
       GRU layer 0: hidden=256
       GRU layer 1: hidden=256
       linear_out: [16, 16, 32] → [512] + ReLU */

    float gru_in[EMB_HIDDEN];
    dfn3_grouped_linear(gru_in, st->emb,
                       w->erb_emb_gru_lin_in_w.data,
                       16, 32, 16);
    dfn3_relu_vec(gru_in, EMB_HIDDEN);

    dfn3_gru_cell(st->erb_gru0_h, gru_in,
                 w->erb_emb_gru0_W.data,
                 w->erb_emb_gru0_R.data,
                 w->erb_emb_gru0_B.data,
                 EMB_HIDDEN, EMB_HIDDEN,
                 st->gru_tmp);

    dfn3_gru_cell(st->erb_gru1_h, st->erb_gru0_h,
                 w->erb_emb_gru1_W.data,
                 w->erb_emb_gru1_R.data,
                 w->erb_emb_gru1_B.data,
                 EMB_HIDDEN, EMB_HIDDEN,
                 st->gru_tmp);

    float dec_emb[EMB_DIM];
    dfn3_grouped_linear(dec_emb, st->erb_gru1_h,
                       w->erb_emb_gru_lin_out_w.data,
                       16, 16, 32);
    dfn3_relu_vec(dec_emb, EMB_DIM);

    /* Reshape [512] → [8, 64] (NHWC) → Transpose NHWC→NCHW → [64, 8] */
    float emb_2d[CONV_CH * 8];
    for (int f = 0; f < 8; f++) {
        for (int ch = 0; ch < CONV_CH; ch++) {
            emb_2d[ch * 8 + f] = dec_emb[f * CONV_CH + ch];
        }
    }

    /* ---- Skip connections with decoder convolutions ---- */

    /* conv3p: Conv[1,1] group=64 on e3 + bias + ReLU, then Add with emb_2d
       ONNX: conv3p is depthwise [64,1,1,1] = per-channel scale + bias */
    {
        for (int c = 0; c < CONV_CH; c++) {
            float wt = w->erb_conv3p_dw_w.data[c];
            float b  = w->erb_conv3p_dw_b.data[c];
            for (int f = 0; f < 8; f++) {
                st->scratch1[c * 8 + f] = dfn3_relu(st->e3[c * 8 + f] * wt + b);
            }
        }
        /* Add skip: conv3p(e3) + emb_2d */
        dfn3_vadd(st->scratch1, emb_2d, CONV_CH * 8);
    }

    /* convt3: DW[64,1,1,3] pads=[0,1,0,1] stride=1 + PW[64,64,1,1] + ReLU */
    {
        int W_io = 8;
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->erb_convt3_dw_w.data + c * 3;
            for (int ow = 0; ow < W_io; ow++) {
                float sum = 0.0f;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow + kw - 1;
                    float val = (iw >= 0 && iw < W_io) ? st->scratch1[c * W_io + iw] : 0.0f;
                    sum += val * wt[kw];
                }
                st->scratch2[c * W_io + ow] = sum;
            }
        }
        dfn3_pointwise_conv2d(st->scratch1, st->scratch2,
                              w->erb_convt3_pw_w.data,
                              w->erb_convt3_pw_b.data,
                              CONV_CH, CONV_CH, W_io);
        dfn3_relu_vec(st->scratch1, CONV_CH * W_io);
    }

    /* conv2p: Conv[1,1] group=64 on e2 + bias + ReLU, then Add with convt3 output */
    {
        for (int c = 0; c < CONV_CH; c++) {
            float wt = w->erb_conv2p_dw_w.data[c];
            float b  = w->erb_conv2p_dw_b.data[c];
            for (int f = 0; f < 8; f++) {
                st->scratch2[c * 8 + f] = dfn3_relu(st->e2[c * 8 + f] * wt + b);
            }
        }
        dfn3_vadd(st->scratch2, st->scratch1, CONV_CH * 8);
    }

    /* convt2: ConvTranspose DW[64,1,1,3] stride=2 output_padding=[0,1] pads=[0,1,0,1]
       + PW[64,64,1,1] + ReLU
       Input [64, 8] → output [64, 16] */
    {
        int W_in = 8;
        int W_out = 16;  /* (8-1)*2 + 3 = 17, but with output_padding=1 and pads → 16 */

        /* ConvTranspose: scatter-add */
        float trans_out[CONV_CH * 17];
        memset(trans_out, 0, sizeof(float) * CONV_CH * 17);
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->erb_convt2_dw_w.data + c * 3;
            for (int i = 0; i < W_in; i++) {
                for (int k = 0; k < 3; k++) {
                    trans_out[c * 17 + i * 2 + k] += st->scratch2[c * W_in + i] * wt[k];
                }
            }
        }
        /* Apply output_padding=[0,1] and pads=[0,1,0,1] on frequency axis:
           ONNX ConvTranspose pads=[0,1,0,1]: remove 1 from left, 1 from right
           output_padding=[0,1]: add 1 to the output
           Effective: start at index 1, take W_out=16 elements */
        for (int c = 0; c < CONV_CH; c++) {
            for (int f = 0; f < W_out; f++) {
                st->scratch1[c * W_out + f] = trans_out[c * 17 + f + 1];
            }
        }

        dfn3_pointwise_conv2d(st->scratch2, st->scratch1,
                              w->erb_convt2_pw_w.data,
                              w->erb_convt2_pw_b.data,
                              CONV_CH, CONV_CH, W_out);
        dfn3_relu_vec(st->scratch2, CONV_CH * W_out);
    }

    /* conv1p: Conv[1,1] group=64 on e1 + bias + ReLU, then Add with convt2 output */
    {
        for (int c = 0; c < CONV_CH; c++) {
            float wt = w->erb_conv1p_dw_w.data[c];
            float b  = w->erb_conv1p_dw_b.data[c];
            for (int f = 0; f < 16; f++) {
                st->scratch1[c * 16 + f] = dfn3_relu(st->e1[c * 16 + f] * wt + b);
            }
        }
        dfn3_vadd(st->scratch1, st->scratch2, CONV_CH * 16);
    }

    /* convt1: ConvTranspose DW[64,1,1,3] stride=2 output_padding=[0,1] pads=[0,1,0,1]
       + PW[64,64,1,1] + ReLU
       Input [64, 16] → output [64, 32] */
    {
        int W_in = 16;
        int W_out = 32;
        int W_trans = (W_in - 1) * 2 + 3;  /* 33 */

        float trans_out[CONV_CH * 33];
        memset(trans_out, 0, sizeof(float) * CONV_CH * 33);
        for (int c = 0; c < CONV_CH; c++) {
            const float* wt = w->erb_convt1_dw_w.data + c * 3;
            for (int i = 0; i < W_in; i++) {
                for (int k = 0; k < 3; k++) {
                    trans_out[c * 33 + i * 2 + k] += st->scratch1[c * W_in + i] * wt[k];
                }
            }
        }
        /* Apply pads=[0,1,0,1] + output_padding=[0,1]: take indices [1..32] */
        for (int c = 0; c < CONV_CH; c++) {
            for (int f = 0; f < W_out; f++) {
                st->scratch2[c * W_out + f] = trans_out[c * 33 + f + 1];
            }
        }

        dfn3_pointwise_conv2d(st->scratch1, st->scratch2,
                              w->erb_convt1_pw_w.data,
                              w->erb_convt1_pw_b.data,
                              CONV_CH, CONV_CH, W_out);
        for (int i = 0; i < CONV_CH * W_out; i++) {
            st->scratch1[i] = dfn3_relu(st->scratch1[i]);
        }
    }

    /* conv0p: Conv[1,1] group=64 on e0 + bias + ReLU, then Add with convt1 output */
    {
        for (int c = 0; c < CONV_CH; c++) {
            float wt = w->erb_conv0p_dw_w.data[c];
            float b  = w->erb_conv0p_dw_b.data[c];
            for (int f = 0; f < NB_ERB; f++) {
                st->scratch2[c * NB_ERB + f] = dfn3_relu(st->e0[c * NB_ERB + f] * wt + b);
            }
        }
        dfn3_vadd(st->scratch2, st->scratch1, CONV_CH * NB_ERB);
    }

    /* conv0_out: Conv[1,3] group=1, C_in=64, C_out=1, pads=[0,1,0,1] + Sigmoid
       → erb_mask [32] */
    {
        const float* wt = w->erb_conv0_out_w.data; /* [1, 64, 1, 3] */
        float b = w->erb_conv0_out_b.data[0];
        for (int ow = 0; ow < NB_ERB; ow++) {
            float sum = b;
            for (int ci = 0; ci < CONV_CH; ci++) {
                const float* w_ci = wt + ci * 3;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow + kw - 1;
                    float val = (iw >= 0 && iw < NB_ERB) ? st->scratch2[ci * NB_ERB + iw] : 0.0f;
                    sum += val * w_ci[kw];
                }
            }
            st->erb_mask[ow] = dfn3_sigmoid(sum);
        }
    }
}

/* ---- DF Decoder ---- */

static void dfn3_df_decoder(DFN3State* st) {
    const DFN3Weights* w = &st->w;

    /* SqueezedGRU with 2 GRU layers:
       linear_in: [8, 64, 32] → [256] + ReLU
       GRU layer 0: hidden=256
       GRU layer 1: hidden=256
       (no linear_out — output goes to df_skip + df_out) */

    float gru_in[EMB_HIDDEN];
    dfn3_grouped_linear(gru_in, st->emb,
                       w->df_gru_lin_in_w.data,
                       8, 64, 32);
    dfn3_relu_vec(gru_in, EMB_HIDDEN);

    dfn3_gru_cell(st->df_gru0_h, gru_in,
                 w->df_gru0_W.data,
                 w->df_gru0_R.data,
                 w->df_gru0_B.data,
                 EMB_HIDDEN, EMB_HIDDEN,
                 st->gru_tmp);

    dfn3_gru_cell(st->df_gru1_h, st->df_gru0_h,
                 w->df_gru1_W.data,
                 w->df_gru1_R.data,
                 w->df_gru1_B.data,
                 EMB_HIDDEN, EMB_HIDDEN,
                 st->gru_tmp);

    /* df_skip: grouped linear [16, 32, 16] → [256]
       ONNX Add node: GRU sequence output + df_skip(emb)
       IMPORTANT: This is added to the GRU *sequence* output, NOT the hidden state.
       The hidden state (df_gru1_h) must remain unmodified for next frame. */
    float df_combined[EMB_HIDDEN];  /* GRU output + skip, separate from hidden state */
    {
        float skip[EMB_HIDDEN];
        dfn3_grouped_linear(skip, st->emb,
                           w->df_skip_w.data,
                           16, 32, 16);
        memcpy(df_combined, st->df_gru1_h, EMB_HIDDEN * sizeof(float));
        dfn3_vadd(df_combined, skip, EMB_HIDDEN);
    }

    /* NOTE: df_fc_a (alpha attention) exists in the ONNX model as a separate output
       but does NOT feed back into the coefs computation. In DeepFilterNet3 (unlike
       DFN2), alpha is a separate output that the Rust tract runtime ignores
       (init_df_decoder_impl only captures "coefs" output). So we skip it. */

    /* ---- df_out: grouped linear [16, 16, 60] → [960] + Tanh ---- */
    /* Output shape: [960] = [96 * 10] = [nb_df * df_order * 2] */
    float coefs_gru[NB_DF * DF_ORDER * 2];  /* [960] */
    dfn3_grouped_linear(coefs_gru, df_combined,
                       w->df_out_w.data,
                       16, 16, 60);
    for (int i = 0; i < NB_DF * DF_ORDER * 2; i++) {
        coefs_gru[i] = tanhf(coefs_gru[i]); /* tanh is scalar — no SIMD */
    }

    /* ---- c0 pathway: df_convp ---- */
    /* c0 [64, 96] → causal pad (4 past frames) → Conv[5,1] group=2 → PW[10,10,1,1]
       → ReLU → Transpose[0,2,3,1] (NCHW→NHWC) → [96, 10] */
    float c0_proj[NB_DF * 10];  /* [96, 10] after transpose */
    {
        /* df_convp DW: Conv2d(64, 10, (5,1), groups=2)
           Weight: [10, 32, 5, 1]
           C_in=64, C_out=10, groups=2
           C_in_per_group=32, C_out_per_group=5 */
        int C_in = CONV_CH;   /* 64 */
        int C_out = 10;
        int groups = 2;
        int C_in_per_group = C_in / groups;    /* 32 */
        int C_out_per_group = C_out / groups;  /* 5  */
        int kH = 5, kW = 1;
        int W_freq = NB_DF;  /* 96 */
        float* pad = st->df_convp_pad;  /* [64, 4, 96] */

        float dw_out[10 * NB_DF];  /* [C_out, W_freq] */

        for (int g = 0; g < groups; g++) {
            for (int co = 0; co < C_out_per_group; co++) {
                int co_abs = g * C_out_per_group + co;
                const float* wt = w->df_convp_dw_w.data + co_abs * C_in_per_group * kH * kW;
                float* dst = dw_out + co_abs * W_freq;

                /* Zero output for accumulation */
                memset(dst, 0, W_freq * sizeof(float));

                /* Accumulate: for each input channel and temporal tap,
                   multiply contiguous frequency vector by scalar weight.
                   src[ow] is stride-1 along freq → perfect for SIMD. */
                for (int ci = 0; ci < C_in_per_group; ci++) {
                    int ci_abs = g * C_in_per_group + ci;
                    for (int kh = 0; kh < kH; kh++) {
                        const float* src;
                        if (kh < 4) {
                            src = pad + ci_abs * 4 * W_freq + kh * W_freq;
                        } else {
                            src = st->c0 + ci_abs * W_freq;
                        }
                        float wval = wt[(ci * kH + kh) * kW];
#if DFN3_SIMD
                        v128_t vw = wasm_f32x4_splat(wval);
                        int ow = 0;
                        /* NB_DF=96 is divisible by 4 */
                        for (; ow < (W_freq & ~3); ow += 4) {
                            v128_t vs = wasm_v128_load(src + ow);
                            v128_t vd = wasm_v128_load(dst + ow);
                            wasm_v128_store(dst + ow, wasm_f32x4_add(vd, wasm_f32x4_mul(vw, vs)));
                        }
                        for (; ow < W_freq; ow++) {
                            dst[ow] += src[ow] * wval;
                        }
#else
                        for (int ow = 0; ow < W_freq; ow++) {
                            dst[ow] += src[ow] * wval;
                        }
#endif
                    }
                }
            }
        }

        /* Update temporal padding: shift left by 1, store current c0 */
        for (int ci = 0; ci < C_in; ci++) {
            memmove(pad + ci * 4 * W_freq,
                    pad + ci * 4 * W_freq + W_freq,
                    3 * W_freq * sizeof(float));
            memcpy(pad + ci * 4 * W_freq + 3 * W_freq,
                   st->c0 + ci * W_freq,
                   W_freq * sizeof(float));
        }

        /* PW conv [10, 10, 1, 1] + bias + ReLU */
        float pw_out[10 * NB_DF];
        dfn3_pointwise_conv2d(pw_out, dw_out,
                              w->df_convp_pw_w.data,
                              w->df_convp_pw_b.data,
                              10, 10, W_freq);
        dfn3_relu_vec(pw_out, 10 * W_freq);

        /* Transpose NCHW → NHWC: [10, 96] → [96, 10] */
        for (int ch = 0; ch < 10; ch++) {
            for (int f = 0; f < W_freq; f++) {
                c0_proj[f * 10 + ch] = pw_out[ch * W_freq + f];
            }
        }
    }

    /* ---- Combine: coefs = Tanh(df_out) + Transpose(df_convp(c0)) ---- */
    /* ONNX Add node [112]: reshape df_out to [B, S, 96, 10] + c0_proj [B, S, 96, 10]
       coefs_gru is [960] = flat [96, 10], c0_proj is [96, 10] */
    float coefs_combined[NB_DF * DF_ORDER * 2];
    memcpy(coefs_combined, coefs_gru, NB_DF * 10 * sizeof(float));
    dfn3_vadd(coefs_combined, c0_proj, NB_DF * 10);

    /* The coefs_combined are [96, 10] = [nb_df, df_order * 2]
       Rust reshapes to [ch, nb_df, df_order, 2] where last 2 = [real, imag]
       Memory layout: for each freq f, for each order n:
         coefs[f * 10 + n * 2]     = real part
         coefs[f * 10 + n * 2 + 1] = imag part */

    /* Store in state for apply_df */
    memcpy(st->scratch3, coefs_combined, NB_DF * DF_ORDER * 2 * sizeof(float));
}

/* ---- Apply ERB mask to spectrum ---- */

static void dfn3_apply_erb_mask(DFN3State* st, float* enh_re, float* enh_im) {
    int offset = 0;
    for (int b = 0; b < NB_ERB; b++) {
        float gain = st->erb_mask[b];
        int bw = ERB_WIDTHS[b];
#if DFN3_SIMD
        v128_t vg = wasm_f32x4_splat(gain);
        int i = 0;
        int bw4 = bw & ~3;
        for (; i < bw4; i += 4) {
            int idx = offset + i;
            wasm_v128_store(enh_re + idx, wasm_f32x4_mul(wasm_v128_load(st->spec_re + idx), vg));
            wasm_v128_store(enh_im + idx, wasm_f32x4_mul(wasm_v128_load(st->spec_im + idx), vg));
        }
        for (; i < bw; i++) {
            int idx = offset + i;
            enh_re[idx] = st->spec_re[idx] * gain;
            enh_im[idx] = st->spec_im[idx] * gain;
        }
#else
        for (int i = 0; i < bw; i++) {
            int idx = offset + i;
            enh_re[idx] = st->spec_re[idx] * gain;
            enh_im[idx] = st->spec_im[idx] * gain;
        }
#endif
        offset += bw;
    }
}

/* ---- Apply deep filtering ---- */
/* Rust df() function (tract.rs lines 724-767):
   - Reads from rolling_spec_buf_x (full FREQ_BINS spectra)
   - Zeros output bins 0..nb_df first
   - Iterates over frames in spec (= rolling_spec_buf_x), zipped with coefs axis 2
   - For each frame: complex multiply-accumulate spec[frame][freq] * coefs[freq][frame]
   - coefs shape after set_shape: [ch, nb_df, df_order, 2]

   In our C code: rolling_spec_buf_x has SPEC_BUF_X_SIZE=5 frames.
   coefs layout: [nb_df, df_order * 2] = [96, 10]
   coefs[f][n] = Complex(coefs[f*10 + n*2], coefs[f*10 + n*2 + 1]) */

static void dfn3_apply_df(DFN3State* st, float* enh_re, float* enh_im) {
    const float* coefs = st->scratch3;

    /* Zero output bins 0..nb_df (Rust: o_f.slice_mut(s![.., ..nb_df]).fill(Complex32::default())) */
    memset(enh_re, 0, NB_DF * sizeof(float));
    memset(enh_im, 0, NB_DF * sizeof(float));

    /* Iterate over DF frames: spec_iter.zip(coefs_arr.axis_iter(Axis(2)))
       coefs has df_order on axis 2, so zip takes min(spec.len(), df_order) = df_order frames.
       Rust iterates spec from front (oldest) to back (newest). */
    for (int n = 0; n < DF_ORDER; n++) {
        const float* frame_re = st->rolling_spec_buf_x[n];              /* re part */
        const float* frame_im = st->rolling_spec_buf_x[n] + FREQ_BINS;  /* im part */

#if DFN3_SIMD
        /* NB_DF=96 is divisible by 4, but coefs layout is strided: coefs[f*10 + n*2]
           We must gather coefs (stride 10), spec is contiguous. */
        for (int f = 0; f < NB_DF; f += 4) {
            /* Gather coefs: cr[k] = coefs[(f+k)*10 + n*2], ci[k] = coefs[(f+k)*10 + n*2 + 1] */
            int base0 = f * 10 + n * 2;
            v128_t vcr = wasm_f32x4_make(coefs[base0], coefs[base0 + 10],
                                          coefs[base0 + 20], coefs[base0 + 30]);
            v128_t vci = wasm_f32x4_make(coefs[base0 + 1], coefs[base0 + 11],
                                          coefs[base0 + 21], coefs[base0 + 31]);
            /* Load contiguous spec */
            v128_t vsr = wasm_v128_load(frame_re + f);
            v128_t vsi = wasm_v128_load(frame_im + f);
            /* Load current accumulators */
            v128_t ver = wasm_v128_load(enh_re + f);
            v128_t vei = wasm_v128_load(enh_im + f);
            /* Complex MAC: o += s * c
               re += cr*sr - ci*si, im += cr*si + ci*sr */
            ver = wasm_f32x4_add(ver, wasm_f32x4_sub(
                wasm_f32x4_mul(vcr, vsr), wasm_f32x4_mul(vci, vsi)));
            vei = wasm_f32x4_add(vei, wasm_f32x4_add(
                wasm_f32x4_mul(vcr, vsi), wasm_f32x4_mul(vci, vsr)));
            wasm_v128_store(enh_re + f, ver);
            wasm_v128_store(enh_im + f, vei);
        }
#else
        for (int f = 0; f < NB_DF; f++) {
            float cr = coefs[f * 10 + n * 2];
            float ci = coefs[f * 10 + n * 2 + 1];
            float sr = frame_re[f];
            float si = frame_im[f];
            /* Complex multiply and accumulate: o += s * c */
            enh_re[f] += cr * sr - ci * si;
            enh_im[f] += cr * si + ci * sr;
        }
#endif
    }
    /* Frequencies NB_DF..FREQ_BINS-1 keep ERB-masked values (already set) */
}

/* ---- Post-filter (Rust lib.rs post_filter) ---- */

static void dfn3_post_filter(const float* noisy_re, const float* noisy_im,
                              float* enh_re, float* enh_im,
                              int n_freqs, float beta) {
    float beta_p1 = beta + 1.0f;
    float eps = 1e-12f;
    float pi = (float)M_PI;

    /* Process in chunks of 4, matching Rust chunks_exact(4) */
    int n4 = (n_freqs / 4) * 4;
    for (int i = 0; i < n4; i += 4) {
#if DFN3_SIMD
        /* Compute enh_mag² and noisy_mag² with SIMD */
        v128_t ver = wasm_v128_load(enh_re + i);
        v128_t vei = wasm_v128_load(enh_im + i);
        v128_t vnr = wasm_v128_load(noisy_re + i);
        v128_t vni = wasm_v128_load(noisy_im + i);
        v128_t enh_mag2 = wasm_f32x4_add(wasm_f32x4_mul(ver, ver), wasm_f32x4_mul(vei, vei));
        v128_t noisy_mag2 = wasm_f32x4_add(wasm_f32x4_mul(vnr, vnr), wasm_f32x4_mul(vni, vni));
        v128_t enh_mag = wasm_f32x4_sqrt(enh_mag2);
        v128_t noisy_mag = wasm_f32x4_sqrt(noisy_mag2);
        v128_t veps = wasm_f32x4_splat(eps);
        v128_t vone = wasm_f32x4_splat(1.0f);
        /* g = enh_mag / (noisy_mag + eps), clamped to [eps, 1] */
        v128_t vg = wasm_f32x4_div(enh_mag, wasm_f32x4_add(noisy_mag, veps));
        vg = wasm_f32x4_min(vg, vone);
        vg = wasm_f32x4_max(vg, veps);

        /* sinf is scalar — extract, compute, re-pack */
        float g[4], pf[4];
        g[0] = wasm_f32x4_extract_lane(vg, 0);
        g[1] = wasm_f32x4_extract_lane(vg, 1);
        g[2] = wasm_f32x4_extract_lane(vg, 2);
        g[3] = wasm_f32x4_extract_lane(vg, 3);

        for (int k = 0; k < 4; k++) {
            float gs = g[k] * sinf(g[k] * pi / 2.0f);
            float ratio = g[k] / gs;
            pf[k] = beta_p1 / (1.0f + beta * ratio * ratio);
        }

        /* Apply pf to enh_re, enh_im with SIMD */
        v128_t vpf = wasm_f32x4_make(pf[0], pf[1], pf[2], pf[3]);
        wasm_v128_store(enh_re + i, wasm_f32x4_mul(ver, vpf));
        wasm_v128_store(enh_im + i, wasm_f32x4_mul(vei, vpf));
#else
        float g[4], pf[4];

        for (int k = 0; k < 4; k++) {
            int idx = i + k;
            float enh_mag_s = sqrtf(enh_re[idx] * enh_re[idx] + enh_im[idx] * enh_im[idx]);
            float noisy_mag_s = sqrtf(noisy_re[idx] * noisy_re[idx] + noisy_im[idx] * noisy_im[idx]);

            g[k] = enh_mag_s / (noisy_mag_s + eps);
            if (g[k] > 1.0f) g[k] = 1.0f;
            if (g[k] < eps) g[k] = eps;
        }

        for (int k = 0; k < 4; k++) {
            float gs = g[k] * sinf(g[k] * pi / 2.0f);
            float ratio = g[k] / gs;
            pf[k] = beta_p1 / (1.0f + beta * ratio * ratio);
        }

        for (int k = 0; k < 4; k++) {
            int idx = i + k;
            enh_re[idx] *= pf[k];
            enh_im[idx] *= pf[k];
        }
#endif
    }
    /* Remaining elements (if n_freqs % 4 != 0) are left unchanged,
       matching Rust chunks_exact behavior */
}

/* ---- Helper: shift rolling buffer left by 1 (pop_front) ---- */
static void dfn3_rolling_shift(float buf[][2 * FREQ_BINS], int size) {
    memmove(buf[0], buf[1], (size - 1) * 2 * FREQ_BINS * sizeof(float));
}

/* ---- Helper: push spectrum to back of rolling buffer ---- */
static void dfn3_rolling_push(float buf[][2 * FREQ_BINS], int size,
                               const float* re, const float* im) {
    float* dst = buf[size - 1];
    memcpy(dst, re, FREQ_BINS * sizeof(float));
    memcpy(dst + FREQ_BINS, im, FREQ_BINS * sizeof(float));
}

/* ---- Helper: get re/im pointers from rolling buffer frame ---- */
static const float* dfn3_rbuf_re(const float buf[][2 * FREQ_BINS], int idx) {
    return buf[idx];
}
static const float* dfn3_rbuf_im(const float buf[][2 * FREQ_BINS], int idx) {
    return buf[idx] + FREQ_BINS;
}

/* ---- Main process function ---- */
/* Matches Rust tract.rs process() lines 509-641 exactly. */

void dfn3_process(DFN3State* st, const float* in, float* out) {
    /* 1. Silence detection (Rust: RMS check)
       Rust process() lines 513-525: rms = sum(x^2) / len, skip if < 1e-7 */
    {
        float e = dfn3_vdot(in, in, HOP_SIZE);
        float rms = e / (float)HOP_SIZE;

        if (rms < SILENCE_RMS_THRESH) {
            st->silence_skip_counter++;
        } else {
            st->silence_skip_counter = 0;
        }

        if (st->silence_skip_counter > SILENCE_SKIP_MAX) {
            /* Output silence, set LSNR to min */
            memset(out, 0, HOP_SIZE * sizeof(float));
            st->lsnr = (float)DFN3_LSNR_MIN;
            return;
        }
    }

    /* 2. Rust lines 531-532: pop_front both rolling buffers */
    dfn3_rolling_shift(st->rolling_spec_buf_y, SPEC_BUF_Y_SIZE);
    dfn3_rolling_shift(st->rolling_spec_buf_x, SPEC_BUF_X_SIZE);

    /* 3. STFT analysis (Rust lines 533-540) */
    dfn3_frame_analysis(st, in);

    /* 4. Push new spectrum to back of both rolling buffers (Rust lines 541-542) */
    dfn3_rolling_push(st->rolling_spec_buf_y, SPEC_BUF_Y_SIZE,
                      st->spec_re, st->spec_im);
    dfn3_rolling_push(st->rolling_spec_buf_x, SPEC_BUF_X_SIZE,
                      st->spec_re, st->spec_im);

    /* 5. Feature extraction — uses current frame spectrum (spec_re/im) */
    dfn3_extract_erb(st);
    dfn3_extract_spec(st);

    /* 6. Encoder (always runs — produces emb, lsnr, c0, e0-e3) */
    dfn3_encoder(st);

    /* 7. Stage selection based on LSNR (Rust: apply_stages, lines 658-672) */
    int apply_gains = 0;
    int apply_gain_zeros = 0;
    int apply_df = 0;

    if (st->lsnr < st->min_db_thresh) {
        apply_gain_zeros = 1;
    } else if (st->lsnr > st->max_db_erb_thresh) {
        /* Clean: no processing */
    } else if (st->lsnr > st->max_db_df_thresh) {
        apply_gains = 1;
    } else {
        apply_gains = 1;
        apply_df = 1;
    }

    /* 8. process_raw: run decoders (Rust lines 478-503) */
    int has_gains = 0;
    if (apply_gains) {
        dfn3_erb_decoder(st);
        has_gains = 1;
    } else if (apply_gain_zeros) {
        /* Zero mask — set erb_mask to all zeros */
        memset(st->erb_mask, 0, NB_ERB * sizeof(float));
        has_gains = 1;
    }
    /* else: gains = None, no decoder runs */

    int has_coefs = 0;
    if (apply_df) {
        dfn3_df_decoder(st);
        has_coefs = 1;
    }

    /* 9. Apply gains to rolling_spec_buf_y[df_order - 1]
       (Rust lines 551-583) */
    {
        int delayed_idx = DF_ORDER - 1;  /* = 4 */
        float* spec_y_re = st->rolling_spec_buf_y[delayed_idx];
        float* spec_y_im = st->rolling_spec_buf_y[delayed_idx] + FREQ_BINS;

        if (has_gains) {
            /* Apply ERB mask in-place to rolling_spec_buf_y[df_order-1]
               Rust: state.apply_mask(spec_ch, gain_slc) */
            int offset = 0;
            for (int b = 0; b < NB_ERB; b++) {
                float gain = st->erb_mask[b];
                int bw = ERB_WIDTHS[b];
#if DFN3_SIMD
                v128_t vg = wasm_f32x4_splat(gain);
                int i = 0;
                int bw4 = bw & ~3;
                for (; i < bw4; i += 4) {
                    int idx = offset + i;
                    wasm_v128_store(spec_y_re + idx, wasm_f32x4_mul(wasm_v128_load(spec_y_re + idx), vg));
                    wasm_v128_store(spec_y_im + idx, wasm_f32x4_mul(wasm_v128_load(spec_y_im + idx), vg));
                }
                for (; i < bw; i++) {
                    int idx = offset + i;
                    spec_y_re[idx] *= gain;
                    spec_y_im[idx] *= gain;
                }
#else
                for (int i = 0; i < bw; i++) {
                    int idx = offset + i;
                    spec_y_re[idx] *= gain;
                    spec_y_im[idx] *= gain;
                }
#endif
                offset += bw;
            }
            st->silence_skip_counter = 0;
        } else {
            /* gains = None → skipped due to LSNR > 30dB
               Rust line 582: self.skip_counter += 1 */
            st->silence_skip_counter++;
        }
    }

    /* 10. Prepare spec_buf for output (Rust lines 586-597):
       - Clone rolling_spec_buf_y[df_order-1] (ERB-processed spectrum, all 481 bins)
       - If has_coefs, apply DF to overwrite bins 0..nb_df using rolling_spec_buf_x */
    float enh_re[FREQ_BINS], enh_im[FREQ_BINS];
    {
        int delayed_idx = DF_ORDER - 1;
        memcpy(enh_re, st->rolling_spec_buf_y[delayed_idx], FREQ_BINS * sizeof(float));
        memcpy(enh_im, st->rolling_spec_buf_y[delayed_idx] + FREQ_BINS, FREQ_BINS * sizeof(float));
    }

    if (has_coefs) {
        /* Apply DF: reads from rolling_spec_buf_x, overwrites bins 0..nb_df of enh */
        dfn3_apply_df(st, enh_re, enh_im);
    }

    /* 11. Post-filter (Rust lines 617-623)
       Noisy reference = rolling_spec_buf_x[max(lookahead,df_order) - lookahead - 1]
                       = rolling_spec_buf_x[max(2,5) - 2 - 1] = rolling_spec_buf_x[2]
       Only applies when apply_erb (= apply_gains) && post_filter_beta > 0 */
    int noisy_idx = SPEC_BUF_X_SIZE - LOOKAHEAD - 1;  /* 5 - 2 - 1 = 2 */
    if (apply_gains && st->post_filter_beta > 0.0f) {
        const float* noisy_re = dfn3_rbuf_re(st->rolling_spec_buf_x, noisy_idx);
        const float* noisy_im = dfn3_rbuf_im(st->rolling_spec_buf_x, noisy_idx);
        dfn3_post_filter(noisy_re, noisy_im,
                         enh_re, enh_im,
                         FREQ_BINS, st->post_filter_beta);
    }

    /* 11b. Attenuation limit (Rust lines 625-629)
       Mix back noisy signal: enh = enh * (1-lim) + noisy * lim */
    if (st->atten_lim > 0.0f) {
        const float* noisy_re2 = dfn3_rbuf_re(st->rolling_spec_buf_x, noisy_idx);
        const float* noisy_im2 = dfn3_rbuf_im(st->rolling_spec_buf_x, noisy_idx);
        float one_m_lim = 1.0f - st->atten_lim;
        for (int i = 0; i < FREQ_BINS; i++) {
            enh_re[i] = enh_re[i] * one_m_lim + noisy_re2[i] * st->atten_lim;
            enh_im[i] = enh_im[i] * one_m_lim + noisy_im2[i] * st->atten_lim;
        }
    }

    /* 12. ISTFT synthesis (Rust lines 631-640) */
    dfn3_frame_synthesis(st, out, enh_re, enh_im);
}

#endif /* DFN3_IMPLEMENTATION */
#endif /* DFN3_H */
