/*
 * DeepFilterNet3 — Emscripten WASM entry point.
 *
 * Exported functions for AudioWorklet:
 *   dfn3_wasm_create(weights_ptr, weights_len)
 *   dfn3_wasm_get_input_ptr()    — pointer to 480-float input buffer
 *   dfn3_wasm_get_output_ptr()   — pointer to 480-float output buffer (48 kHz)
 *   dfn3_wasm_process()          — process one frame
 *   dfn3_wasm_destroy()
 */

#define DFN3_IMPLEMENTATION
#include "dfn3.h"
#include "webrtc_agc.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

static DFN3State* g_state = NULL;

/* Buffers accessible from JS via pointers */
static float g_input[480];   /* 480 samples @ 48 kHz */
static float g_output[480];  /* 480 samples @ 48 kHz (enhanced) */

/* ---- 2nd-order Butterworth High-pass Filter (80 Hz @ 48 kHz) ----
 *
 * Precomputed coefficients for cutoff = 80 Hz, sample rate = 48000 Hz.
 * Transfer function (Direct Form II Transposed):
 *   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 *
 * Python verification:
 *   from scipy.signal import butter
 *   b, a = butter(2, 80/24000, btype='high')
 */
#define HPF_B0  0.99482656f
#define HPF_B1 -1.98965312f
#define HPF_B2  0.99482656f
#define HPF_A1 -1.98964639f
#define HPF_A2  0.98965985f

static float g_hpf_x1 = 0.0f, g_hpf_x2 = 0.0f;  /* input history  */
static float g_hpf_y1 = 0.0f, g_hpf_y2 = 0.0f;  /* output history */
static int   g_hpf_enabled = 1;  /* on by default */

static void hpf_process(float* buf, int len) {
    float x1 = g_hpf_x1, x2 = g_hpf_x2;
    float y1 = g_hpf_y1, y2 = g_hpf_y2;
    for (int i = 0; i < len; i++) {
        float x0 = buf[i];
        float y0 = HPF_B0*x0 + HPF_B1*x1 + HPF_B2*x2 - HPF_A1*y1 - HPF_A2*y2;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
        buf[i] = y0;
    }
    g_hpf_x1 = x1; g_hpf_x2 = x2;
    g_hpf_y1 = y1; g_hpf_y2 = y2;
}

/* WebRTC AGC instances (input + output) */
static WebRtcAgcState g_agc_in;
static WebRtcAgcState g_agc_out;
static int g_agc_input_enabled = 0;
static int g_agc_output_enabled = 0;

EXPORT int dfn3_wasm_create(const void* weights_ptr, int weights_len) {
    (void)weights_len;
    if (g_state) {
        dfn3_destroy(g_state);
    }
    g_state = dfn3_create(weights_ptr);
    return g_state ? 0 : -1;
}

EXPORT float* dfn3_wasm_get_input_ptr(void) {
    return g_input;
}

EXPORT float* dfn3_wasm_get_output_ptr(void) {
    return g_output;
}

EXPORT int dfn3_wasm_get_input_size(void) {
    return 480;
}

EXPORT int dfn3_wasm_get_output_size(void) {
    return 480;
}

EXPORT void dfn3_wasm_process(void) {
    if (!g_state) return;
    if (g_hpf_enabled)        hpf_process(g_input, 480);
    if (g_agc_input_enabled)  webrtc_agc_process(&g_agc_in,  g_input,  480);
    dfn3_process(g_state, g_input, g_output);
    if (g_agc_output_enabled) webrtc_agc_process(&g_agc_out, g_output, 480);
}

EXPORT float dfn3_wasm_get_lsnr(void) {
    return g_state ? g_state->lsnr : -999.0f;
}

/* Runtime parameter setters */

EXPORT void dfn3_wasm_set_atten_lim(float db) {
    if (!g_state) return;
    float lim = fabsf(db);
    if (lim >= 100.0f) {
        g_state->atten_lim = 0.0f;  /* no limit */
    } else if (lim < 0.01f) {
        g_state->atten_lim = 1.0f;  /* full passthrough (no noise reduction) */
    } else {
        g_state->atten_lim = powf(10.0f, -lim / 20.0f);
    }
}

EXPORT void dfn3_wasm_set_post_filter_beta(float beta) {
    if (!g_state) return;
    g_state->post_filter_beta = beta > 0.0f ? beta : 0.0f;
}

EXPORT void dfn3_wasm_set_min_db_thresh(float val) {
    if (!g_state) return;
    g_state->min_db_thresh = val;
}

EXPORT void dfn3_wasm_set_max_db_erb_thresh(float val) {
    if (!g_state) return;
    g_state->max_db_erb_thresh = val;
}

EXPORT void dfn3_wasm_set_max_db_df_thresh(float val) {
    if (!g_state) return;
    g_state->max_db_df_thresh = val;
}

EXPORT void dfn3_wasm_destroy(void) {
    if (g_state) {
        dfn3_destroy(g_state);
        g_state = NULL;
    }
}

/* ---- WebRTC AGC control ---- */

EXPORT void dfn3_wasm_agc_init(void) {
    webrtc_agc_init(&g_agc_in);
    webrtc_agc_init(&g_agc_out);
}

EXPORT void dfn3_wasm_set_input_agc(int enabled) {
    g_agc_input_enabled = enabled ? 1 : 0;
}

EXPORT void dfn3_wasm_set_output_agc(int enabled) {
    g_agc_output_enabled = enabled ? 1 : 0;
}

EXPORT void dfn3_wasm_set_agc_target(int level_dbfs) {
    webrtc_agc_set_target_level_dbfs(&g_agc_in, level_dbfs);
    webrtc_agc_set_target_level_dbfs(&g_agc_out, level_dbfs);
}

EXPORT void dfn3_wasm_set_agc_compression(int gain_db) {
    webrtc_agc_set_compression_gain_db(&g_agc_in, gain_db);
    webrtc_agc_set_compression_gain_db(&g_agc_out, gain_db);
}

EXPORT void dfn3_wasm_set_input_agc_compression(int gain_db) {
    webrtc_agc_set_compression_gain_db(&g_agc_in, gain_db);
}

EXPORT void dfn3_wasm_set_output_agc_compression(int gain_db) {
    webrtc_agc_set_compression_gain_db(&g_agc_out, gain_db);
}

/* ---- High-pass filter control ---- */

EXPORT void dfn3_wasm_set_hpf(int enabled) {
    g_hpf_enabled = enabled ? 1 : 0;
    if (enabled) {
        /* Reset filter state to avoid transients */
        g_hpf_x1 = g_hpf_x2 = 0.0f;
        g_hpf_y1 = g_hpf_y2 = 0.0f;
    }
}
