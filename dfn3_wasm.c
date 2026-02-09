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
    dfn3_process(g_state, g_input, g_output);
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
