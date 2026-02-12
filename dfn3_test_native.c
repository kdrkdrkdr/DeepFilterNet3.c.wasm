/*
 * DFN3 Native test — process WAV files through DFN3 and write output WAVs.
 * Compile: clang -O2 -o dfn3_test dfn3_test_native.c kiss_fft.c kiss_fftr.c -lm
 * Usage:   ./dfn3_test <weights.bin> <input_dir> <output_dir> [max_files]
 *
 * Input:  16kHz 16-bit mono WAV (DNS Challenge format)
 * Output: 48kHz 32-bit float WAV (DFN3 native output)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <time.h>

#define DFN3_IMPLEMENTATION
#include "dfn3.h"

/* ---- Minimal WAV reader/writer ---- */

typedef struct {
    int channels;
    int sample_rate;
    int bits_per_sample;
    int num_samples;
    void* data;
} WavFile;

static WavFile* wav_read(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return NULL; }

    unsigned int file_size; fread(&file_size, 4, 1, f);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return NULL; }

    int channels = 0, sample_rate = 0, bits = 0;
    unsigned int data_size = 0;
    void* data = NULL;

    while (!feof(f)) {
        char chunk_id[4];
        unsigned int chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            unsigned short fmt, ch;
            unsigned int sr, byte_rate;
            unsigned short block_align, bps;
            fread(&fmt, 2, 1, f);
            fread(&ch, 2, 1, f);
            fread(&sr, 4, 1, f);
            fread(&byte_rate, 4, 1, f);
            fread(&block_align, 2, 1, f);
            fread(&bps, 2, 1, f);
            channels = ch;
            sample_rate = sr;
            bits = bps;
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            data = malloc(data_size);
            fread(data, 1, data_size, f);
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);

    if (!data || channels == 0) {
        if (data) free(data);
        return NULL;
    }

    WavFile* w = (WavFile*)malloc(sizeof(WavFile));
    w->channels = channels;
    w->sample_rate = sample_rate;
    w->bits_per_sample = bits;
    w->num_samples = data_size / (channels * bits / 8);
    w->data = data;
    return w;
}

static void wav_free(WavFile* w) {
    if (w) { free(w->data); free(w); }
}

static int wav_write_f32(const char* path, const float* samples, int num_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    int channels = 1;
    int bits = 32;
    int byte_rate = sample_rate * channels * bits / 8;
    int block_align = channels * bits / 8;
    int data_size = num_samples * block_align;
    int file_size = 36 + data_size;

    /* IEEE float format = 3 */
    unsigned short fmt_tag = 3;
    unsigned short ch = channels;
    unsigned int sr = sample_rate;
    unsigned short ba = block_align;
    unsigned short bps = bits;

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    unsigned int fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&fmt_tag, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    unsigned int br = byte_rate;
    fwrite(&br, 4, 1, f);
    fwrite(&ba, 2, 1, f);
    fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f);
    unsigned int ds = data_size;
    fwrite(&ds, 4, 1, f);
    fwrite(samples, sizeof(float), num_samples, f);

    fclose(f);
    return 0;
}

/* ---- Simple 16k→48k linear interpolation resample ---- */
static float* resample_16k_to_48k(const short* in16, int n16, int* out_n48) {
    int n48 = n16 * 3;
    float* out = (float*)calloc(n48, sizeof(float));
    for (int i = 0; i < n16; i++) {
        float s = (float)in16[i] / 32768.0f;
        /* Simple 3x upsample with linear interpolation */
        float s_next = (i + 1 < n16) ? (float)in16[i + 1] / 32768.0f : s;
        out[i * 3]     = s;
        out[i * 3 + 1] = s + (s_next - s) * (1.0f / 3.0f);
        out[i * 3 + 2] = s + (s_next - s) * (2.0f / 3.0f);
    }
    *out_n48 = n48;
    return out;
}

/* ---- Process single file ---- */
static int process_file(DFN3State* st, const char* in_path, const char* out_path) {
    WavFile* wav = wav_read(in_path);
    if (!wav) {
        fprintf(stderr, "  ERROR: cannot read %s\n", in_path);
        return -1;
    }

    int n_in = wav->num_samples;
    int n48;
    float* samples48;

    if (wav->sample_rate == 16000 && wav->bits_per_sample == 16) {
        samples48 = resample_16k_to_48k((const short*)wav->data, n_in, &n48);
    } else if (wav->sample_rate == 48000 && wav->bits_per_sample == 16) {
        n48 = n_in;
        samples48 = (float*)calloc(n48, sizeof(float));
        const short* s = (const short*)wav->data;
        for (int i = 0; i < n48; i++) samples48[i] = (float)s[i] / 32768.0f;
    } else {
        fprintf(stderr, "  SKIP: unsupported format %dHz %dbit\n",
                wav->sample_rate, wav->bits_per_sample);
        wav_free(wav);
        return -1;
    }
    wav_free(wav);

    /* Process in 480-sample frames */
    int hop = 480;
    int n_frames = n48 / hop;
    int out_len = n_frames * hop;
    float* output48 = (float*)calloc(out_len, sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        dfn3_process(st, samples48 + f * hop, output48 + f * hop);
    }

    free(samples48);

    /* Write output */
    wav_write_f32(out_path, output48, out_len, 48000);
    free(output48);
    return 0;
}

/* ---- Load weights ---- */
static void* load_file(const char* path, int* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = (int)ftell(f);
    fseek(f, 0, SEEK_SET);
    void* data = malloc(*size);
    fread(data, 1, *size, f);
    fclose(f);
    return data;
}

/* ---- Sort helper ---- */
static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <weights.bin> <input_dir> <output_dir> [max_files]\n", argv[0]);
        return 1;
    }

    const char* weights_path = argv[1];
    const char* in_dir = argv[2];
    const char* out_dir = argv[3];
    int max_files = (argc > 4) ? atoi(argv[4]) : 999999;

    /* Load weights */
    int w_size;
    void* w_data = load_file(weights_path, &w_size);
    if (!w_data) {
        fprintf(stderr, "ERROR: cannot load weights: %s\n", weights_path);
        return 1;
    }

    /* Create DFN3 */
    DFN3State* st = dfn3_create(w_data);
    if (!st) {
        fprintf(stderr, "ERROR: dfn3_create failed\n");
        free(w_data);
        return 1;
    }

    /* Set default parameters (match worklet) */
    st->atten_lim = 0.0f;
    st->post_filter_beta = 0.0f;
    st->min_db_thresh = -10.0f;
    st->max_db_erb_thresh = 30.0f;
    st->max_db_df_thresh = 20.0f;

    /* Scan input directory */
    DIR* dir = opendir(in_dir);
    if (!dir) {
        fprintf(stderr, "ERROR: cannot open dir: %s\n", in_dir);
        dfn3_destroy(st);
        free(w_data);
        return 1;
    }

    char** files = NULL;
    int n_files = 0;
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        int len = strlen(ent->d_name);
        if (len > 4 && strcmp(ent->d_name + len - 4, ".wav") == 0) {
            files = (char**)realloc(files, (n_files + 1) * sizeof(char*));
            files[n_files++] = strdup(ent->d_name);
        }
    }
    closedir(dir);

    /* Sort */
    qsort(files, n_files, sizeof(char*), cmp_str);

    if (n_files > max_files) n_files = max_files;

    printf("Processing %d files from %s → %s\n", n_files, in_dir, out_dir);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int ok = 0, fail = 0;
    for (int i = 0; i < n_files; i++) {
        char in_path[1024], out_path[1024];
        snprintf(in_path, sizeof(in_path), "%s/%s", in_dir, files[i]);
        snprintf(out_path, sizeof(out_path), "%s/%s", out_dir, files[i]);

        /* Re-create state for each file (clean state) */
        dfn3_destroy(st);
        st = dfn3_create(w_data);
        st->atten_lim = 0.0f;
        st->post_filter_beta = 0.0f;
        st->min_db_thresh = -10.0f;
        st->max_db_erb_thresh = 30.0f;
        st->max_db_df_thresh = 20.0f;

        if (process_file(st, in_path, out_path) == 0) {
            ok++;
        } else {
            fail++;
        }

        if ((i + 1) % 100 == 0 || i == n_files - 1) {
            printf("  [%d/%d] ok=%d fail=%d\n", i + 1, n_files, ok, fail);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    printf("Done: %d ok, %d fail, %.1f seconds total\n", ok, fail, elapsed);

    /* Cleanup */
    dfn3_destroy(st);
    free(w_data);
    for (int i = 0; i < n_files; i++) free(files[i]);
    free(files);

    return 0;
}
