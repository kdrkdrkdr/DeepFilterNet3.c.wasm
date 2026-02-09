#!/usr/bin/env python3
"""
DeepFilterNet3 ONNX → C binary weight extractor.

Extracts all weights from enc.onnx, erb_dec.onnx, df_dec.onnx
and writes them as:
  1. A single binary blob (dfn3_weights.bin)
  2. A C header with offsets/shapes (dfn3_weights.h)

Weight naming follows the actual layer structure from the ONNX graph.
"""

import os
from collections import OrderedDict

import numpy as np
import onnx
from onnx import numpy_helper

ONNX_DIR = "/tmp/dfn3_onnx/tmp/export"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map onnx::Conv_XXX / onnx::GRU_XXX to meaningful names based on graph analysis
ENC_NAME_MAP = {
    # erb_conv0: Conv2d(1,64,3,3) group=1  [node 17]
    "onnx::Conv_282": "enc_erb_conv0_dw_w",
    "onnx::Conv_283": "enc_erb_conv0_dw_b",
    # erb_conv1: DW [node 19] + PW [node 20]
    "erb_conv1.0.weight": "enc_erb_conv1_dw_w",
    "onnx::Conv_285": "enc_erb_conv1_pw_w",
    "onnx::Conv_286": "enc_erb_conv1_pw_b",
    # erb_conv2: DW [node 22] + PW [node 23]
    "erb_conv2.0.weight": "enc_erb_conv2_dw_w",
    "onnx::Conv_288": "enc_erb_conv2_pw_w",
    "onnx::Conv_289": "enc_erb_conv2_pw_b",
    # erb_conv3: DW [node 25] + PW [node 26]
    "erb_conv3.0.weight": "enc_erb_conv3_dw_w",
    "onnx::Conv_291": "enc_erb_conv3_pw_w",
    "onnx::Conv_292": "enc_erb_conv3_pw_b",
    # df_conv0: DW group=2 [node 45] + PW [node 46]
    "df_conv0.1.weight": "enc_df_conv0_dw_w",
    "onnx::Conv_294": "enc_df_conv0_pw_w",
    "onnx::Conv_295": "enc_df_conv0_pw_b",
    # df_conv1: DW [node 48] + PW [node 49]
    "df_conv1.0.weight": "enc_df_conv1_dw_w",
    "onnx::Conv_297": "enc_df_conv1_pw_w",
    "onnx::Conv_298": "enc_df_conv1_pw_b",
    # grouped linear layers
    "df_fc_emb.0.weight": "enc_df_fc_emb_w",
    "emb_gru.linear_in.0.weight": "enc_emb_gru_lin_in_w",
    "emb_gru.linear_out.0.weight": "enc_emb_gru_lin_out_w",
    # lsnr
    "lsnr_fc.0.bias": "enc_lsnr_fc_b",
    "onnx::MatMul_337": "enc_lsnr_fc_w",
}

ERB_NAME_MAP = {
    # grouped linear
    "emb_gru.linear_in.0.weight": "erb_emb_gru_lin_in_w",
    "emb_gru.linear_out.0.weight": "erb_emb_gru_lin_out_w",
    # conv3p: pointwise 1x1 (skip projection)
    "onnx::Conv_288": "erb_conv3p_dw_w",
    "onnx::Conv_289": "erb_conv3p_dw_b",
    # convt3: depthwise 1x3 + pointwise 1x1
    "convt3.0.weight": "erb_convt3_dw_w",
    "onnx::Conv_291": "erb_convt3_pw_w",
    "onnx::Conv_292": "erb_convt3_pw_b",
    # conv2p: pointwise (skip)
    "onnx::Conv_294": "erb_conv2p_dw_w",
    "onnx::Conv_295": "erb_conv2p_dw_b",
    # convt2: transposed conv + pointwise
    "convt2.0.weight": "erb_convt2_dw_w",
    "onnx::Conv_297": "erb_convt2_pw_w",
    "onnx::Conv_298": "erb_convt2_pw_b",
    # conv1p: pointwise (skip)
    "onnx::Conv_300": "erb_conv1p_dw_w",
    "onnx::Conv_301": "erb_conv1p_dw_b",
    # convt1: transposed conv + pointwise
    "convt1.0.weight": "erb_convt1_dw_w",
    "onnx::Conv_303": "erb_convt1_pw_w",
    "onnx::Conv_304": "erb_convt1_pw_b",
    # conv0p: pointwise (skip)
    "onnx::Conv_306": "erb_conv0p_dw_w",
    "onnx::Conv_307": "erb_conv0p_dw_b",
    # conv0_out: final output conv
    "onnx::Conv_309": "erb_conv0_out_w",
    "onnx::Conv_310": "erb_conv0_out_b",
}

DF_NAME_MAP = {
    # df_convp: depthwise temporal + pointwise
    "df_convp.1.weight": "df_convp_dw_w",
    "onnx::Conv_268": "df_convp_pw_w",
    "onnx::Conv_269": "df_convp_pw_b",
    # grouped linear
    "df_gru.linear_in.0.weight": "df_gru_lin_in_w",
    "df_skip.weight": "df_skip_w",
    "df_out.0.weight": "df_out_w",
    # alpha
    "df_fc_a.0.bias": "df_fc_a_b",
    "onnx::MatMul_321": "df_fc_a_w",
    # GRU layer 0
    "onnx::GRU_291": "df_gru0_W",
    "onnx::GRU_292": "df_gru0_R",
    "onnx::GRU_293": "df_gru0_B",
    # GRU layer 1
    "onnx::GRU_311": "df_gru1_W",
    "onnx::GRU_312": "df_gru1_R",
    "onnx::GRU_313": "df_gru1_B",
}


def extract_named_weights(model, name_map, prefix_fallback):
    """Extract weights from initializers with proper naming."""
    weights = OrderedDict()
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        name = name_map.get(init.name, f"{prefix_fallback}_{init.name.replace('.', '_').replace('::', '_')}")
        weights[name] = arr
    return weights


def extract_constant_gru_weights(model, prefix):
    """Extract GRU W/R/B from Constant nodes feeding GRU ops."""
    gru_input_names = ["X", "W", "R", "B", "seq_lens", "init_h"]
    gru_feed = {}

    gru_idx = 0
    for node in model.graph.node:
        if node.op_type == "GRU":
            suffix = f"{gru_idx}"
            for i, inp in enumerate(node.input):
                if inp and i < len(gru_input_names) and gru_input_names[i] in ("W", "R", "B"):
                    gru_feed[inp] = f"{prefix}_gru{suffix}_{gru_input_names[i]}"
            gru_idx += 1

    weights = OrderedDict()
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] in gru_feed:
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    weights[gru_feed[node.output[0]]] = arr
    return weights


def main():
    all_weights = OrderedDict()

    # --- Encoder ---
    enc = onnx.load(os.path.join(ONNX_DIR, "enc.onnx"))
    all_weights.update(extract_named_weights(enc, ENC_NAME_MAP, "enc"))
    all_weights.update(extract_constant_gru_weights(enc, "enc_emb"))

    # --- ERB Decoder ---
    erb = onnx.load(os.path.join(ONNX_DIR, "erb_dec.onnx"))
    all_weights.update(extract_named_weights(erb, ERB_NAME_MAP, "erb"))
    all_weights.update(extract_constant_gru_weights(erb, "erb_emb"))

    # --- DF Decoder ---
    df = onnx.load(os.path.join(ONNX_DIR, "df_dec.onnx"))
    all_weights.update(extract_named_weights(df, DF_NAME_MAP, "df"))
    # df_dec GRU weights are already in initializers, no need for constant extraction

    # --- Write binary blob ---
    bin_path = os.path.join(OUT_DIR, "dfn3_weights.bin")
    offset = 0
    entries = []

    with open(bin_path, "wb") as f:
        for name, arr in all_weights.items():
            flat = arr.flatten()
            f.write(flat.tobytes())
            entries.append((name, offset, flat.size, list(arr.shape)))
            offset += flat.size * 4

    total_bytes = offset
    total_params = sum(e[2] for e in entries)
    print(f"Wrote {bin_path}: {total_bytes:,} bytes ({total_params:,} params)")

    # --- Write C header ---
    h_path = os.path.join(OUT_DIR, "dfn3_weights.h")
    with open(h_path, "w") as f:
        f.write("/* Auto-generated by extract_weights.py - DO NOT EDIT */\n")
        f.write("#ifndef DFN3_WEIGHTS_H\n")
        f.write("#define DFN3_WEIGHTS_H\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write(f"#define DFN3_WEIGHTS_SIZE {total_bytes}\n")
        f.write(f"#define DFN3_TOTAL_PARAMS {total_params}\n\n")

        # Model config constants
        f.write("/* Model configuration from config.ini */\n")
        f.write("#define DFN3_SR          48000\n")
        f.write("#define DFN3_FFT_SIZE    960\n")
        f.write("#define DFN3_HOP_SIZE    480\n")
        f.write("#define DFN3_FREQ_BINS   481   /* FFT_SIZE/2 + 1 */\n")
        f.write("#define DFN3_NB_ERB      32\n")
        f.write("#define DFN3_NB_DF       96\n")
        f.write("#define DFN3_DF_ORDER    5\n")
        f.write("#define DFN3_DF_LOOKAHEAD 2\n")
        f.write("#define DFN3_CONV_CH     64\n")
        f.write("#define DFN3_EMB_DIM     512\n")
        f.write("#define DFN3_EMB_HIDDEN  256\n")
        f.write("#define DFN3_DF_HIDDEN   256\n")
        f.write("#define DFN3_LSNR_MAX    35\n")
        f.write("#define DFN3_LSNR_MIN    (-15)\n")
        f.write("#define DFN3_NORM_TAU    1.0f\n")
        f.write("#define DFN3_CONV_LOOKAHEAD 2\n")
        f.write("#define DFN3_ENC_LINEAR_GROUPS 32\n")
        f.write("#define DFN3_LINEAR_GROUPS 16\n")
        f.write("#define DFN3_EMB_NUM_LAYERS 3\n")
        f.write("#define DFN3_DF_NUM_LAYERS 2\n\n")

        f.write("typedef struct {\n")
        f.write("    const float* data;\n")
        f.write("    int ndim;\n")
        f.write("    int shape[4];\n")
        f.write("    int numel;\n")
        f.write("} DFN3Tensor;\n\n")

        # Group weights by section
        f.write("typedef struct {\n")
        f.write("    /* === Encoder === */\n")
        section = "enc"
        for name, off, numel, shape in entries:
            new_section = name.split("_")[0]
            if new_section != section:
                section = new_section
                label = {"erb": "ERB Decoder", "df": "DF Decoder"}.get(section, section)
                f.write(f"\n    /* === {label} === */\n")
            f.write(f"    DFN3Tensor {name};  /* {shape} */\n")
        f.write("} DFN3Weights;\n\n")

        f.write("static inline void dfn3_weights_init(DFN3Weights* w, const float* base) {\n")
        for name, off, numel, shape in entries:
            ndim = len(shape)
            shape_str = ", ".join(str(s) for s in shape)
            if ndim < 4:
                shape_str += ", " + ", ".join(["0"] * (4 - ndim))
            f.write(
                f"    w->{name} = (DFN3Tensor){{base + {off // 4}, {ndim}, {{{shape_str}}}, {numel}}};\n"
            )
        f.write("}\n\n")

        f.write("#endif /* DFN3_WEIGHTS_H */\n")

    print(f"Wrote {h_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"WEIGHT SUMMARY: {len(entries)} tensors, {total_params:,} params, {total_bytes/1024/1024:.2f} MB")
    print(f"{'='*70}")
    section = ""
    for name, off, numel, shape in entries:
        new_sec = name.split("_")[0]
        if new_sec != section:
            section = new_sec
            print(f"\n  [{section.upper()}]")
        print(f"    {name:40s} {str(shape):25s} {numel:>10,}")


if __name__ == "__main__":
    main()
