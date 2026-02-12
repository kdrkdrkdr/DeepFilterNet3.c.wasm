#!/usr/bin/env python3
"""
Compare DFN3 outputs: original vs optimized.
Reads 48kHz float32 WAV outputs from both versions, computes per-file SNR.

Usage:
  python3 compare_snr.py <orig_dir> <opt_dir> [max_files]

Output:
  Per-file SNR (dB), summary statistics.
  SNR = 10 * log10(sum(orig^2) / sum((orig - opt)^2))
  Higher = more similar. >40 dB = bit-near-identical, >15 dB = acceptable.
"""
import os
import sys
import struct
import numpy as np

def read_wav_f32(path):
    """Read 32-bit float WAV, return numpy array."""
    with open(path, 'rb') as f:
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not RIFF: {path}")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not WAVE: {path}")

        fmt_tag = 0
        data = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                fmt_tag = struct.unpack('<H', fmt_data[0:2])[0]
            elif chunk_id == b'data':
                data = f.read(chunk_size)
            else:
                f.read(chunk_size)

    if data is None:
        raise ValueError(f"No data chunk: {path}")

    if fmt_tag == 3:  # IEEE float
        return np.frombuffer(data, dtype=np.float32)
    elif fmt_tag == 1:  # PCM int16
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unknown fmt_tag={fmt_tag}: {path}")


def compute_snr(ref, test):
    """Compute SNR in dB between reference and test signals."""
    n = min(len(ref), len(test))
    ref = ref[:n]
    test = test[:n]
    sig_power = np.sum(ref ** 2)
    noise_power = np.sum((ref - test) ** 2)
    if noise_power < 1e-30:
        return 999.0  # essentially identical
    if sig_power < 1e-30:
        return 0.0
    return 10.0 * np.log10(sig_power / noise_power)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <orig_dir> <opt_dir> [max_files]")
        sys.exit(1)

    orig_dir = sys.argv[1]
    opt_dir = sys.argv[2]
    max_files = int(sys.argv[3]) if len(sys.argv) > 3 else 999999

    # Find common files
    orig_files = set(f for f in os.listdir(orig_dir) if f.endswith('.wav'))
    opt_files = set(f for f in os.listdir(opt_dir) if f.endswith('.wav'))
    common = sorted(orig_files & opt_files)[:max_files]

    print(f"Comparing {len(common)} files: {orig_dir} vs {opt_dir}")
    print(f"{'File':<20} {'SNR(dB)':>10} {'MaxDiff':>12} {'MeanDiff':>12} {'RMS_orig':>10}")
    print("-" * 70)

    snrs = []
    max_diffs = []
    low_snr_files = []

    for fname in common:
        try:
            orig = read_wav_f32(os.path.join(orig_dir, fname))
            opt = read_wav_f32(os.path.join(opt_dir, fname))
            n = min(len(orig), len(opt))
            orig = orig[:n]
            opt = opt[:n]

            snr = compute_snr(orig, opt)
            diff = np.abs(orig - opt)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rms_orig = np.sqrt(np.mean(orig ** 2))

            snrs.append(snr)
            max_diffs.append(max_diff)

            marker = ""
            if snr < 15.0:
                marker = " *** LOW ***"
                low_snr_files.append((fname, snr))
            elif snr < 30.0:
                marker = " * warn *"

            print(f"{fname:<20} {snr:>10.1f} {max_diff:>12.6f} {mean_diff:>12.8f} {rms_orig:>10.4f}{marker}")

        except Exception as e:
            print(f"{fname:<20} ERROR: {e}")

    print("-" * 70)
    if snrs:
        snrs = np.array(snrs)
        print(f"\n=== Summary ({len(snrs)} files) ===")
        print(f"  Mean SNR:   {np.mean(snrs):.1f} dB")
        print(f"  Median SNR: {np.median(snrs):.1f} dB")
        print(f"  Min SNR:    {np.min(snrs):.1f} dB")
        print(f"  Max SNR:    {np.max(snrs):.1f} dB")
        print(f"  Std SNR:    {np.std(snrs):.1f} dB")
        print(f"  Files ≥ 15 dB: {np.sum(snrs >= 15)}/{len(snrs)} ({100*np.sum(snrs >= 15)/len(snrs):.1f}%)")
        print(f"  Files ≥ 30 dB: {np.sum(snrs >= 30)}/{len(snrs)} ({100*np.sum(snrs >= 30)/len(snrs):.1f}%)")
        print(f"  Files ≥ 40 dB: {np.sum(snrs >= 40)}/{len(snrs)} ({100*np.sum(snrs >= 40)/len(snrs):.1f}%)")
        print(f"  Max abs diff (across all files): {np.max(max_diffs):.6f}")

        if low_snr_files:
            print(f"\n  *** {len(low_snr_files)} files below 15 dB SNR: ***")
            for fname, snr in low_snr_files[:10]:
                print(f"    {fname}: {snr:.1f} dB")
    else:
        print("No files compared.")


if __name__ == '__main__':
    main()
