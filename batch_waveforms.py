import math
import sys
import wave
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

SUFFIX = "_waveform"             # suffix for saved PNGs
MAX_POINTS = 200_000             # downsample to this many points for plotting


def read_wav(path: Path):
    """Read a wav file, normalising to float32 mono. Falls back to stdlib wave for unusual PCM formats."""
    try:
        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim == 2:  # stereo -> mono
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
        maxabs = float(np.max(np.abs(y))) if y.size else 1.0
        if maxabs > 1.5:  # likely integer scale
            y /= maxabs
        return y, int(sr)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)

        if sw == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 3:
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (a[:,0].astype(np.uint32) |
                      (a[:,1].astype(np.uint32) << 8) |
                      (a[:,2].astype(np.uint32) << 16)).astype(np.int32)
            signed[signed & 0x800000 != 0] -= 1 << 24
            data = signed.astype(np.float32) / (1 << 23)
        elif sw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / (1 << 31)
        else:
            raise RuntimeError(f"Unsupported sample width: {sw}")

        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        return data.astype(np.float32, copy=False), int(sr)


def plot_waveform(y: np.ndarray, sr: int, out_path: Path):
    """Plot waveform with downsampling."""
    if y.size == 0:
        return
    if y.size > MAX_POINTS:
        step = math.ceil(y.size / MAX_POINTS)
        y_plot = y[::step]
        t = np.arange(y_plot.size) * (step / sr)
    else:
        y_plot = y
        t = np.arange(y.size) / sr

    plt.figure(figsize=(12, 3.8))
    plt.plot(t, y_plot)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(out_path.stem.replace(SUFFIX, ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in {folder}.")
        return

    print(f"Found {len(wavs)} wav file(s).")
    for i, wav in enumerate(wavs, 1):
        try:
            y, sr = read_wav(wav)
            out_png = wav.with_name(wav.stem + SUFFIX + ".png")

            if out_png.exists():
                print(f"[{i}/{len(wavs)}] SKIP {wav.name}: {out_png.name} already exists")
                continue

            plot_waveform(y, sr, out_png)
            print(f"[{i}/{len(wavs)}] {wav.name} -> {out_png.name}")
        except Exception as e:
            print(f"[{i}/{len(wavs)}] ERROR {wav.name}: {e}")


if __name__ == "__main__":
    main()
