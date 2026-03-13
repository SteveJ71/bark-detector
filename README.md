# Bark Detector

A Python tool for real-time dog bark detection and logging using the [YAMNet](https://tfhub.dev/google/yamnet/1) audio classification model. Listens continuously to a microphone, detects and logs bark events, saves audio recordings of bark sessions, and includes scripts for visualising the data.

## How it works

- Captures audio from a microphone in 1.2-second chunks at 16kHz
- Classifies each chunk using YAMNet (a TensorFlow audio model)
- Detects bark events by volume threshold and class confidence
- Groups individual barks into sessions separated by quiet gaps
- Logs bark events and sessions to a daily text file
- Saves a WAV recording for qualifying sessions (at most once per hour)

## Scripts

| Script | Description |
|---|---|
| `bark_session_logger.py` | Main script — runs real-time detection and logging |
| `diary_writer.py` | Parses log files and writes a human-readable daily summary |
| `diary_plots.py` | Generates heatmap and bar charts from log files |
| `plot_bark_graph.py` | Plots bark frequency over time from log files |
| `batch_waveforms.py` | Generates waveform PNG images from WAV recordings |

## Requirements

- Python 3.10–3.13 (TensorFlow does not yet support Python 3.14+)
- A working microphone
- Git
- ffmpeg (see below)
- Microsoft Visual C++ Redistributable (see below)

## Installing Microsoft Visual C++ Redistributable (Windows)

TensorFlow requires this to be installed on Windows. If you see a DLL load error when running the script, this is the cause.

1. Search Google for **"Microsoft Visual C++ Redistributable latest"** and go to the Microsoft download page
2. Download and install the **x64** version
3. Restart your computer and try again

## Installing ffmpeg (Windows)

1. Go to [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and download **ffmpeg-release-essentials.zip**
2. Extract the zip to a permanent location, e.g. `C:\ffmpeg`
3. Add the `bin` folder to your system PATH:
   - Search Windows for **"Edit the system environment variables"**
   - Click **Environment Variables**
   - Under **System variables**, select **Path** → **Edit**
   - Click **New** and add the path to the bin folder, e.g. `C:\ffmpeg\ffmpeg-7.x-essentials_build\bin`
   - Click OK on all dialogs
4. Open a new terminal and verify:
   ```bash
   ffmpeg -version
   ```

## Installation

```bash
git clone https://github.com/SteveJ71/bark-detector.git
cd bark-detector
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Real-time bark detection

```bash
python bark_session_logger.py
```

Press `Ctrl+C` to stop. Logs are written to `bark_log_YYYY-MM-DD.txt` in the same folder. WAV recordings are saved as `bark_audio_YYYY-MM-DD_HH-MM-SS.wav`.

### Daily diary summary

```bash
python diary_writer.py "path/to/log/files"
```

Writes `bark_diary_summary.txt` into the specified folder.

### Heatmap and bar charts

```bash
python diary_plots.py "path/to/logs/2025" "path/to/logs/2026"
```

Saves plots as PNG files in a `Plots/` subfolder. Change `HEATMAP_METRIC` at the top of the file between `"minutes"` and `"barks"` to switch what is graphed.

### Bark frequency graph

```bash
python plot_bark_graph.py "path/to/log/files"
```

Displays an interactive plot of bark frequency over time across all logged days.

### Waveform images

```bash
python batch_waveforms.py "path/to/wav/files"
```

Generates a `_waveform.png` alongside each WAV file. Skips files that already have a PNG.

## Configuration

Key constants in `bark_session_logger.py` can be adjusted to suit your environment:

| Constant | Default | Description |
|---|---|---|
| `MIN_BARK_VOLUME` | `0.012` | Minimum audio volume to count as a bark — increase to filter out distant dogs |
| `SESSION_GAP_SECONDS` | `90` | Quiet gap (seconds) that ends a bark session |
| `MIN_SESSION_DURATION` | `20` | Minimum session length (seconds) to be logged |
| `MIN_BARKS_PER_SESSION` | `10` | Minimum bark count for a session to be logged |
| `RECORD_DURATION` | `240` | Length of WAV recording saved per session (seconds) |
| `RECORD_INTERVAL` | `3600` | Minimum time between WAV recordings (seconds) |

## Output files

| File | Description |
|---|---|
| `bark_log_YYYY-MM-DD.txt` | Daily log of individual barks and sessions |
| `bark_audio_YYYY-MM-DD_HH-MM-SS.wav` | Audio recording of a bark session |
| `bark_diary_summary.txt` | Human-readable summary of bark sessions |
| `Plots/bark_heatmap_*.png` | Heatmap of barking by day and hour |
| `Plots/bark_daily_totals_*.png` | Bar chart of daily barking totals |
| `Plots/bark_hourly_profile_*.png` | Bar chart of barking by hour across all days |
