import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Settings
START_WINDOW = time(8, 0, 0)
END_WINDOW = time(20, 0, 0)
MIN_BARKS_THRESHOLD = 20

# What to graph:
#   "minutes" = total barking minutes in each hour block
#   "barks"   = estimated bark count in each hour block
HEATMAP_METRIC = "minutes"

session_pattern = re.compile(
    r"\[BARK SESSION\] From (?P<start>[\d\-: ]+) to (?P<end>[\d\-: ]+)"
    r" \(Duration: (?P<duration>[\d.]+) seconds, (?P<barks>\d+) barks\)"
)


def daterange_split_by_hour(start_dt, end_dt):
    """Split a session into hour-sized chunks.
    Yields: (date, hour, seconds_in_that_hour)
    """
    current = start_dt
    while current < end_dt:
        next_hour = current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        chunk_end = min(end_dt, next_hour)
        yield current.date(), current.hour, (chunk_end - current).total_seconds()
        current = chunk_end


def clip_to_window(start_dt, end_dt, start_window, end_window):
    """Clip a session to the daily graph window.
    Returns (clipped_start, clipped_end) or (None, None) if outside window.
    """
    day = start_dt.date()
    window_start = datetime.combine(day, start_window)
    window_end = datetime.combine(day, end_window)
    clipped_start = max(start_dt, window_start)
    clipped_end = min(end_dt, window_end)
    if clipped_start >= clipped_end:
        return None, None
    return clipped_start, clipped_end


def load_data(log_dirs):
    """Read all bark log files from the given directories and return aggregated grids."""
    minutes_grid = defaultdict(lambda: defaultdict(float))
    barks_grid = defaultdict(lambda: defaultdict(float))
    daily_minutes = defaultdict(float)
    daily_barks = defaultdict(float)
    hourly_minutes = defaultdict(float)
    hourly_barks = defaultdict(float)
    log_file_dates = set()

    for log_dir in log_dirs:
        if not log_dir.is_dir():
            print(f"Skipping missing folder: {log_dir}")
            continue

        for filename in sorted(os.listdir(log_dir)):
            if not filename.startswith("bark_log_") or not filename.endswith(".txt"):
                continue

            filepath = log_dir / filename
            print(f"Processing {filepath}")

            try:
                date_str = filename.replace("bark_log_", "").replace(".txt", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                log_file_dates.add(file_date)
            except ValueError:
                print(f"Skipping file with unexpected name: {filename}")
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    match = session_pattern.search(line)
                    if not match:
                        continue

                    start_dt = datetime.strptime(match.group("start"), "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(match.group("end"), "%Y-%m-%d %H:%M:%S")
                    total_barks = int(match.group("barks"))

                    if total_barks <= MIN_BARKS_THRESHOLD:
                        continue

                    clipped_start, clipped_end = clip_to_window(start_dt, end_dt, START_WINDOW, END_WINDOW)
                    if clipped_start is None:
                        continue

                    full_duration = (end_dt - start_dt).total_seconds()
                    clipped_duration = (clipped_end - clipped_start).total_seconds()
                    if full_duration <= 0 or clipped_duration <= 0:
                        continue

                    barks_per_second = total_barks / full_duration

                    for chunk_date, chunk_hour, chunk_seconds in daterange_split_by_hour(clipped_start, clipped_end):
                        chunk_minutes = chunk_seconds / 60.0
                        chunk_barks = chunk_seconds * barks_per_second

                        minutes_grid[chunk_date][chunk_hour] += chunk_minutes
                        barks_grid[chunk_date][chunk_hour] += chunk_barks
                        daily_minutes[chunk_date] += chunk_minutes
                        daily_barks[chunk_date] += chunk_barks
                        hourly_minutes[chunk_hour] += chunk_minutes
                        hourly_barks[chunk_hour] += chunk_barks

    return minutes_grid, barks_grid, daily_minutes, daily_barks, hourly_minutes, hourly_barks, log_file_dates


def plot_heatmap(matrix, plot_dates, hours, output_dir):
    metric_label = "Estimated Barks" if HEATMAP_METRIC == "barks" else "Minutes Barking"
    fig_height = max(8, len(plot_dates) * 0.28)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    cmap = matplotlib.colormaps["YlOrRd"].copy()
    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        origin="upper",
        vmin=0,
        vmax=np.nanmax(matrix)
    )

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")

    if len(plot_dates) <= 40:
        y_positions = range(len(plot_dates))
    else:
        step = max(1, len(plot_dates) // 30)
        y_positions = list(range(0, len(plot_dates), step))

    ax.set_yticks(y_positions)
    ax.set_yticklabels([plot_dates[i].strftime("%d %b %Y") for i in y_positions])
    ax.set_title(f"Dog Barking Heat Map by Day and Hour ({metric_label})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Date")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    plt.tight_layout()
    out = output_dir / f"bark_heatmap_{HEATMAP_METRIC}.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    return out


def plot_daily_totals(plot_dates, daily_barks, daily_minutes, output_dir):
    metric_label = "Estimated Barks" if HEATMAP_METRIC == "barks" else "Minutes Barking"
    daily_values = [
        daily_barks[d] if HEATMAP_METRIC == "barks" else daily_minutes[d]
        for d in plot_dates
    ]
    daily_labels = [d.strftime("%d %b %Y") for d in plot_dates]

    fig_width = max(14, len(plot_dates) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(daily_labels, daily_values)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"Daily Dog Barking Totals ({metric_label})")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = output_dir / f"bark_daily_totals_{HEATMAP_METRIC}.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    return out


def plot_hourly_profile(hours, hourly_barks, hourly_minutes, output_dir):
    metric_label = "Estimated Barks" if HEATMAP_METRIC == "barks" else "Minutes Barking"
    hourly_values = [
        hourly_barks[h] if HEATMAP_METRIC == "barks" else hourly_minutes[h]
        for h in hours
    ]
    hourly_labels = [f"{h:02d}:00" for h in hours]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly_labels, hourly_values)
    ax.set_title(f"Barking by Hour Across All Logged Days ({metric_label})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(metric_label)
    plt.tight_layout()

    out = output_dir / f"bark_hourly_profile_{HEATMAP_METRIC}.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    return out


def write_plot_notes(output_dir, log_dirs):
    out = output_dir / "plot_notes.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("Dog barking plot notes\n")
        f.write("======================\n\n")
        f.write(f"Source folders: {', '.join(str(d) for d in log_dirs)}\n")
        f.write(f"Metric: {HEATMAP_METRIC}\n")
        f.write(f"Time window: {START_WINDOW.strftime('%H:%M')} to {END_WINDOW.strftime('%H:%M')}\n")
        f.write(f"Minimum barks threshold: {MIN_BARKS_THRESHOLD}\n\n")
        f.write("Notes:\n")
        f.write("- Only dates with an actual log file are included in the plots.\n")
        f.write("- A zero value means a log file exists for that day/hour but no qualifying barking was recorded.\n")
        f.write("- Dates with no log file are omitted entirely.\n")
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python diary_plots.py <log_dir> [log_dir2 ...]")
        print("       Provide one or more folders containing bark_log_*.txt files.")
        sys.exit(1)

    log_dirs = [Path(p) for p in sys.argv[1:]]
    output_dir = log_dirs[0] / "Plots"
    output_dir.mkdir(exist_ok=True)

    minutes_grid, barks_grid, daily_minutes, daily_barks, hourly_minutes, hourly_barks, log_file_dates = load_data(log_dirs)

    if not log_file_dates:
        print("No bark_log_YYYY-MM-DD.txt files found in the selected folders.")
        sys.exit(1)

    plot_dates = sorted(log_file_dates)
    hours = list(range(START_WINDOW.hour, END_WINDOW.hour + 1))

    matrix = np.array([
        [barks_grid[d][h] if HEATMAP_METRIC == "barks" else minutes_grid[d][h] for h in hours]
        for d in plot_dates
    ], dtype=float)

    heatmap_file = plot_heatmap(matrix, plot_dates, hours, output_dir)
    daily_file = plot_daily_totals(plot_dates, daily_barks, daily_minutes, output_dir)
    hourly_file = plot_hourly_profile(hours, hourly_barks, hourly_minutes, output_dir)
    info_file = write_plot_notes(output_dir, log_dirs)

    print("\nPlots created:")
    for f in [heatmap_file, daily_file, hourly_file, info_file]:
        print(f)


if __name__ == "__main__":
    main()
