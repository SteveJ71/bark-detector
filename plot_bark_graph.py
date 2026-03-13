import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

bark_pattern = re.compile(
    r"\[BARKING DETECTED\] (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2}"
)


def load_bark_data(log_dir: Path) -> dict:
    """Read all bark log files in log_dir and return {date: {minute: count}}."""
    daily_bark_counts = defaultdict(lambda: defaultdict(int))

    for filepath in sorted(log_dir.glob("bark_log_*.txt")):
        with open(filepath, "r") as f:
            for line in f:
                match = bark_pattern.search(line)
                if match:
                    dt = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M")
                    daily_bark_counts[dt.date()][dt] += 1

    return daily_bark_counts


def plot_bark_data(daily_bark_counts: dict):
    """Plot bark frequency over time for all days."""
    if not daily_bark_counts:
        print("No bark data found.")
        return

    plt.figure(figsize=(14, 6))
    for date, barks in sorted(daily_bark_counts.items()):
        times = sorted(barks.keys())
        counts = [barks[t] for t in times]
        plt.plot(times, counts, marker='o', linestyle='-', label=date.strftime("%Y-%m-%d"))

    plt.title("Dog Barks Over Time (All Days)")
    plt.xlabel("Time")
    plt.ylabel("Barks per Minute")
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    daily_bark_counts = load_bark_data(log_dir)
    plot_bark_data(daily_bark_counts)


if __name__ == "__main__":
    main()
