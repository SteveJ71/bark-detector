import os
import re
import sys
from datetime import datetime, time

MIN_BARKS_PER_SESSION = 20
START_WINDOW = time(3, 0, 0)
END_WINDOW = time(23, 0, 0)

session_pattern = re.compile(
    r"\[BARK SESSION\] From (?P<start>[\d\-: ]+) to (?P<end>[\d\-: ]+)"
    r" \(Duration: (?P<duration>[\d.]+) seconds, (?P<barks>\d+) barks\)"
)


def parse_log_file(filepath):
    """Parse a single bark log file and return a list of session summary strings
    along with the total seconds and bark count for sessions within the time window."""
    lines = []
    total_seconds = 0
    total_barks = 0

    with open(filepath, "r") as f:
        for line in f:
            match = session_pattern.search(line)
            if not match:
                continue

            start_time = datetime.strptime(match.group("start"), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(match.group("end"), "%Y-%m-%d %H:%M:%S")

            start_rounded = start_time.replace(second=0)
            end_rounded = end_time.replace(second=0)

            if not (START_WINDOW <= start_rounded.time() <= END_WINDOW):
                continue

            barks = int(match.group("barks"))
            if barks <= MIN_BARKS_PER_SESSION:
                continue

            duration = int((end_rounded - start_rounded).total_seconds())
            minutes = duration // 60
            seconds = duration % 60
            start_str = start_rounded.strftime("%H:%M")
            end_str = end_rounded.strftime("%H:%M")

            lines.append(f"{start_str}–{end_str} ({minutes}:{seconds:02d})")
            total_seconds += duration
            total_barks += barks

    return lines, total_seconds, total_barks


def format_duration(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 0:
        return f"{hours} hrs {minutes} min"
    return f"{minutes} min"


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = os.path.join(log_dir, "bark_diary_summary.txt")

    summary_lines = []
    total_seconds = 0
    total_barks = 0

    for filename in sorted(os.listdir(log_dir)):
        if not filename.startswith("bark_log"):
            continue

        filepath = os.path.join(log_dir, filename)
        print(f"Processing {filepath}")

        try:
            date_str = filename.replace("bark_log_", "").replace(".txt", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            summary_lines.append(f"=== {file_date.strftime('%A %d %B %Y')} ===")
        except ValueError:
            summary_lines.append(f"=== {filename} ===")

        sessions, file_seconds, file_barks = parse_log_file(filepath)
        summary_lines.extend(sessions)
        summary_lines.append("")  # blank line between days
        total_seconds += file_seconds
        total_barks += file_barks

    with open(output_file, "w") as f:
        for entry in summary_lines:
            f.write(entry + "\n")
        f.write("\n")
        f.write(f"Total barking time: {format_duration(total_seconds)}\n")
        f.write(f"Total estimated barks: {total_barks}\n")

    print(f"Summary written to {output_file}")


if __name__ == "__main__":
    main()
