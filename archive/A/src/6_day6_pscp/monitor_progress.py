#!/usr/bin/env python3
"""Monitor the improved PSCP experiment progress."""

import time
import os
from datetime import datetime, timedelta

LOG_FILE = "/home/home_public/aisci/AIscientist/src/6_day6_pscp/improved_experiment_log.txt"
RESULTS_FILE = "/home/home_public/aisci/AIscientist/src/6_day6_pscp/improved_results.json"

def get_last_lines(filepath, n=20):
    """Get last n lines from file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except Exception as e:
        return [f"Error reading file: {e}"]

def check_process():
    """Check if experiment process is still running."""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        return "run_improved_pscp.py" in result.stdout
    except:
        return False

def main():
    print("=" * 70)
    print("MONITORING IMPROVED PSCP EXPERIMENT")
    print("=" * 70)
    print(f"Log file: {LOG_FILE}")
    print(f"Results file: {RESULTS_FILE}")
    print(f"Started checking: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    check_interval = 60  # Check every 60 seconds
    max_checks = 300  # Maximum 300 checks (5 hours)
    check_count = 0

    while check_count < max_checks:
        check_count += 1

        # Check if results file exists
        if os.path.exists(RESULTS_FILE):
            print(f"\n{'=' * 70}")
            print("EXPERIMENT COMPLETED!")
            print(f"Results file found at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            return True

        # Check if process is still running
        process_running = check_process()
        if not process_running:
            print(f"\n{'=' * 70}")
            print("WARNING: Experiment process not detected!")
            print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            # Don't return yet, might be a temporary issue

        # Show last few lines of log
        print(f"\n--- Check #{check_count} at {datetime.now().strftime('%H:%M:%S')} ---")
        print("Recent log output:")
        recent_lines = get_last_lines(LOG_FILE, 15)
        for line in recent_lines:
            print(f"  {line.rstrip()}")

        print(f"\nProcess running: {'Yes' if process_running else 'No'}")
        print(f"Next check in {check_interval} seconds...")
        print("-" * 70)

        time.sleep(check_interval)

    print(f"\n{'=' * 70}")
    print(f"MAXIMUM MONITORING TIME REACHED ({check_count} checks)")
    print("=" * 70)
    return False

if __name__ == "__main__":
    main()
