#!/usr/bin/env python3
"""Wrapper to run experiment with unbuffered output."""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# Import and run main
import run_experiment
run_experiment.main()
