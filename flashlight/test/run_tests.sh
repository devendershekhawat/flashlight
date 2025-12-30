#!/bin/bash
# Simple test runner script for Flashlight tests

cd "$(dirname "$0")/../.." || exit 1

echo "Running Flashlight tests..."
python -m pytest flashlight/test/ -v "$@"

