#!/bin/zsh

# Find and delete all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "All __pycache__ directories have been removed."
