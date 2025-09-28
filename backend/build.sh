#!/bin/bash

# Force upgrade pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel --force-reinstall --no-cache-dir

# Install dependencies from your Requirement file
pip install --no-cache-dir -r Requirement

# Optional: run any migrations or setup here if needed
# Example: python bend.py  # Only if needed
