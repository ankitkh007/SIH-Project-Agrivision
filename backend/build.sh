#!/bin/bash

# Upgrade pip & setuptools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies (force reinstall to make uvicorn available)
pip install --no-cache-dir --force-reinstall -r requirements.txt
