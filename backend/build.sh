#!/usr/bin/env bash
set -o errexit

# Upgrade pip + build tools first
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt