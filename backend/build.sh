#!/bin/bash

# 1️⃣ Force upgrade pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel --force-reinstall --no-cache-dir

# 2️⃣ Install Cython first (so other C extensions compile safely)
pip install --no-cache-dir Cython==0.29.36

# 3️⃣ Install rest of the dependencies
pip install --no-cache-dir -r requirements.txt

# ✅ Optional: migrations or pre-start commands
# python bend.py  # only if needed
