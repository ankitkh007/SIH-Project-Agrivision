
#!/bin/bash

# Forcefully upgrade pip, setuptools, wheel before anything else
python -m pip install --upgrade pip setuptools wheel --force-reinstall

# Install dependencies from your Requirement file
pip install --no-cache-dir -r requirement.txt

# Optional: run your FastAPI app or migrations
# uvicorn main:app --host 0.0.0.0 --port 10000
