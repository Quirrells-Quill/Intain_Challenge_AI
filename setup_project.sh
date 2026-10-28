#!/bin/bash

# Create foundational directory structure for Intain-Sight
mkdir -p intain-sight/src/{data,profiling,features,models,survival,anomaly,scenarios,explain,llm,dashboard,utils}
mkdir -p intain-sight/{notebooks,configs,data,models,reports,tests}

# Create .gitignore in data directory to prevent committing raw data
echo "*.csv" > intain-sight/data/.gitignore
echo "*.parquet" >> intain-sight/data/.gitignore

# Create placeholder files
touch intain-sight/requirements.txt
touch intain-sight/README.md
touch intain-sight/setup.py

echo "Intain-Sight directory structure successfully initialized."
