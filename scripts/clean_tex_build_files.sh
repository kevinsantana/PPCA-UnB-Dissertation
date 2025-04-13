#!/bin/bash

# Script to clean LaTeX auxiliary files from a specified directory.

# Check if the target directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <target_directory>"
  exit 1
fi

TARGET_DIR="$1"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Target directory '$TARGET_DIR' not found."
  exit 1
fi

echo "Cleaning LaTeX build files in '$TARGET_DIR'..."

# Use find to locate and delete common LaTeX auxiliary files within the target directory.
# -maxdepth 1 ensures we only delete files directly within TARGET_DIR, not in subdirectories.
# Add or remove extensions as needed for your specific build process.
find "$TARGET_DIR" -maxdepth 1 -type f \( \
  -name '*.aux' -o \
  -name '*.log' -o \
  -name '*.toc' -o \
  -name '*.lof' -o \
  -name '*.lot' -o \
  -name '*.out' -o \
  -name '*.bbl' -o \
  -name '*.blg' -o \
  -name '*.glg' -o \
  -name '*.gls' -o \
  -name '*.glo' -o \
  -name '*.ist' -o \
  -name '*.acn' -o \
  -name '*.acr' -o \
  -name '*.alg' -o \
  -name '*.synctex.gz' -o \
  -name '*.fls' -o \
  -name '*.fdb_latexmk' \
\) -delete

# If you also generate PDF files you want to clean, uncomment the line below:
# find "$TARGET_DIR" -maxdepth 1 -type f -name '*.pdf' -delete

echo "Cleaning complete in '$TARGET_DIR'."
exit 0
