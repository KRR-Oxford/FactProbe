#!/bin/bash

# Define the input file containing the URLs
INPUT_FILE="v1_7.txt"
#INPUT_FILE="v1_6-sample.txt"
# Define the directory where files will be downloaded
DOWNLOAD_DIR="./data/raw"

# Check if the input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found!"
    exit 1
fi

# Check if the correct number of arguments is provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <start_line_number> <end_line_number>"
    echo "Example: $0 1 10"
    exit 1
fi

START_LINE=$1
END_LINE=$2

# Validate that start and end lines are integers
if ! [[ "$START_LINE" =~ ^[0-9]+$ ]] || ! [[ "$END_LINE" =~ ^[0-9]+$ ]]; then
    echo "Error: Start and end line numbers must be integers."
    exit 1
fi

# Ensure start line is less than end line
if [[ $START_LINE -ge $END_LINE ]]; then
    echo "Error: Start line number must be less than end line number."
    exit 1
fi

# Create the download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Read the specified range of lines from the input file
LINE_NUMBER=0
while IFS= read -r url || [[ -n "$url" ]]; do
    LINE_NUMBER=$((LINE_NUMBER + 1))

    # Skip lines outside the specified range
    if [[ $LINE_NUMBER -lt $START_LINE ]]; then
        continue
    elif [[ $LINE_NUMBER -ge $END_LINE ]]; then
        break
    fi

    # Skip empty lines and lines starting with '#'
    if [[ -z "$url" || "$url" == \#* ]]; then
        continue
    fi

    echo "Downloading line $LINE_NUMBER: $url"

    # Use curl in silent mode to download the file to the specified directory
    curl -s -o "$DOWNLOAD_DIR/$(basename "$url")" "$url"

    # Check if the download was successful
    if [[ $? -ne 0 ]]; then
        echo "Failed to download $url"
    else
        echo "Successfully downloaded $(basename "$url") to $DOWNLOAD_DIR"
    fi

done < "$INPUT_FILE"
