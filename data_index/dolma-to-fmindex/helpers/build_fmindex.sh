#!/bin/bash

# Check if the correct number of arguments is provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <start_index> <end_index>"
    echo "Example: $0 1 10"
    exit 1
fi

# Parse arguments
START_INDEX=$1
END_INDEX=$2

# Validate that start and end are integers
if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]] || ! [[ "$END_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Error: Start and end indices must be integers."
    exit 1
fi

# Ensure start index is less than or equal to end index
if [[ $START_INDEX -gt $END_INDEX ]]; then
    echo "Error: Start index must be less than or equal to end index."
    exit 1
fi

# Define directories
RAW_DIR="./data/raw"
PROCESSED_DIR="./data/processed"
FM_INDEX_DIR="./data/fm_index"
PYTHON_SCRIPT="./helpers/json_gz_to_text_gz.py"
FM_INDEX_BUILDER="./library/sdsl-lite/examples/fm_index_build.exe"

# Create necessary directories if they don't exist
mkdir -p "$PROCESSED_DIR"
mkdir -p "$FM_INDEX_DIR"

# Get a sorted list of json.gz files
mapfile -t json_files < <(ls "$RAW_DIR"/*.json.gz | sort)

# Total number of files
TOTAL_FILES=${#json_files[@]}

# Check if indices are within range
if [[ $START_INDEX -lt 1 ]] || [[ $END_INDEX -gt $TOTAL_FILES ]]; then
    echo "Error: Indices out of range. There are $TOTAL_FILES files."
    END_INDEX=$((TOTAL_FILES + 1))
fi

# Adjust indices for array (arrays are zero-indexed)
START_INDEX=$((START_INDEX - 1))
END_INDEX=$((END_INDEX - 1))

# Process files in the specified range
for (( i=START_INDEX; i<=END_INDEX; i++ )); do
    json_file="${json_files[i]}"
    base_name="$(basename "$json_file" .json.gz)"
#    echo "Processing $json_file"

#    if compgen -G "$FM_INDEX_DIR/$base_name*" > /dev/null; then
##      echo "FM file for $json_file already exists. Skipping."
#      continue
#    else
#      echo "FM file for $json_file does not exist. Processing."
#    fi

    # Run the Python script, specifying the output directory
#    python "$PYTHON_SCRIPT" "$json_file" "$PROCESSED_DIR/$base_name.txt"

    # Find all generated text files for this json.gz file in PROCESSED_DIR
    text_files=$(find "$PROCESSED_DIR" -maxdepth 1 -type f -name "${base_name}*.txt")
    echo "text files are ${text_files}"

    # Build FM-index for each text file
    for txt_file in $text_files; do
        fm_index_file="$FM_INDEX_DIR/$(basename "$txt_file").fm9"

        # Check if FM-index file already exists
        if [[ -f "$fm_index_file" ]]; then
            echo "FM-index for $txt_file already exists at $fm_index_file. Skipping."
            rm "$txt_file"
            continue
        fi

        echo "Building FM-index for $txt_file..."
        "$FM_INDEX_BUILDER" "$txt_file"

        if [[ $? -ne 0 ]]; then
            echo "Error: Failed to build FM-index for $txt_file."
            continue
        fi

        echo "Built FM-index for $txt_file and put it to $fm_index_file."

#        mv "${txt_file}.fm9" "$fm_index_file"
#        rm "$txt_file"
#        echo "Removed $txt_file."
    done
done
