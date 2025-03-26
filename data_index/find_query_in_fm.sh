#!/bin/bash

# Paths to the test files
FM_INDEX_DIR="./dolma-to-fmindex/data/fm_index"
JSON_FILE="./dolma-to-fmindex/data/wiki/preprocess_wikidata5m_entity.json"
FM_INDEX_EXECUTABLE="./dolma-to-fmindex/library/sdsl-lite/examples/fm_get_freq.exe"

# Temporary directory to store individual output files
TEMP_OUTPUT_DIR="./dolma-to-fmindex/data/wiki/temp_output"
mkdir -p "$TEMP_OUTPUT_DIR"

# Final combined output JSON file
FINAL_OUTPUT_JSON="./dolma-to-fmindex/data/wiki/dolma_entity_frequencies.json"

# Progress tracking file to keep track of completed files
PROGRESS_FILE="./dolma-to-fmindex/data/wiki/progress_file.txt"

# Get the total number of FM-index files
FM_INDEX_FILES=($(find "$FM_INDEX_DIR" -type f -name "*.fm9"))
TOTAL_FILES=${#FM_INDEX_FILES[@]}

# Load the JSON file once and store its content
JSON_CONTENT=$(cat "$JSON_FILE")

# Function to create a progress bar
ProgressBar() {
    local total=$1
    while true; do
        local completed=$(cat "$PROGRESS_FILE" 2>/dev/null | wc -l)
        local percentage=$(( (completed * 100) / total ))
        local filled=$(( (completed * 40) / total ))
        local empty=$(( 40 - filled ))
        local bar=$(printf "%-${filled}s" "#" | sed 's/ /#/g')
        local spaces=$(printf "%-${empty}s" "-")
        printf "\rProgress: [${bar}${spaces}] ${percentage}%%"
        sleep 1
    done
}

# Start the progress bar in the background
ProgressBar $TOTAL_FILES &
progress_pid=$!

export TEMP_OUTPUT_DIR
export FM_INDEX_EXECUTABLE
export PROGRESS_FILE
export JSON_FILE

# Run the executable in parallel for each FM-index file using GNU Parallel
find "$FM_INDEX_DIR" -type f -name "*.fm9" | \
    parallel --bar --joblog "$TEMP_OUTPUT_DIR/parallel_log" -j16 --retries 3 --memfree 2G --halt soon,fail=1 '
        fm_index_file={}
        output_file="$TEMP_OUTPUT_DIR/$(basename "$fm_index_file").json"

        # Skip processing if the file has already been processed (exists in progress file)
        if grep -Fxq "$fm_index_file" "$PROGRESS_FILE"; then
            echo "$fm_index_file already processed, skipping."
            exit 0
        fi

        # Process the FM-index file in parallel
        "$FM_INDEX_EXECUTABLE" "$fm_index_file" "$JSON_FILE" "$output_file"

        # If the execution was successful, record it in the progress file
        if [ $? -eq 0 ]; then
            echo "$fm_index_file" >> "$PROGRESS_FILE"
        else
            echo "Error executing $FM_INDEX_EXECUTABLE with $fm_index_file" >&2
        fi
    '

# Kill the progress bar process once all tasks are complete
kill $progress_pid
wait $progress_pid 2>/dev/null

echo ""

# Debug: List generated JSON files
echo "Generated JSON files:"
ls -lh "$TEMP_OUTPUT_DIR"

# Combine all individual JSON files into the final output JSON
if [ -n "$(ls -A $TEMP_OUTPUT_DIR/*.json 2>/dev/null)" ]; then
    # Combine JSON files by adding the values for identical keys
    combined_json=$(jq -s 'reduce .[] as $item ({}; reduce (keys_unsorted[] | select(has($item))) as $key (.;
        .[$key] += $item[$key]))' "$TEMP_OUTPUT_DIR"/*.json)

    # If combining the JSON was successful
    if [ -n "$combined_json" ]; then
        echo "$combined_json" > "$FINAL_OUTPUT_JSON"
        echo "Combined output written to: $FINAL_OUTPUT_JSON"
    else
        echo "Error: Combined JSON output is empty."
    fi
else
    echo "No JSON files were generated. Please check for errors."
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

# Display total elapsed time
echo "Execution Time: $(($ELAPSED / 60)) minutes and $(($ELAPSED % 60)) seconds."
