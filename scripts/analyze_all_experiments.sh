#!/bin/bash

# Base directories
EXPERIMENT_DIR="experiments"
DATA_DIR="data/cleaned"
OUTPUT_DIR="analysis"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Relations to process
RELATIONS=("P26" "P47" "P190" "P460" "P3373")

# Models that we expect to have results for
MODELS=(
    "Llama-3.1-8B-Instruct"
    "OLMo-2-0325-32B-Instruct"
    "OLMo-2-1124-13B-Instruct"
    "OLMo-2-1124-7B-Instruct"
    "Qwen2.5-7B-Instruct"
)

# Result types and temperatures
TYPES=("question" "statement")
TEMPS=("0" "1")

for relation in "${RELATIONS[@]}"; do
    echo "Processing relation: $relation"
    
    # Get the corresponding triple file
    TRIPLE_FILE="$DATA_DIR/${relation}_triples.csv"
    
    if [ ! -f "$TRIPLE_FILE" ]; then
        echo "Warning: Triple file not found for $relation: $TRIPLE_FILE"
        continue
    fi
    
    for model in "${MODELS[@]}"; do
        echo "  Processing model: $model"
        
        for type in "${TYPES[@]}"; do
            for temp in "${TEMPS[@]}"; do
                RESULT_FILE="$EXPERIMENT_DIR/$relation/$model/${relation}_all_${type}_temp${temp}.pkl"
                
                if [ ! -f "$RESULT_FILE" ]; then
                    echo "    Warning: Result file not found: $RESULT_FILE"
                    continue
                fi
                
                OUTPUT_FILE="$OUTPUT_DIR/$relation/${model}/${relation}_all_${type}_temp${temp}.analysis.json"
                mkdir -p "$(dirname "$OUTPUT_FILE")"
                
                echo "    Analyzing: ${relation}_all_${type}_temp${temp}"
                python scripts/analyze_experiment.py "$RESULT_FILE" "$TRIPLE_FILE" -o "$OUTPUT_FILE"
            done
        done
    done
done

echo "Analysis complete. Results saved in $OUTPUT_DIR"
