# FM-index Implementation for Dolma-v1.7

This repository contains code for implementing FM-index on the Dolma-v1.7 pre-training data. The implementation includes building the FM-index and querying it to generate entity frequencies.

## Prerequisites

Before starting, ensure you have:
- Sufficient disk space for FM-index files
- At least 2GB free memory for parallel processing
- GNU Parallel installed
- C++ compiler with C++17 support

## Step-by-Step Guide

### 1. Install Required Libraries

First, install the SDSL-lite library which is required for FM-index operations:

```bash
cd library 
cd sdsl-lite
./install.sh 
```

### 2. Compile FM-index Builder

Compile the FM-index builder with optimized settings:

```bash
g++ -std=c++17  -O3 -DNDEBUG -msse4.2 -mbmi -mbmi2 -Wall -Wextra -pedantic -funroll-loops -D__extern_always_inline="extern __always_inline"  -ffast-math \
   -I/PATHTO/library/sdsl-lite/include -L/PATHTO/library/FM-Index/libdivsufsort/lib \
    -o fm_index_build.exe fm_index_build.cpp -ldivsufsort
```

### 3. Download Dolma-v1.7 and Build FM-index

Run the following script to download Dolma-v1.7 and build the FM-index:

```bash
bash download_build_index.sh 
```

This will:
- Download Dolma-v1.7 dataset
- Process the data
- Build FM-index files in `./dolma-to-fmindex/data/fm_index/`

### 4. Query FM-index and Generate Entity Frequencies

After the FM-index is built, you can query it to generate entity frequencies:

```bash
bash find_query_in_fm.sh
```

This script will:
- Process all FM-index files in parallel 
- Generate frequency statistics for entities from the preprocessed Wikidata file
- Combine all results into a single JSON file at `./dolma-to-fmindex/data/wiki/dolma_entity_frequencies.json`

Required files for this step:
- FM-index files in `./dolma-to-fmindex/data/fm_index/`
- Preprocessed Wikidata entity file at `./dolma-to-fmindex/data/wiki/wikidata5m_entity.json`
- Compiled `fm-get-freq.exe` in `./dolma-to-fmindex/library/sdsl-lite/examples/`

## Output

The final output will be a JSON file containing entity frequencies:
- Location: `./dolma-to-fmindex/data/wiki/dolma_entity_frequencies.json`
- Format: JSON with entity IDs as keys and their frequencies as values
