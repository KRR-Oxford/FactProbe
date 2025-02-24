# Copyright 2025 Yuan He
import pdb
import itertools

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import pandas as pd
from vllm import LLM
from factprobe.probe import FactProbe
from deeponto.utils import save_file, load_file
from yacs.config import CfgNode
import os
import glob


@click.command()
@click.option("--config_file", "-c", type=str, help="Path to the configuration file.")
@click.option("--model", "-m", type=str, default=None, help="Name of the model to use.")
@click.option("--test", is_flag=True, help="Run in test mode with only 100 samples")
@click.option("--run_all", "-r", is_flag=True, help="Whether to ignore the high-low count threshold and run on all triples.")

def main(config_file: str, model: str, test: bool, run_all: bool):
    print("\nCommand line arguments:")
    print(f"config_file: {config_file}")
    print(f"model: {model}")
    print(f"test: {test}")
    print(f"run_all: {run_all}")
    
    # 0. Load the configuration file
    config = CfgNode(load_file(config_file))
    if model:
        config.model = model

    # 1. Load the preprocess data with batch processing
    BATCH_SIZE = 10000
    
    # Read data
    if test:
        df = pd.read_csv(config.dataset, nrows=100)
    else:
        df = pd.read_csv(config.dataset)
    
    # Filter data according to run_all parameter
    if not run_all:
        data_settings = {
            "high2low": df[
                (df["subject_count"] >= config.count_high)
                & (df["object_count"] <= config.count_low)
            ],
            "low2high": df[
                (df["subject_count"] <= config.count_low)
                & (df["object_count"] >= config.count_high)
            ],
        }
    else:
        data_settings = {"all": df}

    total_rows = len(df)
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total samples: {total_rows}, will process in {num_batches} batches")
    
    # Process data in batches
    batches = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        batch = df.iloc[start_idx:end_idx]
        batches.append(batch)

    # Check if all required settings already have result files
    all_files_exist = True
    save_paths = []
    for so_setting in data_settings.keys():
        base_path = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/Dolma/FactProbe/experiments"
        
        # Adjust file name format based on run_all parameter
        if run_all:
            file_name = f"{config.relation}_{so_setting}_{config.template_type}.pkl"
        else:
            file_name = f"{config.relation}_{config.count_high}_{config.count_low}_{so_setting}_{config.template_type}.pkl"
        
        if test:
            save_path = os.path.join(
                base_path,
                "test",
                config.relation,
                config.model,
                file_name
            )
        else:
            save_path = os.path.join(
                base_path,
                config.relation,
                config.model,
                file_name
            )
        
        save_paths.append((so_setting, save_path))
        if not os.path.exists(save_path):
            all_files_exist = False

    if all_files_exist:
        print("All result files already exist, no need to reprocess.")
        return

    # 3. Only initialize the model if processing is needed
    print(f"Loading the model...{config.model}")
    llm = LLM(model=config.model)
    probe = FactProbe(llm=llm, **config)

    # 4. Process each batch and merge results
    print(f"Relation: {config.relation}, Template Type: {config.template_type}")
    for so_setting, save_path in save_paths:
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            continue

        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Check for any incomplete batch results
        last_completed_batch = -1
        merged_results = {
            "forward": {},
            "backward": {}
        }
        
        # Find the latest intermediate result file
        for i in range(len(batches)):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, total_rows)
            base_name = save_path[:-4]  # Remove '.pkl' extension
            temp_file = f"{base_name}_incomplete_{start_idx}_to_{end_idx}.pkl"
            if os.path.exists(temp_file):
                try:
                    temp_results = load_file(temp_file)
                    merged_results = temp_results
                    last_completed_batch = i
                    print(f"Found existing results for data range {start_idx}-{end_idx}")
                except Exception as e:
                    print(f"Error loading data range {start_idx}-{end_idx}: {e}")
                    break

        # Process each batch starting from the last completed batch
        for i, batch_df in enumerate(batches):
            if i <= last_completed_batch:
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, total_rows)
                print(f"Skipping data range {start_idx}-{end_idx} (already processed)")
                continue
                
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, total_rows)
            print(f"\nProcessing data range {start_idx}-{end_idx} ({len(batch_df)} samples)")
            
            try:
                batch_results = probe.probe(batch_df, so_setting)
                
                merged_results["forward"].update(batch_results["forward"])
                merged_results["backward"].update(batch_results["backward"])
                
                # Save intermediate results
                base_name = save_path[:-4]
                temp_save_path = f"{base_name}_incomplete_{start_idx}_to_{end_idx}.pkl"
                save_file(merged_results, temp_save_path)
                print(f"Saved intermediate results to {temp_save_path}")
                
                # Remove previous intermediate results to save space
                if i > 0:
                    prev_start = (i - 1) * BATCH_SIZE
                    prev_end = min(i * BATCH_SIZE, total_rows)
                    prev_temp_file = f"{base_name}_incomplete_{prev_start}_to_{prev_end}.pkl"
                    if os.path.exists(prev_temp_file):
                        os.remove(prev_temp_file)
                        print(f"Removed previous intermediate file: {prev_temp_file}")
                
            except Exception as e:
                print(f"Error processing data range {start_idx}-{end_idx}: {e}")
                print("You can restart the program to continue from this point")
                print(f"Latest results saved in: {temp_save_path}")
                raise e

        # Validate data integrity
        try:
            print("\nValidating data integrity...")
            # Collect all processed subject-object pairs
            processed_pairs = set()
            for results in [merged_results["forward"], merged_results["backward"]]:
                processed_pairs.update(results.keys())
            
            # Collect all subject-object pairs that should be processed
            expected_pairs = set()
            for df in batches:
                for _, row in df.iterrows():
                    subjects = eval(row["subject_name"])
                    objects = eval(row["object_name"])
                    for s, o in itertools.product(subjects, objects):
                        expected_pairs.add((row["subject"], row["object"]))
            
            # Check for any missing pairs
            missing_pairs = expected_pairs - processed_pairs
            if missing_pairs:
                print(f"⚠️ Found {len(missing_pairs)} unprocessed data pairs!")
                print("Example missing data:")
                for pair in list(missing_pairs)[:5]:
                    print(f"  - Subject: {pair[0]}, Object: {pair[1]}")
                raise ValueError("Data processing incomplete")
            
            # Check for any extra pairs
            extra_pairs = processed_pairs - expected_pairs
            if extra_pairs:
                print(f"⚠️ Found {len(extra_pairs)} extra data pairs!")
                print("Example extra data:")
                for pair in list(extra_pairs)[:5]:
                    print(f"  - Subject: {pair[0]}, Object: {pair[1]}")
                raise ValueError("Unexpected data found")
            
            print(f"✅ Data integrity validation passed! Processed {len(processed_pairs)} data pairs")
            
            # Save final results after all batches are processed and validated
            save_file(merged_results, save_path)
            print(f"Saved final results to {save_path}")
            
            # Clean up all intermediate files
            base_name = save_path[:-4]
            cleanup_pattern = f"{base_name}_incomplete_*_to_*.pkl"
            for f in glob.glob(cleanup_pattern):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove intermediate file {f}: {e}")
            print("Cleaned up all intermediate files")
            
        except Exception as e:
            print(f"Error during validation or saving: {e}")
            print("Latest intermediate results are still available in .incomplete files")
            # List all available intermediate files
            incomplete_files = [f for f in os.listdir(os.path.dirname(save_path)) 
                              if f.startswith(os.path.basename(save_path) + "_incomplete")]
            if incomplete_files:
                print("Available intermediate files:")
                for f in sorted(incomplete_files):
                    print(f"  - {f}")
            raise e

        # After complete success, check and clean any leftover intermediate files
        cleanup_pattern = save_path[:-4] + "_incomplete_*"
        for f in glob.glob(cleanup_pattern):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: Could not remove intermediate file {f}: {e}")


if __name__ == "__main__":
    main()
