# Copyright 2025 Zifeng Ding, Yuan He, Bailan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import click
import pickle
from collections import defaultdict as ddict
from typing import List, Any
from textwrap import dedent
from vllm import LLM, SamplingParams
from deeponto.utils import load_file, save_file
from factprobe.prompt import SemanticMatchPrompt, AffirmationPrompt


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def batch_iter(items: List[Any], batch_size: int):
    """Yields batches of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        List of items for each batch
    """
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

@click.command()
@click.option("--input_path", "-i", type=str, help="Path to the input pickle file.")
@click.option("--model", "-m", type=str, required=True, help="Name of the model to use for evaluation.")
@click.option("--run_test", is_flag=True, help="Run in test mode with limited samples.")
@click.option("--tensor_parallel_size", "-tp", type=int, default=2, 
              help="Number of GPUs to use for tensor parallelism.")
@click.option("--batch_size", "-b", type=int, default=32, 
              help="Batch size for processing keys.")
def main(input_path: str, model: str, run_test: bool, tensor_parallel_size: int, batch_size: int):
    """Main function to execute the soft EM evaluation pipeline."""
    if tensor_parallel_size > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Display command-line arguments
    command_msg = f"""
        input_path: {input_path}
        model: {model}
        run_test: {run_test}
        tensor_parallel_size: {tensor_parallel_size}
        batch_size: {batch_size}
    """
    logger.info(dedent(command_msg))
    
    # Construct output path
    output_path = input_path.replace('.pkl', '_softem.pkl')
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    with open(input_path, 'rb') as file:
        data = pickle.load(file)
    
    # Check for existing results and load them if available
    results = ddict(lambda: ddict(dict))
    if os.path.exists(output_path):
        logger.info(f"Found existing results at {output_path}, resuming from checkpoint")
        results = load_file(output_path)
    
    # Determine template type from input path
    template_type = "question" if "question" in input_path else "statement"
    reference_answer = "Yes" if template_type == "question" else "True"
    
    if run_test:
        for mode in ["forward", "backward"]:
            test_keys = list(data[mode].keys())[:100]  
            data[mode] = {k: data[mode][k] for k in test_keys}
    
    # Initialize model and prompts
    llm = LLM(
        model=model, 
        dtype="half", 
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True
    )
    semantic_prompt = SemanticMatchPrompt()
    affirmation_prompt = AffirmationPrompt()
    sampling_params = SamplingParams(temperature=0)
    
    # Process forward and backward results
    for mode in ["forward", "backward"]:
        
        # Initialize counters
        total = len(data[mode])
        processed = 0
        semantic_anyone_correct = 0
        semantic_majority_correct = 0
        affirmation_anyone_correct = 0
        affirmation_majority_correct = 0
        
        # Get already processed keys
        processed_keys = set(results[mode].keys()) if mode in results else set()
        logger.info(f"Found {len(processed_keys)} already processed keys in {mode} direction")
        
        # Get keys that need processing
        keys_to_process = [k for k in data[mode].keys() if k not in processed_keys]
        
        if not keys_to_process:
            logger.info(f"All keys in {mode} direction already processed. Skipping.")
            continue
            
        
        # Process data in batches
        for batch_idx, batch_keys in enumerate(batch_iter(keys_to_process, batch_size)):
            logger.info(f"Processing batch {batch_idx+1}/{(len(keys_to_process) + batch_size - 1) // batch_size}")
            
            # Collect all prompts for this batch
            all_semantic_prompts = []
            all_affirmation_prompts = []
            key_to_responses = {}  # Maps key to list of responses
            
            for key in batch_keys:
                responses = data[mode][key]["text"]
                key_to_responses[key] = responses
                
                for response in responses:
                    all_semantic_prompts.append(semantic_prompt.render(response, reference_answer))
                    all_affirmation_prompts.append(affirmation_prompt.render(response))
            
            # Process all semantic prompts at once s
            outputs = llm.chat(all_semantic_prompts, sampling_params)
            semantic_results = [output.outputs[0].text.lower().strip() for output in outputs]
            
            # Process all affirmation prompts at once
            outputs = llm.chat(all_affirmation_prompts, sampling_params)
            affirmation_results = [output.outputs[0].text.lower().strip() for output in outputs]
            
            result_idx = 0
            for key in batch_keys:
                num_responses = len(key_to_responses[key])
                
                # Get results for this key
                key_semantic_results = semantic_results[result_idx:result_idx + num_responses]
                key_affirmation_results = affirmation_results[result_idx:result_idx + num_responses]
                result_idx += num_responses
                
                # Store results
                results.setdefault(mode, {}).setdefault(key, {})
                results[mode][key]['text'] = key_to_responses[key]
                results[mode][key]["answer_sem"] = key_semantic_results
                results[mode][key]["answer_aff"] = key_affirmation_results
                
                # Update metrics
                if "yes" in key_semantic_results:
                    semantic_anyone_correct += 1
                
                sem_vote_count = ddict(int)
                for result in key_semantic_results:
                    sem_vote_count[result] += 1
                if "yes" == max(sem_vote_count, key=sem_vote_count.get):
                    semantic_majority_correct += 1
                
                if "yes" in key_affirmation_results:
                    affirmation_anyone_correct += 1
                
                aff_vote_count = ddict(int)
                for result in key_affirmation_results:
                    aff_vote_count[result] += 1
                if "yes" == max(aff_vote_count, key=aff_vote_count.get):
                    affirmation_majority_correct += 1
                
                processed += 1
            
            # Save results after each batch
            logger.info(f"Processed {processed}/{total} keys in {mode} direction, saving intermediate results")
            save_file(results, output_path)
            
        
        # Calculate and log final metrics
        if processed > 0:
            semantic_anyone_score = semantic_anyone_correct / processed
            semantic_majority_score = semantic_majority_correct / processed
            affirmation_anyone_score = affirmation_anyone_correct / processed
            affirmation_majority_score = affirmation_majority_correct / processed
            
            logger.info(f"Final {mode.capitalize()} Soft EM (semantic, anyone): {semantic_anyone_score:.4f}")
            logger.info(f"Final {mode.capitalize()} Soft EM (semantic, majority): {semantic_majority_score:.4f}")
            logger.info(f"Final {mode.capitalize()} Soft EM (affirmation, anyone): {affirmation_anyone_score:.4f}")
            logger.info(f"Final {mode.capitalize()} Soft EM (affirmation, majority): {affirmation_majority_score:.4f}")
    
    # Save final results
    logger.info(f"Saving final results to {output_path}")
    save_file(results, output_path)

if __name__ == "__main__":
    main() 