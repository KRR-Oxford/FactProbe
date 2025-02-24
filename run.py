import os
import pdb
import subprocess
import threading
import queue
from hydra import initialize, compose
from omegaconf import OmegaConf
import sys


def get_free_cuda_devices():
    """Get a list of all free GPU device IDs."""
    result = subprocess.run(
        "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True
    )
    free_gpus = []
    for i, line in enumerate(result.stdout.strip().split("\n")):
        used, free = map(int, line.split(","))
        if free > 10240 * 4:  # 40GB memory threshold
            free_gpus.append(i)
    return free_gpus


def worker_with_progress(task_queue, gpu_queue, update_progress):
    """Worker thread function with progress tracking."""
    max_retries = 3  # Maximum number of retries
    
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
            
        command, retries = task if isinstance(task, tuple) else (task, 0)
        cuda_device = gpu_queue.get()

        try:
            # Execute command
            full_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} {command}"
            print(f"🚀 Running: {full_cmd} (attempt {retries + 1})")
            
            # Add memory limit
            result = subprocess.run(
                full_cmd, 
                shell=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(cuda_device)}
            )

            if result.returncode != 0:
                print(f"❌ Error in command: {full_cmd}")
                if retries < max_retries:
                    # Clean GPU memory before retrying
                    subprocess.run(f"nvidia-smi -r", shell=True)
                    task_queue.put((command, retries + 1))
                else:
                    print(f"❌ Command failed after {max_retries} attempts: {full_cmd}")
                    update_progress()
            else:
                update_progress()

        except Exception as e:
            print(f"❌ Exception in command: {full_cmd}")
            print(f"Error: {str(e)}")
            if retries < max_retries:
                task_queue.put((command, retries + 1))
            else:
                update_progress()
        
        finally:
            # Ensure GPU is returned
            gpu_queue.put(cuda_device)
            task_queue.task_done()


def generate_commands(cfg):
    """Generate a list of all commands to execute."""
    commands = []
    skipped = []  # Record skipped configurations
    
    # Determine if in test mode
    test_mode = cfg.test_mode if hasattr(cfg, 'test_mode') else False
    
    if test_mode:
        print("Running in test mode")
    
    for relation in cfg.relations:
        for mode in cfg.modes:
            for model_name, model_cfg in cfg.models.items():
                config_path = os.path.join(
                    cfg.base_data_dir,
                    "configs",
                    relation + '_configs',
                    # mode,
                    f"{mode}.yaml"
                )
                if not os.path.exists(config_path):
                    skipped.append((relation, mode, model_name))
                    print(f"⚠️  Config missing: {config_path}")
                    continue

                command_parts = [
                    "python",
                    cfg.script_path,
                    "--config_file",
                    config_path,
                    "--model",
                    model_cfg.name,
                ]
                
                # If in test mode, add --test flag
                if test_mode:
                    command_parts.append("--test")
                
                # Add run_all flag if specified in the config
                if cfg.run_all:
                    command_parts.append("--run_all")
                
                # Combine command parts into a full command
                command = " ".join(command_parts)
                commands.append(command)
    
    # Summarize all skipped configurations at the end
    if skipped:
        print("\nWarning: The following combinations were skipped due to missing configuration files:")
        for rel, mod, model in skipped:
            print(f"- Relation: {rel}, Mode: {mod}, Model: {model}")
        
        # Optionally ask the user for confirmation to continue
        response = input("\nDo you want to continue executing other tasks? (y/n): ")
        if response.lower() != 'y':
            return []
            
    return commands


def main(cfg):
    # Print configuration information
    print("\nCurrent configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Generate all commands
    commands = generate_commands(cfg)
    total_tasks = len(commands)
    
    # Print all commands to be executed
    print("\nPlanned task list:")
    print("=" * 80)
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")
    print("=" * 80)
    print(f"\nTotal number of tasks: {total_tasks}\n")
    
    completed_tasks = 0
    
    if not commands:
        print("No tasks to run.")
        return

    # Initialize GPU queue
    available_gpus = get_free_cuda_devices()
    if not available_gpus:
        print("❌ No available GPUs found")
        return

    print(f"✅ Available GPUs: {available_gpus}")

    # Create queues
    task_queue = queue.Queue()
    gpu_queue = queue.Queue()

    # Fill queues
    for cmd in commands:
        task_queue.put(cmd)
    for gpu in available_gpus:
        gpu_queue.put(gpu)

    # Add poison pills (one for each worker thread)
    for _ in range(len(available_gpus)):
        task_queue.put(None)

    # Add progress tracking
    def update_progress():
        nonlocal completed_tasks
        completed_tasks += 1
        print(f"Progress: {completed_tasks}/{total_tasks} ({completed_tasks/total_tasks*100:.2f}%)")

    # Use the new worker function, passing in update_progress
    threads = []
    for _ in range(len(available_gpus)):
        t = threading.Thread(
            target=worker_with_progress, 
            args=(task_queue, gpu_queue, update_progress)
        )
        t.start()
        threads.append(t)

    # Wait for all tasks to complete
    task_queue.join()

    # Wait for threads to finish
    for t in threads:
        t.join()


if __name__ == "__main__":
    with initialize(config_path=".", version_base="1.1"):
        # Use overrides parameter
        cfg = compose(
            config_name="config",
            overrides=["test_mode=true"] if "test_mode=True" in sys.argv else []
        )
        main(cfg)
