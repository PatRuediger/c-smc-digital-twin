import subprocess
import os
import argparse
import json
import shutil 
from datetime import datetime  
import time


def format_duration(seconds):
    """Converts seconds to a readable string (minutes and seconds)."""
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)} minute(s) and {int(secs)} second(s)"

def run_blender_simulation(blender_executable, blend_file, script_file, seed, config_path):
    """
    Starts a single Blender simulation in the background.

    :param blender_executable: Path to the Blender application.
    :param blend_file: Path to the .blend start file.
    :param script_file: Path to the adapted Python script for Blender.
    :param seed: The random seed passed to the script.
    :param config_path: Path to the configuration file to use.
    """
    print(f"\n{'='*20} Starting simulation with seed: {seed} {'='*20}")
    
    # Build the command
    # blender_executable -b blend_file -P script_file -- --seed <value> --config <path>
    command = [
        blender_executable,
        '-b',  # Starts Blender in background (no UI)
        blend_file,
        '-P',  # Runs the specified Python script
        script_file,
        '--',  # Separator: Everything after is passed to the Python script
        '--seed',
        str(seed),
        '--config',
        config_path
    ]
    
    print(f"Executing command: {' '.join(command)}")
    
    try:
        # Run the command and wait until it finishes
        subprocess.run(command, check=True)
        print(f"--- Simulation with seed {seed} completed successfully. ---")
    except subprocess.CalledProcessError as e:
        print(f"ERROR during simulation with seed {seed}: {e}")
    except FileNotFoundError:
        print(f"ERROR: The path to the Blender application '{blender_executable}' was not found.")
        print("Please check the path and try again.")

if __name__ == "__main__":
    # TIMING: Record start time for the entire execution
    total_start_time = time.monotonic()
    # --- Configuration ---
    # BLENDER_PATH is resolved in this order:
    #   1. BLENDER_PATH environment variable (set this to override everything)
    #   2. Platform auto-detection:
    #      - macOS: /Applications/Blender.app/Contents/MacOS/Blender
    #      - Linux/other: blender  (assumes Blender is on PATH)
    import platform
    _env_blender = os.environ.get("BLENDER_PATH")
    if _env_blender:
        BLENDER_PATH = _env_blender
    elif platform.system() == "Darwin":
        BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
    else:
        # Linux and other Unix-like systems: rely on PATH
        BLENDER_PATH = "blender"

    # Paths to your files (assumed to be in the same folder as this script)
    # os.path.abspath ensures we always have the full path
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BLEND_FILE_PATH = os.path.join(CURRENT_DIR, "StripsGen_outLetSimulation_init.blend")
    SCRIPT_FILE_PATH = os.path.join(CURRENT_DIR, "stripGen_Ana_automate.py") # Name of the adapted script
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Blender simulation.")
    parser.add_argument("config_path", nargs="?", help="Path to the configuration file")
    args = parser.parse_args()

    if args.config_path:
        CONFIG_FILE_PATH = os.path.abspath(args.config_path)
    else:
        CONFIG_FILE_PATH = os.path.join(CURRENT_DIR, "config.json") # Default path

    # --- Load the configuration from the JSON file ---
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config_data = json.load(f)
        
        # Get the list of seeds from the loaded configuration
        seeds_to_run = config_data.get("simulation_run", {}).get("seeds", [])
        
        if not seeds_to_run:
            print("Error: No seeds found in 'config.json' under 'simulation_run -> seeds'.")
            exit()
            
    except FileNotFoundError:
        print(f"ERROR: The configuration file 'config.json' was not found in: {CURRENT_DIR}")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: The file 'config.json' contains invalid JSON. Please check the syntax.")
        exit()

    # --- Save a copy of the configuration file in the output folder ---
    try:
        # 1. Get the path from the loaded configuration
        output_path = config_data.get("output_paths", {}).get("db_output_path")
        if not output_path:
            print("Warning: 'db_output_path' not found in config.json. Cannot backup configuration.")
        else:
            # 2. Ensure the target folder exists
            os.makedirs(output_path, exist_ok=True)
            
            # 3. Create a timestamp and a new filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_config_name = f"config_{timestamp}.json"
            destination_path = os.path.join(output_path, backup_config_name)
            
            # 4. Copy the file
            shutil.copy(CONFIG_FILE_PATH, destination_path)
            print(f"\n✅ Configuration successfully backed up at: {destination_path}\n")
            
    except Exception as e:
        print(f"Warning: Could not backup the configuration file. Error: {e}")
    # --- End of new block ---


    # --- The loop to iterate over all seeds ---
    print(f"Starting simulation batch for {len(seeds_to_run)} seeds: {seeds_to_run}")

    for i, current_seed in enumerate(seeds_to_run):
        # TIMING: Record start time for this single seed
        seed_start_time = time.monotonic()

        run_blender_simulation(
            blender_executable=BLENDER_PATH,
            blend_file=BLEND_FILE_PATH,
            script_file=SCRIPT_FILE_PATH,
            seed=current_seed,
            config_path=CONFIG_FILE_PATH
        )

        # TIMING: Record end time for this seed and calculate duration
        seed_end_time = time.monotonic()
        seed_duration = seed_end_time - seed_start_time
        print(f"--- Simulation with seed {current_seed} (run {i+1}/{len(seeds_to_run)}) completed successfully in {format_duration(seed_duration)}. ---")
        
    # TIMING: Record end time for the entire execution and calculate duration
    total_end_time = time.monotonic()
    total_duration = total_end_time - total_start_time
        
    print("\nAll simulations in the batch have been completed.")
    print(f"Total duration: {format_duration(total_duration)}")