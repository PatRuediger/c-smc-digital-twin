import json
import os
import copy

# Define the parameter ranges
# Assuming 100 -> 1000 and 200 -> 2000 based on the sequence 1250, 1500, 1750
strips_values = [1000, 1250, 1500, 1750, 2000]
speed_values = [-0.0075, -0.0125, -0.025, -0.05]

# Base config file path
base_config_path = 'config.json'
output_dir = 'configs'

# Read the base config
with open(base_config_path, 'r') as f:
    base_config = json.load(f)

# Generate configs
for strips in strips_values:
    for speed in speed_values:
        # Create a deep copy of the base config
        new_config = copy.deepcopy(base_config)
        
        # Update parameters
        new_config['simulation_parameters']['number_of_strips'] = strips
        new_config['simulation_parameters']['belt_speed'] = speed
        
        # Update description
        new_config['simulation_run']['description'] = f"Generated config: strips={strips}, speed={speed}"
        
        # Update output paths to avoid overwriting
        # We append a suffix to the existing paths
        # Format speed string for filename (remove -, replace . with _)
        speed_str = str(speed).replace('.', '').replace('-', 'neg')
        suffix = f"strips{strips}_speed{speed_str}"
        
        original_db_path = new_config['output_paths']['db_output_path']
        # Remove trailing slash if present for clean joining
        original_db_path = original_db_path.rstrip('/')
        
        new_db_path = f"{original_db_path}_{suffix}"
        new_render_path = os.path.join(new_db_path, "imgs")
        
        new_config['output_paths']['db_output_path'] = new_db_path
        new_config['output_paths']['render_output_path'] = new_render_path
        
        # Define new config filename
        config_filename = f"config_{suffix}.json"
        config_path = os.path.join(output_dir, config_filename)
        
        # Write the new config file
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
            
        print(f"Generated {config_path}")

print("Done generating configuration files.")
