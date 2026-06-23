import os
import glob

# Get absolute path of the workspace root
workspace_root = os.path.abspath(os.getcwd())

# Directories
config_dir = os.path.join(workspace_root, 'configs')
script_dir = os.path.join(workspace_root, 'batch_scripts')
log_dir = os.path.join(workspace_root, 'logs')
run_script_path = os.path.join(workspace_root, 'run_simulation.py')

os.makedirs(script_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Template (based on genPolImages.sh)
template = """#!/bin/bash
#SBATCH -t 1:00:00             # time limit set to 1 hour
#SBATCH --mem=48G               # 48G of memory are reserved
#SBATCH -J {job_name}              # the job is named
#SBATCH -o {log_dir}/{job_name}.%j.out	# Output: %j expands to jobid
#SBATCH -e {log_dir}/{job_name}.%j.err	# Error: %j expands to jobid
#SBATCH --mail-type=END      # an email is send at the end of the job
#SBATCH -n 8                 # 8 processors to be used
#SBATCH --gres=gpu:1       # Get a volta GPU
#SBATCH -N 1                 # 1 node is used

module purge				# don't inherit, use clean environment

HOST=`hostname`
echo " Slurm scheduled it on node $HOST"
# Run the simulation with the specific config file
python3 {run_script_path} {config_path}


"""

# Get all config files
config_files = sorted(glob.glob(os.path.join(config_dir, '*.json')))
submit_script_lines = ["#!/bin/bash"]

print(f"Found {len(config_files)} config files.")

for config_path in config_files:
    # Extract name for job
    filename = os.path.basename(config_path)
    name_part = os.path.splitext(filename)[0] # e.g., config_strips1000_speedneg00075
    
    # Create a job name (shortened if necessary, but SLURM handles reasonable lengths)
    job_name = name_part.replace("config_", "job_")
    
    # Fill the template
    script_content = template.format(
        job_name=job_name,
        log_dir=log_dir,
        config_path=config_path,
        run_script_path=run_script_path
    )
    
    # Write the batch script
    script_path = os.path.join(script_dir, f"{job_name}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Add to the master submit script
    submit_script_lines.append(f"sbatch {script_path}")


# Write the master submit script
submit_all_path = "submit_all_jobs.sh"
with open(submit_all_path, "w") as f:
    f.write("\n".join(submit_script_lines))
    f.write("\n")

# Make the submit script executable
os.chmod(submit_all_path, 0o755)

print(f"Generated {len(config_files)} batch scripts in '{script_dir}/'")
print(f"Generated master submit script: '{submit_all_path}'")
