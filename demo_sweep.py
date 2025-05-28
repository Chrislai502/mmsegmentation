import os
import re
import subprocess

# Paths and folders
configs_dir = "configs"
data_input_folder = "/home/art/Depth-Anything/semseg/data/to_ashwin"
data_output_base = "/home/art/Depth-Anything/semseg/data"
checkpoints_dir = "./checkpoints"
demo_script = "demo/image_demo_folder.py"

# Ensure checkpoints directory exists
os.makedirs(checkpoints_dir, exist_ok=True)

# Regex to match Cityscapes-trained configs and extract resolution
cityscapes_pattern = r"cityscapes-(\d+)x(\d+)"
resolution_pattern = re.compile(cityscapes_pattern)

# Function to run a shell command
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(e.stderr.decode())
        return False

# Step 1: Collect all Cityscapes-trained configs and their resolutions
configs_to_process = []  # List to hold (config_path, config_foldername, width, height)

for root, dirs, files in os.walk(configs_dir):
    # Skip the `_base_` folder
    if "_base_" in root:
        continue

    for file in files:
        if file.endswith(".py"):
            config_path = os.path.join(root, file)
            config_filename_only = os.path.splitext(file)[0]

            # Check if the config is for Cityscapes and extract resolution
            match = resolution_pattern.search(file)
            if not match:
                continue  # Skip configs that are not trained on Cityscapes

            model_native_height, model_native_width = match.groups()
            configs_to_process.append((config_path, config_filename_only, model_native_width, model_native_height))

# Step 2: Print all Cityscapes-trained configs and their resolutions
print("The following configs will be processed:\n")
for config_path, config_filename_only, model_native_width, model_native_height in configs_to_process:
    print(f"Config: {config_path}")
    print(f"  Model File Config: {config_filename_only}")
    print(f"  Resolution: {model_native_width}x{model_native_height}\n")

# Step 3: Process each config
for config_path, config_filename_only, model_native_width, model_native_height in configs_to_process:
    print(f"Processing config: {config_path}")

    # Step 3.1: Download the model
    download_command = f"mim download mmsegmentation --config {config_filename_only} --dest {checkpoints_dir}"
    if not run_command(download_command):
        print(f"Failed to download model for config: {config_path}")
        continue

    # Find the latest downloaded checkpoint file with .pth or .pt extension
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(('.pth', '.pt'))]
    if not checkpoint_files:
        print("No valid checkpoint files (.pth or .pt) found in the checkpoints directory.")
        continue

    checkpoint_file = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
    print(f"Using checkpoint: {checkpoint_path}")

    # Step 3.2: Evaluate the model
    output_folder = os.path.join(data_output_base, config_filename_only)
    os.makedirs(output_folder, exist_ok=True)

    eval_command = (
        f"python {demo_script} "
        f"--input_folder {data_input_folder} "
        f"--output_folder {output_folder} "
        f"--config {config_path} "
        f"--checkpoint {checkpoint_path} "
        f"--device cuda:0 "
        f"--width {model_native_width} "
        f"--height {model_native_height}"
    )

    if not run_command(eval_command):
        print(f"Evaluation failed for config: {config_path}")
        continue

    print(f"Successfully processed {config_path}\n")
