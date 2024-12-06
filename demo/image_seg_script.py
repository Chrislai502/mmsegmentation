import os
import time
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
import glob
import json

PAD = True

def pad_or_crop_image(image, target_height, target_width):
    """Pad or crop an image to the desired size (target_height, target_width)."""
    h, w, c = image.shape

    # If the image is smaller, pad with zeros
    if h < target_height or w < target_width:
        top_pad = max(0, (target_height - h) // 2)
        bottom_pad = max(0, target_height - h - top_pad)
        left_pad = max(0, (target_width - w) // 2)
        right_pad = max(0, target_width - w - left_pad)
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # If the image is larger, crop to the target size
    if h > target_height or w > target_width:
        top_crop = max(0, (h - target_height) // 2)
        bottom_crop = top_crop + target_height
        left_crop = max(0, (w - target_width) // 2)
        right_crop = left_crop + target_width
        image = image[top_crop:bottom_crop, left_crop:right_crop]

    return image

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def detect_model_type(model):
    """Detect whether the model is Transformer-based or CNN-based."""
    # Heuristic-based detection by inspecting module names
    transformer_keywords = ['Transformer', 'Attention']
    cnn_keywords = ['Conv', 'Conv2d']

    model_type = 'Unknown'
    for name, module in model.named_modules():
        module_name = type(module).__name__
        # Check for Transformer keywords first
        if any(keyword in module_name for keyword in transformer_keywords):
            model_type = 'Transformer'
            break
        # Check for CNN keywords after Transformer
        if any(keyword in module_name for keyword in cnn_keywords):
            model_type = 'CNN'
            break

    return model_type


def get_gpu_utilization(device_id=0):
    """Get GPU utilization using NVIDIA Management Library (NVML)."""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_id)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu, utilization.memory
    except Exception as e:
        print(f"GPU monitoring failed: {e}")
        return None, None

def get_all_paths_of_png(directory: str, suffix: str = ".png"):
    return glob.glob(f"{directory}/**/*{suffix}", recursive=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_folder', help='Path to the folder containing input images. Contains .png images as the data within each folder.')
    parser.add_argument('--output_folder', help='Output folder that will save the stats + metadata.')
    parser.add_argument('--filter_string', default=None, help='The path must also contain the keyword (eg image02)')
    parser.add_argument('--config', default=None, help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--with-labels', action='store_true', default=False, help='Whether to display the class labels.')
    parser.add_argument('--title', default='result', help='The image identifier.')
    parser.add_argument('--width', type=int, default=896, help='Target width for padding or cropping.')
    parser.add_argument('--height', type=int, default=896, help='Target height for padding or cropping.')
    args = parser.parse_args()
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        raise ValueError(f"Input folder '{input_folder}' does not exist or is not a directory.")
    
    all_png_paths = get_all_paths_of_png(input_folder)
    if args.filter_string:
        all_png_paths = [path for path in all_png_paths if args.filter_string in path]

    # Initialize the model
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Count parameters in the model
    total_params, trainable_params = count_parameters(model)
    total_params_str = f"{total_params:,}"  # Add commas to total parameters
    trainable_params_str = f"{trainable_params:,}"  # Add commas to trainable parameters

    # Detect model type
    model_type = detect_model_type(model)

    # Stats to collect
    inference_times = []
    processed_images = 0
    
    
    # iterating through each PNG and writing the result in a new path
    for png_path in tqdm(all_png_paths):
        print(f"Processing image {png_path}...")
    
        # Read the image
        image = cv2.imread(str(png_path))
        if image is None:
            print(f"Could not read image {png_path}. Skipping.")
            continue
    
        # Pad or crop the image to the target size
        if PAD:
            image = pad_or_crop_image(image, args.height, args.width)

        # Measure inference time
        start_time = time.time()
        result = inference_model(model, image)
        end_time = time.time()
        
        # Save inference time
        inference_times.append(end_time - start_time)
        
        # Saving image with same name, but as .npy file
        file_base_name, _ = os.path.splitext(png_path)
        np.save(f"{file_base_name}.npy", result.pred_sem_seg.data.cpu().numpy())
        
    # GPU utilization stats
    gpu_util, mem_util = get_gpu_utilization() if args.device.startswith('cuda') else (None, None)

    output_folder = Path(args.output_folder)
    # Save stats to a file
    stats_file = output_folder / "stats.txt"
    with open(stats_file, 'w') as f:
        # saving the segmentation classes metadata
        f.write(f"{json.dumps(model.dataset_meta, indent=4)}")
        
        # writing the additional collected stats/metdata for the segmentation process
        f.write(f"Processed images: {processed_images}\n")
        f.write(f"Input folder: {input_folder}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"Model config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Image dimensions: {args.width}x{args.height}\n")
        f.write(f"Total parameters: {total_params_str}\n")
        f.write(f"Trainable parameters: {trainable_params_str}\n")
        f.write(f"Model type: {model_type}\n")

        if inference_times:
            mean_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            f.write(f"Mean inference time (s): {mean_time:.4f}\n")
            f.write(f"Std inference time (s): {std_time:.4f}\n")

        if gpu_util is not None and mem_util is not None:
            f.write(f"GPU utilization (%): {gpu_util}\n")
            f.write(f"Memory utilization (%): {mem_util}\n")
    
    print(f"Processing complete! Stats saved to {stats_file}")


if __name__ == '__main__':
    main()
