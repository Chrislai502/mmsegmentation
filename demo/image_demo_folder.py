import os
import time
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

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

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_folder', help='Path to the folder containing input images')
    parser.add_argument('--output_folder', help='Path to the folder to save output images')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
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
    
    # Create output folder if it doesn't exist
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

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

    # Process each image in the input folder
    for img_file in input_folder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print(f"Processing {img_file.name}...")
            processed_images += 1

            # Read the image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Could not read image {img_file}. Skipping.")
                continue

            # Pad or crop the image to the target size
            if PAD:
                image = pad_or_crop_image(image, args.height, args.width)

            # Measure inference time
            start_time = time.time()
            result = inference_model(model, image)
            end_time = time.time()
            
            breakpoint()
            
            # Save inference time
            inference_times.append(end_time - start_time)

            # Define output file path
            out_file = output_folder / img_file.name
            
            # Save the result
            show_result_pyplot(
                model,
                image,
                result,
                title=args.title,
                opacity=args.opacity,
                with_labels=args.with_labels,
                draw_gt=False,
                show=False,
                out_file=str(out_file)
            )
            print(f"Saved result to {out_file}")

    # GPU utilization stats
    gpu_util, mem_util = get_gpu_utilization() if args.device.startswith('cuda') else (None, None)

    # Save stats to a file
    stats_file = output_folder / "stats.txt"
    with open(stats_file, 'w') as f:
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
