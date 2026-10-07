import os
import time
import argparse
import psutil
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import shutil

from model import SimpleCNN
from monitor import SystemMonitor
from gcf import calculate_gcf


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_mapping(mapping_path):
    label_to_char = {}

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, "r") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) >= 2:
                label = int(parts[0])
                ascii_code = int(parts[1])
                label_to_char[label] = chr(ascii_code)

    return label_to_char


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")

    img_array = np.array(image)

    # If background is white, invert it.
    # After inversion: character becomes white, background becomes black.
    if img_array.mean() > 127:
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # Remove gray noise
    img_array = np.where(img_array > 50, 255, 0).astype(np.uint8)

    # Find character bounding box
    coords = np.argwhere(img_array > 0)

    if coords.size == 0:
        raise ValueError(f"No visible character found in image: {image_path}")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]
    cropped_image = Image.fromarray(cropped)

    # Resize while preserving aspect ratio
    cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center in 28x28 canvas
    canvas = Image.new("L", (28, 28), color=0)

    left = (28 - cropped_image.width) // 2
    top = (28 - cropped_image.height) // 2

    canvas.paste(cropped_image, (left, top))

    image_array = np.array(canvas).astype(np.float32) / 255.0

    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


def get_true_label_from_filename(image_path):
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    parts = name_without_ext.split("_")

    if len(parts) < 2:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            "Use format like upper_A_01.png, lower_a_01.png, or digit_5_01.png"
        )

    label_type = parts[0]
    label_value = parts[1]

    if label_type == "upper":
        return label_value.upper()

    elif label_type == "lower":
        return label_value.lower()

    elif label_type == "digit":
        return label_value

    else:
        raise ValueError(
            f"Invalid label type in filename: {filename}. "
            "Use prefix upper, lower, or digit."
        )


def get_image_files(image_dir, limit=None):
    files = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(image_dir, filename))

    files = sorted(files)

    if limit is not None:
        files = files[:limit]

    return files


def main():
    parser = argparse.ArgumentParser(description="PyTorch 100-image prediction benchmark")

    parser.add_argument("--image-dir", type=str, default="prediction_test")
    parser.add_argument("--model-path", type=str, default="models/pytorch_simple_cnn.pth")
    parser.add_argument("--mapping-path", type=str, default="data/emnist-balanced-mapping.txt")
    parser.add_argument("--num-classes", type=int, default=47)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--power-watt", type=float, default=5.0)

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    debug_dir = os.path.join("results", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image folder not found: {args.image_dir}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    device = torch.device("cpu")

    label_to_char = load_mapping(args.mapping_path)

    model = SimpleCNN(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    image_files = get_image_files(args.image_dir, limit=args.limit)

    if len(image_files) == 0:
        raise ValueError(f"No image files found in: {args.image_dir}")

    print(f"Total images found: {len(image_files)}")
    print("Starting PyTorch post-deployment prediction benchmark...")

    monitor = SystemMonitor(interval=0.1)
    monitor.start()

    prediction_records = []

    correct = 0
    total = 0

    start_total_time = time.time()

    with torch.no_grad():
        for image_path in image_files:
            true_label = get_true_label_from_filename(image_path)

            input_tensor = preprocess_image(image_path).to(device)

            start_prediction_time = time.time()

            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            end_prediction_time = time.time()

            predicted_class_id = predicted_class.item()
            predicted_label = label_to_char.get(predicted_class_id, str(predicted_class_id))
            confidence_percent = confidence.item() * 100

            prediction_time = end_prediction_time - start_prediction_time

            is_correct = predicted_label == true_label

            if is_correct:
                correct += 1
            else:
                try:
                    # copy misclassified image to debug folder for inspection
                    base = os.path.basename(image_path)
                    dest_name = f"{total+1:04d}_pred_{predicted_label}_true_{true_label}." + base.split('.')[-1]
                    shutil.copy2(image_path, os.path.join(debug_dir, dest_name))
                except Exception:
                    pass

            total += 1

            prediction_records.append({
                "image": os.path.basename(image_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_class_id": predicted_class_id,
                "confidence_percent": confidence_percent,
                "prediction_time_seconds": prediction_time,
                "correct": is_correct
            })

    end_total_time = time.time()

    monitor.stop()

    total_prediction_time = end_total_time - start_total_time
    average_latency = total_prediction_time / total
    throughput = total / total_prediction_time
    accuracy = correct / total

    monitor_summary = monitor.summary()

    gcf_score = calculate_gcf(
        power_watt=args.power_watt,
        training_time_seconds=total_prediction_time,
        memory_percent=monitor_summary["avg_memory_percent"],
        temperature_c=monitor_summary["avg_temperature_c"]
    )

    summary_result = {
        "framework": "PyTorch",
        "phase": "post_deployment_prediction",
        "total_images": total,
        "correct_predictions": correct,
        "prediction_accuracy": accuracy,
        "total_prediction_time_seconds": total_prediction_time,
        "average_latency_seconds_per_image": average_latency,
        "throughput_images_per_second": throughput,
        "avg_cpu_percent": monitor_summary["avg_cpu_percent"],
        "max_cpu_percent": monitor_summary["max_cpu_percent"],
        "avg_memory_percent": monitor_summary["avg_memory_percent"],
        "max_memory_percent": monitor_summary["max_memory_percent"],
        "avg_temperature_c": monitor_summary["avg_temperature_c"],
        "max_temperature_c": monitor_summary["max_temperature_c"],
        "estimated_power_watt": args.power_watt,
        "prediction_gcf_score": gcf_score
    }

    summary_df = pd.DataFrame([summary_result])
    detail_df = pd.DataFrame(prediction_records)

    summary_path = "results/pytorch_prediction_summary.csv"
    detail_path = "results/pytorch_prediction_details.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("\nPyTorch Prediction Summary:")
    print(summary_df.T)

    print("\nSaved files:")
    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()