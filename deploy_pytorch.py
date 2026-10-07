import os
import time
import argparse
import psutil
import pandas as pd
import torch

from model import SimpleCNN
from monitor import SystemMonitor


def get_file_size_mb(file_path):
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="PyTorch deployment benchmark")

    parser.add_argument("--model-path", type=str, default="models/pytorch_simple_cnn.pth")
    parser.add_argument("--num-classes", type=int, default=47)

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    device = torch.device("cpu")

    model_file_size_mb = get_file_size_mb(args.model_path)

    memory_before_mb = get_process_memory_mb()

    monitor = SystemMonitor(interval=0.1)

    print("Starting PyTorch deployment benchmark...")
    print(f"Loading model from: {args.model_path}")

    start_time = time.time()
    monitor.start()

    model = SimpleCNN(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Dummy input to make sure the model is ready for inference
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        _ = model(dummy_input)

    monitor.stop()
    end_time = time.time()

    memory_after_mb = get_process_memory_mb()

    loading_time_seconds = end_time - start_time
    memory_increase_mb = memory_after_mb - memory_before_mb

    monitor_summary = monitor.summary()

    result = {
        "framework": "PyTorch",
        "model_path": args.model_path,
        "model_file_size_mb": model_file_size_mb,
        "model_loading_time_seconds": loading_time_seconds,
        "memory_before_loading_mb": memory_before_mb,
        "memory_after_loading_mb": memory_after_mb,
        "memory_increase_mb": memory_increase_mb,
        "avg_cpu_percent": monitor_summary["avg_cpu_percent"],
        "max_cpu_percent": monitor_summary["max_cpu_percent"],
        "avg_memory_percent": monitor_summary["avg_memory_percent"],
        "max_memory_percent": monitor_summary["max_memory_percent"],
        "avg_temperature_c": monitor_summary["avg_temperature_c"],
        "max_temperature_c": monitor_summary["max_temperature_c"]
    }

    df = pd.DataFrame([result])
    output_path = "results/pytorch_deployment_metrics.csv"
    df.to_csv(output_path, index=False)

    print("\nPyTorch Deployment Result:")
    print(df.T)

    print("\nSaved file:")
    print(output_path)


if __name__ == "__main__":
    main()