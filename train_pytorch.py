import os
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from model import SimpleCNN
from monitor import SystemMonitor
from gcf import calculate_gcf


class EMNISTCSV(Dataset):
    def __init__(self, csv_path, limit=None, fix_orientation=True):
        print(f"Loading dataset from: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        if limit is not None:
            data = pd.read_csv(csv_path, header=None, nrows=limit)
        else:
            data = pd.read_csv(csv_path, header=None)

        # First column = label
        labels = data.iloc[:, 0].values.astype(np.int64)

        # Remaining 784 columns = 28x28 image pixels
        images = data.iloc[:, 1:].values.astype(np.float32)

        # Normalize pixel values: 0-255 to 0-1
        images = images / 255.0

        # Reshape to image format: N x 28 x 28
        images = images.reshape(-1, 28, 28)

        # EMNIST images are often rotated/flipped.
        # This fixes orientation for better image representation.
        if fix_orientation:
            fixed_images = []
            for img in images:
                img = np.rot90(img, k=-1)   # rotate clockwise
                img = np.fliplr(img)        # mirror horizontally
                fixed_images.append(img)

            images = np.array(fixed_images, dtype=np.float32)

        # Convert to PyTorch shape: N x C x H x W
        images = images.reshape(-1, 1, 28, 28)

        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        print(f"Loaded {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def evaluate_model(model, test_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Train Simple CNN on EMNIST CSV dataset")

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--train-file", type=str, default="emnist-balanced-train.csv")
    parser.add_argument("--test-file", type=str, default="emnist-balanced-test.csv")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--limit-train", type=int, default=1000)
    parser.add_argument("--limit-test", type=int, default=200)

    parser.add_argument("--num-classes", type=int, default=47)
    parser.add_argument("--power-watt", type=float, default=5.0)

    parser.add_argument("--no-fix-orientation", action="store_true")

    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cpu")
    print("Using device:", device)

    train_csv_path = os.path.join(args.data_dir, args.train_file)
    test_csv_path = os.path.join(args.data_dir, args.test_file)

    fix_orientation = not args.no_fix_orientation

    train_dataset = EMNISTCSV(
        csv_path=train_csv_path,
        limit=args.limit_train,
        fix_orientation=fix_orientation
    )

    test_dataset = EMNISTCSV(
        csv_path=test_csv_path,
        limit=args.limit_test,
        fix_orientation=fix_orientation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = SimpleCNN(num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    monitor = SystemMonitor(interval=1.0)

    print("\nTraining started...")

    start_time = time.time()
    monitor.start()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

    monitor.stop()
    end_time = time.time()

    training_time = end_time - start_time

    print("\nTraining finished.")
    print(f"Training Time: {training_time:.2f} seconds")

    print("\nEvaluating model...")

    accuracy, precision, recall, f1 = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )

    monitor_summary = monitor.summary()

    gcf_score = calculate_gcf(
        power_watt=args.power_watt,
        training_time_seconds=training_time,
        memory_percent=monitor_summary["avg_memory_percent"],
        temperature_c=monitor_summary["avg_temperature_c"]
    )

    model_path = "models/pytorch_simple_cnn.pth"
    scripted_model_path = "models/pytorch_simple_cnn_scripted.pt"

    torch.save(model.state_dict(), model_path)

    example_input = torch.randn(1, 1, 28, 28)
    traced_model = torch.jit.trace(model.cpu(), example_input)
    traced_model.save(scripted_model_path)

    result = {
        "framework": "PyTorch",
        "dataset": "EMNIST Balanced CSV",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        "training_time_seconds": training_time,

        "avg_cpu_percent": monitor_summary["avg_cpu_percent"],
        "max_cpu_percent": monitor_summary["max_cpu_percent"],

        "avg_memory_percent": monitor_summary["avg_memory_percent"],
        "max_memory_percent": monitor_summary["max_memory_percent"],

        "avg_temperature_c": monitor_summary["avg_temperature_c"],
        "max_temperature_c": monitor_summary["max_temperature_c"],

        "estimated_power_watt": args.power_watt,
        "gcf_score": gcf_score
    }

    df = pd.DataFrame([result])

    result_path = "results/pytorch_results.csv"
    df.to_csv(result_path, index=False)

    print("\nFinal Result:")
    print(df)

    print("\nSaved files:")
    print(f"1. {model_path}")
    print(f"2. {scripted_model_path}")
    print(f"3. {result_path}")


if __name__ == "__main__":
    main()