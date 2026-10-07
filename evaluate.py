import os
import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import ImageOps

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import SimpleCNN


def get_num_classes(split):
    class_map = {
        "balanced": 47,
        "byclass": 62,
        "bymerge": 47,
        "digits": 10,
        "letters": 27,
        "mnist": 10
    }

    return class_map.get(split, 47)


def build_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.mirror(img.rotate(-90))),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--split", type=str, default="balanced")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-test", type=int, default=200)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-path", type=str, default="models/pytorch_simple_cnn.pth")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    device = torch.device("cpu")

    transform = build_transform()

    test_dataset = datasets.EMNIST(
        root=args.data_dir,
        split=args.split,
        train=False,
        download=args.download,
        transform=transform
    )

    if args.limit_test is not None:
        test_dataset = Subset(test_dataset, range(min(args.limit_test, len(test_dataset))))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    num_classes = get_num_classes(args.split)

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
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
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    result = {
        "framework": "PyTorch",
        "dataset": "EMNIST",
        "split": args.split,
        "test_samples": len(test_dataset),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    df = pd.DataFrame([result])
    df.to_csv("results/pytorch_evaluation.csv", index=False)

    print("\nEvaluation Result:")
    print(df)

    print("\nSaved file:")
    print("results/pytorch_evaluation.csv")


if __name__ == "__main__":
    main()