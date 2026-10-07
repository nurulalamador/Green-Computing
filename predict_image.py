import os
import argparse
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

from model import SimpleCNN


def load_mapping(mapping_path):
    label_to_char = {}

    if not os.path.exists(mapping_path):
        print("Mapping file not found. Prediction will show class number only.")
        return label_to_char

    with open(mapping_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[0])
                ascii_code = int(parts[1])
                label_to_char[label] = chr(ascii_code)

    return label_to_char


def preprocess_image(image_path, save_debug=True):
    image = Image.open(image_path).convert("L")

    # Convert to numpy
    img_array = np.array(image)

    # If background is white, invert it.
    # After this: character = white, background = black
    if img_array.mean() > 127:
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # Threshold: remove gray noise
    img_array = np.where(img_array > 50, 255, 0).astype(np.uint8)

    # Find bounding box of character
    coords = np.argwhere(img_array > 0)

    if coords.size == 0:
        raise ValueError("No visible character found in the image.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop only character area
    cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]
    cropped_image = Image.fromarray(cropped)

    # Resize while preserving aspect ratio
    cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Put in center of 28x28 black canvas
    canvas = Image.new("L", (28, 28), color=0)

    left = (28 - cropped_image.width) // 2
    top = (28 - cropped_image.height) // 2

    canvas.paste(cropped_image, (left, top))

    if save_debug:
        os.makedirs("debug", exist_ok=True)
        canvas.save("debug/processed_input.png")
        print("Processed image saved at: debug/processed_input.png")

    image_array = np.array(canvas).astype(np.float32) / 255.0

    # Shape: batch x channel x height x width
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


def predict(image_path, model_path, mapping_path, num_classes=47, top_k=5):
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    label_to_char = load_mapping(mapping_path)

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    print("\nPrediction Result:")
    print("------------------")

    for i in range(top_k):
        class_id = top_indices[0][i].item()
        confidence = top_probs[0][i].item() * 100
        predicted_char = label_to_char.get(class_id, str(class_id))

        print(
            f"Top {i + 1}: {predicted_char} | "
            f"Class: {class_id} | "
            f"Confidence: {confidence:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict digit or letter using trained EMNIST CNN model"
    )

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="models/pytorch_simple_cnn.pth")
    parser.add_argument("--mapping-path", type=str, default="data/emnist-balanced-mapping.txt")
    parser.add_argument("--num-classes", type=int, default=47)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    predict(
        image_path=args.image,
        model_path=args.model_path,
        mapping_path=args.mapping_path,
        num_classes=args.num_classes,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()