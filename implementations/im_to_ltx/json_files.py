import os
import json

with open("formulas file", "r", encoding="utf-8") as f:
    formulas = [line.strip() for line in f]

train_map = {}

with open("the lst file for training/testing/validation", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            image_name, index_str = parts
            index = int(index_str)
            if 0 <= index < len(formulas):
                train_map[image_name] = formulas[index]

image_dir = "images"
train_pairs = []

for image_name, formula in train_map.items():
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        train_pairs.append((image_path, formula))
    else:
        print(f"Image not found: {image_path}")

output_path = "saving the jsonl file"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

#saving the images(path), formula into a jsonl file.

with open(output_path, "w", encoding="utf-8") as f:
    for path, formula in train_pairs:
        f.write(json.dumps({"image": path, "formula": formula}) + "\n")

print(f"Saved training data to: {output_path}")
