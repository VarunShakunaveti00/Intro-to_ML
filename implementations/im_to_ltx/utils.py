import json
import os
import torch
from PIL import Image
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r") as f:
            token_to_id = json.load(f)["token_to_id"]

        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

        self.pad_token_id = token_to_id["<PAD>"]
        self.start_token_id = token_to_id["<START>"]
        self.end_token_id = token_to_id["<END>"]
        self.unk_token_id = token_to_id["<UNK>"]

    def encode(self, formula_str):
        tokens = formula_str.strip().split()
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        return [self.start_token_id] + ids + [self.end_token_id]

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids if i not in {
            self.pad_token_id, self.start_token_id, self.end_token_id}]

class BucketedDataset:
    def __init__(self, jsonl_path, tokenizer, transform=None):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.transform = transform or transforms.ToTensor()
        self.buckets = defaultdict(list)
        self._load_and_bucket()

    def _load_and_bucket(self):
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    image_path = Path(data["image"])
                    formula = data["formula"]

                    if not image_path.exists():
                        image_path = Path(self.jsonl_path.parent) / image_path
                    if not image_path.exists():
                        continue

                    with Image.open(image_path) as img:
                        img = img.convert("L")
                        h, w = img.size[::-1]

                    token_ids = self.tokenizer.encode(formula)
                    self.buckets[(h, w)].append((image_path, token_ids))

                except Exception as e:
                    print(f"Skipping: {e}")

    def print_bucket_summary(self):
        print()
        for size in sorted(self.buckets):
            print(f"{size}: {len(self.buckets[size])} images")
        print(f"\nTotal buckets: {len(self.buckets)}")

    def get_dataloaders(self, batch_size=16, shuffle=True):
        loaders = {}
        for size, samples in self.buckets.items():
            dataset = SingleBucketDataset(samples, self.transform)
            loaders[size] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate)
        return loaders

    def _collate(self, batch):
        images, seqs = zip(*batch)
        images = torch.stack(images)
        max_len = max(len(s) for s in seqs)
        padded = torch.full((len(seqs), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return images, padded

class SingleBucketDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label_ids = self.data[idx]
        with Image.open(path) as img:
            img = img.convert("L")
            img = self.transform(img)
        return img, label_ids
    
    
__all__ = ["Tokenizer", "BucketedDataset"]
