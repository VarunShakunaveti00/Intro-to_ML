import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time

from model import ImageToLatexModel
from utils import Tokenizer, BucketedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = Tokenizer("vocabulary path")
transform = transforms.ToTensor()

dataset = BucketedDataset(" the jsonl file for training", tokenizer, transform)
dataset.print_bucket_summary()
loaders = dataset.get_dataloaders(batch_size=8)

model = ImageToLatexModel(vocab_size=len(tokenizer.token_to_id)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")

def train_one_epoch(model, dataloaders, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_samples = 0
    start_time = time.time()

    print(f"Epoch {epoch} Training")

    for bucket_size, loader in dataloaders.items():
        print(f"Bucket {bucket_size} - {len(loader.dataset)} samples")

        for batch_idx, (images, target) in enumerate(loader):
            images = images.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(images, target[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            total_samples += images.size(0)

            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                print(f"Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Samples: {total_samples}")

        print(f"Completed bucket {bucket_size}")

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
    return avg_loss

start_epoch = 1
num_epochs = 22

for epoch in range(start_epoch, num_epochs + 1):
    avg_loss = train_one_epoch(model, loaders, criterion, optimizer, device, epoch)

    checkpoint_path = f"checkpoint_epoch{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
