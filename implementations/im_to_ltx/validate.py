import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import namedtuple


def flatten_and_encode(encoder_output):
    B, C, H, W = encoder_output.shape
    return encoder_output.view(B, C, -1).permute(0, 2, 1)  # (B, L, C)


def decode_step(decoder, input_token, encoder_seq, h, c, context):
    x_t = decoder.embedding(input_token.squeeze(1))  # (B, E)
    x = torch.cat([x_t, context], dim=-1)

    new_h, new_c = [], []
    for i, lstm in enumerate(decoder.lstm_layers):
        h_i, c_i = lstm(x, (h[i], c[i]))
        x = decoder.dropout(h_i)
        new_h.append(h_i)
        new_c.append(c_i)
    h, c = new_h, new_c

    context, _ = decoder.attention(h[-1], encoder_seq)
    combined = torch.cat([h[-1], context], dim=-1)
    attn_hidden = torch.tanh(decoder.context_projector(combined))
    logits = decoder.output_layer(attn_hidden)
    return logits, h, c, context


def beam_search_decode(model, image_tensor, tokenizer, beam_size=2, max_len=150, alpha=0.6, device='cuda'):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        enc_out = model.encoder(image_tensor)
        memory = flatten_and_encode(enc_out)

        mean_enc = memory.mean(dim=1)
        h0 = torch.tanh(model.decoder.init_hidden(mean_enc))
        c0 = torch.tanh(model.decoder.init_cell(mean_enc))

        h_init = [h0.clone() for _ in range(len(model.decoder.lstm_layers))]
        c_init = [c0.clone() for _ in range(len(model.decoder.lstm_layers))]
        context_init = torch.zeros_like(mean_enc)

        Beam = namedtuple("Beam", ["tokens", "logprob", "h", "c", "context"])
        beams = [Beam([tokenizer.start_token_id], 0.0, h_init, c_init, context_init)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                if beam.tokens[-1] == tokenizer.end_token_id:
                    completed.append(beam)
                    continue

                input_token = torch.tensor([[beam.tokens[-1]]], device=device)
                logits, h, c, context = decode_step(
                    model.decoder, input_token, memory, beam.h, beam.c, beam.context)
                log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
                topk_probs, topk_ids = log_probs.topk(beam_size)

                for k in range(beam_size):
                    tok = topk_ids[k].item()
                    new_seq = beam.tokens + [tok]
                    new_score = beam.logprob + topk_probs[k].item()
                    new_beams.append(Beam(new_seq, new_score, h, c, context))

            beams = sorted(new_beams, key=lambda b: b.logprob / (len(b.tokens) ** alpha), reverse=True)[:beam_size]
            if all(b.tokens[-1] == tokenizer.end_token_id for b in beams):
                break

        final = completed if completed else beams
        best = max(final, key=lambda b: b.logprob / (len(b.tokens) ** alpha))
        pred_ids = best.tokens
        if tokenizer.end_token_id in pred_ids:
            pred_ids = pred_ids[:pred_ids.index(tokenizer.end_token_id)]
        return " ".join(tokenizer.decode(pred_ids[1:]))  # skip <START>


def validate(model, dataloaders, tokenizer, device='cuda', show_samples=30, output_file=None):
    model.to(device)
    model.eval()
    bleu_scores = []
    smooth_fn = SmoothingFunction().method4

    predictions = []
    shown = 0

    print("\nRunning Validation...")

    for bucket_size, loader in dataloaders.items():
        print(f"Validating Bucket {bucket_size} with {len(loader.dataset)} samples")

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            for i in range(images.size(0)):
                img = images[i]
                tgt_ids = targets[i].tolist()

                ref_tokens = [
                    tokenizer.id_to_token[i]
                    for i in tgt_ids
                    if i not in {tokenizer.pad_token_id, tokenizer.start_token_id, tokenizer.end_token_id}
                ]
                reference = " ".join(ref_tokens)
                reference_tokens = reference.split()

                predicted_str = beam_search_decode(model, img, tokenizer, device=device)
                predicted_tokens = predicted_str.split()

                bleu = sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)

                if shown < show_samples:
                    predictions.append((reference, predicted_str))
                    shown += 1

            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                print(f"Bucket {bucket_size} | Batch {batch_idx+1}/{len(loader)}")

    print(f"\nShowing {len(predictions)} Prediction Samples:\n")
    for idx, (ref, pred) in enumerate(predictions):
        print(f"Sample {idx + 1}")
        print(f"GT  : {ref}")
        print(f"PRED: {pred}")
        print("-" * 60)

    if output_file:
        with open(output_file, "w") as f:
            for idx, (ref, pred) in enumerate(predictions):
                f.write(f"Sample {idx + 1}\n")
                f.write(f"GT  : {ref}\n")
                f.write(f"PRED: {pred}\n")
                f.write("-" * 60 + "\n")
        print(f"Predictions saved to: {output_file}")

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\nValidation Complete. Average BLEU Score: {avg_bleu:.4f}")
    return avg_bleu


def load_model_and_optimizer(checkpoint_path, tokenizer, device='cuda'):
    from model import ImageToLatexModel  # ensure import works
    model = ImageToLatexModel(vocab_size=len(tokenizer.token_to_id)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

if __name__ == "__main__":
    from utils import Tokenizer, BucketedDataset
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer("vocabulary path")
    transform = transforms.ToTensor()
    dataset = BucketedDataset("json for validation/testing", tokenizer, transform)
    loaders = dataset.get_dataloaders(batch_size=8)

    model, optimizer = load_model_and_optimizer("checkpoint_epoch22.pt", tokenizer, device=device)
    val_bleu = validate(model, loaders, tokenizer, device=device, show_samples=30)

