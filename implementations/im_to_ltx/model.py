import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.d_model = d_model

    def forward(self, x):
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        d_model = self.d_model
        d_quarter = d_model // 4

        pe = torch.zeros(d_model, H, W, device=device, dtype=dtype)

        div_term = torch.exp(torch.arange(d_quarter, device=device, dtype=dtype) *
                             -(math.log(10000.0) / d_quarter))

        pos_w = torch.arange(W, device=device, dtype=dtype)
        pos_h = torch.arange(H, device=device, dtype=dtype)

        # (d_quarter, W)
        sin_w = torch.sin(pos_w[None, :] * div_term[:, None])
        cos_w = torch.cos(pos_w[None, :] * div_term[:, None])

        # (d_quarter, H)
        sin_h = torch.sin(pos_h[None, :] * div_term[:, None])
        cos_h = torch.cos(pos_h[None, :] * div_term[:, None])

        pe[0:d_quarter, :, :] = sin_w[:, None, :].expand(-1, H, -1)
        pe[d_quarter:2*d_quarter, :, :] = cos_w[:, None, :].expand(-1, H, -1)
        pe[2*d_quarter:3*d_quarter, :, :] = sin_h[:, :, None].expand(-1, -1, W)
        pe[3*d_quarter:4*d_quarter, :, :] = cos_h[:, :, None].expand(-1, -1, W)

        return x + pe.unsqueeze(0)


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2

            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4

            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(256),
            #nn.MaxPool2d(kernel_size=2, stride=2),  # H/8, W/8
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),

            # Stage 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),  # W/16
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),

            # Stage 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))  # H/16
        )

        self.pos_encoder = PositionalEncoding2D(d_model=512)

    def forward(self, x):
        features = self.conv(x)  # (B, 512, H/16, W/16)
        features = self.pos_encoder(features)
        return features
    
class Attention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_feature_dim, attn_dim):
        super().__init__()
        self.attn_h = nn.Linear(decoder_hidden_dim, attn_dim)
        self.attn_enc = nn.Linear(encoder_feature_dim, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, 2*H), encoder_outputs: (B, L, C)
        enc_proj = self.attn_enc(encoder_outputs)         # (B, L, A)
        dec_proj = self.attn_h(decoder_hidden).unsqueeze(1)  # (B, 1, A)
        scores = self.attn_v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B, L)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, C)
        return context, attn_weights
    
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=512, feature_dim=512, attn_dim=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.init_hidden = nn.Linear(feature_dim, hidden_dim)
        self.init_cell = nn.Linear(feature_dim, hidden_dim)

        self.attention = Attention(
            decoder_hidden_dim=hidden_dim,
            encoder_feature_dim=feature_dim,
            attn_dim=attn_dim
        )

        # One LSTM for each layer
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(input_size=(embed_dim + feature_dim) if i == 0 else hidden_dim, hidden_size=hidden_dim)
            for i in range(4)
        ])

        self.dropout = nn.Dropout(dropout)
        self.context_projector = nn.Linear(hidden_dim + feature_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_output, tgt_seq):
        """
        encoder_output: (B, C=512, H, W)
        tgt_seq: (B, T)
        """
        B, C, H, W = encoder_output.shape
        encoder_seq = encoder_output.view(B, C, -1).permute(0, 2, 1)  # (B, L, C)

        mean_enc = encoder_seq.mean(dim=1)  # (B, C)
        h = [torch.tanh(self.init_hidden(mean_enc)) for _ in range(4)]  # list of (B, H)
        c = [torch.tanh(self.init_cell(mean_enc)) for _ in range(4)]

        embedded = self.embedding(tgt_seq)  # (B, T, E)
        context = torch.zeros(B, self.feature_dim, device=tgt_seq.device)
        outputs = []

        for t in range(tgt_seq.size(1)):
            x_t = embedded[:, t, :]  # (B, E)
            x = torch.cat([x_t, context], dim=-1)  # (B, E + C)

            new_h, new_c = [], []
            for i, lstm in enumerate(self.lstm_layers):
                h_i, c_i = lstm(x, (h[i], c[i]))
                x = self.dropout(h_i)
                new_h.append(h_i)
                new_c.append(c_i)
            h, c = new_h, new_c

            context, attn_weights = self.attention(h[-1], encoder_seq)

            combined = torch.cat([h[-1], context], dim=-1)
            attn_hidden = torch.tanh(self.context_projector(combined))
            output = self.output_layer(attn_hidden)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, T, vocab_size)
    
class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=512, feature_dim=512, attn_dim=256, dropout=0.4):
        super(ImageToLatexModel, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            attn_dim=attn_dim,
            dropout=dropout
        )

    def forward(self, images, tgt_seq):
        encoder_output = self.encoder(images)  # (B, C=512, H', W')
        logits = self.decoder(encoder_output, tgt_seq)  # (B, T, vocab_size)
        return logits
    

if __name__ == "__main__":
    vocab_size = 495 

    model = ImageToLatexModel(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=512,
        feature_dim=512,
        attn_dim=256,
        dropout=0.4
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nParameter breakdown by layer:")
    for name, param in model.named_parameters():
        print(f"{name:60} {param.numel():>10}")
    
__all__ = ["ImageToLatexModel"]


