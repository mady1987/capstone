import torch
import torch.nn as nn
import timm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.vit.blocks[-1].parameters():
            p.requires_grad = True
        for p in self.vit.norm.parameters():
            p.requires_grad = True

    def forward(self, images):
        x = self.vit.forward_features(images)  # [B, 197, 768]
        return x[:, 1:, :]                     # remove CLS

class TransformerCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=256,
        encoder_dim=768,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos = PositionalEncoding(embed_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.enc_proj = nn.Linear(encoder_dim, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

        # optional weight tying
        self.fc.weight = self.embed.weight

    def forward(self, encoder_out, captions):
        tgt = self.pos(self.embed(captions))
        memory = self.enc_proj(encoder_out)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            captions.size(1)
        ).to(captions.device)

        tgt_key_padding_mask = (captions == 0)

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.fc(out)