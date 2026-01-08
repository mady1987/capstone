import torch.nn as nn
import torchvision.models as models
import torch
import timm
import math
    
class  TransformerEncoderViT(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0   # returns features directly
        )
        for p in self.vit.parameters():
            p.requires_grad = False

        self.feat_dim = self.vit.num_features  # 768
        self.proj = nn.Linear(self.feat_dim, embed_size)

    def forward(self, images):
        """
        images: [B, 3, 224, 224]
        returns: [B, N, embed_size]
        """
        # timm returns patch tokens if forward_features is used
        tokens = self.vit.forward_features(images)  # [B, N+1, 768]

        # Remove CLS token
        patch_tokens = tokens[:, 1:, :]             # [B, N, 768]

        patch_tokens = self.proj(patch_tokens)      # [B, N, embed_size]
        return patch_tokens
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, max_len=80, num_layers=4, nhead=8, ff_dim=2048, dropout=0.1, pad_idx=0 ):
        super(TransformerDecoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed  = nn.Embedding(max_len, embed_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
        
        self.pad_idx = pad_idx
        self.embed_size = embed_size
        
    def forward(self, encoder_out, captions):
        """
        encoder_out: [B, N, D]
        captions   : [B, T]
        """

        B, T = captions.shape
        device = captions.device

        positions = torch.arange(T, device=device).unsqueeze(0) # [1, T]
        x = self.embed(captions) + self.pos_embed(positions)
        x = x * math.sqrt(self.embed_size)

        # padding_mask = (captions == self.pad_idx)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(captions.device)
        
        out = self.decoder(
            tgt=x,
            memory=encoder_out,
            tgt_mask=attn_mask,
            # tgt_key_padding_mask=padding_mask,
            tgt_is_causal=True 
        )

        logits = self.fc(out)
        return logits
