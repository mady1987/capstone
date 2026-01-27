import torch.nn as nn
import torchvision.models as models
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import timm

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained = True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)   #[B, 2048]
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) #[2048, embed_size(256)]
        self.bn = nn.BatchNorm1d(embed_size) # batch normalization [embed_size]
    
    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

# VIT Encoder 
class EncoderViT(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0   # returns features directly
        )
        self.embed = nn.Linear(self.vit.num_features, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        x = self.vit.patch_embed(images)
        cls_token = self.vit.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        cls = x[:, 0]
        mean = x[:, 1:].mean(dim=1)

        features = cls + mean
        features = self.embed(features)
        features = self.norm(features)
        features = self.dropout(features)
        return features
        
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.text_proj = nn.Linear(hidden_size, embed_size)

    def forward(self, features, captions, lengths):
        """
        features: [batch, embed_size]
        captions: [batch, max_len]
        lengths:  [batch]
        """

        # ---- Teacher forcing: feed caption[:-1] ----
        captions_in = captions[:, :-1]   # remove <end> token

        embeddings = self.embed(captions_in)    # [B, T-1, embed_size]

        # ---- Concatenate image feature as the first token ----
        features = features.unsqueeze(1)         # [B, 1, embed_size]
        embeddings = torch.cat((features, embeddings), dim=1)   # [B, T, embed_size]

        # ---- Pack padded sequences ----
        packed = pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # ---- Forward LSTM ----
        packed_outputs, _ = self.lstm(packed)

        # ---- Unpack ----
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )
        # outputs shape: [B, T, hidden_size]

        # ---- Map to vocab ----
        outputs = self.linear(outputs)      # [B, T, vocab_size]

        return outputs
