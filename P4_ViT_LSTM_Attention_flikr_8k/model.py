import torch.nn as nn
import torchvision.models as models
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import timm
    
class AttentionEncoderViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        # 🔓 Unfreeze only last block + norm
        for p in self.vit.parameters():
            p.requires_grad = False

        for p in self.vit.blocks[-1].parameters():
            p.requires_grad = True

        for p in self.vit.norm.parameters():
            p.requires_grad = True

    def forward(self, images):
        return self.vit.forward_features(images)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.coverage_att = nn.Linear(1, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden, coverage):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att3 = self.coverage_att(coverage.unsqueeze(2))
        att = torch.tanh(att1 + att2 + att3)

        alpha = torch.softmax(self.full_att(att).squeeze(2), dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        coverage = coverage + alpha
        return context, alpha, coverage
   
class AttentionDecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        encoder_dim=768,   # ViT patch dim
        attention_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)

        self.lstm = nn.LSTMCell(
            embed_size + encoder_dim,
            hidden_size
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out, captions, lengths, teacher_forcing_ratio):
        """
        encoder_out: [B, N, encoder_dim]  (ViT patch embeddings)
        captions:    [B, T]
        """
        batch_size = encoder_out.size(0)
        max_len = captions.size(1)

        embeddings = self.embed(captions)
        h = torch.zeros(batch_size, self.lstm.hidden_size).to(encoder_out.device)
        c = torch.zeros_like(h)

        outputs = []

        coverage = torch.zeros(
            encoder_out.size(0),
            encoder_out.size(1),
            device=encoder_out.device
        )

        for t in range(max_len - 1):
            context, alpha, coverage = self.attention(encoder_out, h, coverage)

            if t == 0:
                word_emb = embeddings[:, t]
            else:
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher:
                    word_emb = embeddings[:, t]
                else:
                    word_emb = self.embed(preds.argmax(1))

            lstm_input = torch.cat((word_emb, context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))
            outputs.append(preds)

        return torch.stack(outputs, dim=1)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, embed_size)

    def forward(self, captions, lengths):
        emb = self.embed(captions)  # [B, T, E]

        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)     # h: [1, B, hidden]
        sent = self.proj(h[-1])           # [B, embed_size]

        return torch.nn.functional.normalize(sent, dim=1)