import torch.nn as nn
import torchvision.models as models
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from phase3_flikr import Attention

class AttentionEncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(AttentionEncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained = True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-2]  # delete the last fc layer & Avg Pooling.
        self.resnet = nn.Sequential(*modules)   
        # self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size)
    
    def forward(self, images):
        features = self.resnet(images)   # [B, 2048, 7, 7]
        # print(features.shape)
        features = features.permute(0, 2, 3, 1) # [B, 7, 7, 2048]
        # print(features.shape)
        features = features.view(features.size(0), -1, features.size(-1)) # [B, 49, 2048]
        # print(features.shape)
        # features = self.bn(self.embed(features))
        return features
    
class AttentionDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim, attention_dim, dropout=0.3):
        super(AttentionDecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(attention_dim=attention_dim, decoder_dim=hidden_size, encoder_dim=encoder_dim)
        self.lstm = nn.LSTMCell(
            embed_size+encoder_dim,
            hidden_size
        )

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden_state(self, encoder_out):
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        h = torch.zeros(batch_size, self.lstm.hidden_size).to(device)
        c = torch.zeros(batch_size, self.lstm.hidden_size).to(device)
        return h, c
    
    def forward(self, encoder_out, captions, lengths):
        """
        encoder_out: [B, 49, 2048]
        captions:    [B, max_len]
        lengths:     [B]
        """

        batch_size = encoder_out.size(0)
        max_len = max(lengths) - 1   # remove <end>

        # ---- Teacher forcing: feed caption[:-1] ----
        captions_in = captions[:, :-1]   # remove <end> token

        embeddings = self.embed(captions_in)    # [B, T-1, embed_size]
        h, c = self.init_hidden_state(encoder_out)

        outputs = torch.zeros(batch_size, max_len, self.linear.out_features).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_len):
            context, alpha = self.attention(encoder_out, h)

            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            preds = self.linear(self.dropout(h))

            outputs[:, t] = preds
            alphas[:, t] = alpha

        return outputs, alphas
