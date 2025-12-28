from torch import nn

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: [B, 49, encoder_dim]
        decoder_hidden: [B, decoder_dim]
        """
        att1 = self.encoder_att(encoder_out)              # [B, 49, att_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)

        alpha = self.softmax(att)                         # [B, 49]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha