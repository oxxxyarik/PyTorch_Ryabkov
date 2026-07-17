import math
import torch
import torch.nn as nn
from transformer_basics.layers import Embedding, MultiheadAttention, FeedForward, DecoderLayer, Decoder

class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_len=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = Embedding(d_model=d_model, vocab_len=vocab_size, pad_index=0)
        
        mha = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        dec_layer = DecoderLayer(mha=mha, enc_dec_mha=mha, ffn=ffn, dropout=dropout)
        
        self.decoder = Decoder(dec_layer=dec_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.tril(torch.ones(sz, sz, device=device))
        return mask.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        
        out = self.embedding(x)

        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        zero_memory = torch.zeros_like(out)
        
        out = self.decoder(out, encoder_memory=zero_memory, src_mask=None, tgt_mask=tgt_mask)
        
        logits = self.fc_out(out)
        return logits