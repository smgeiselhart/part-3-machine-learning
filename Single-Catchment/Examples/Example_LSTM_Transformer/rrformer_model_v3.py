import torch
import torch.nn as nn


class RRFormer(nn.Module):
    """
    RRFormer: predicts runoff exclusively from rainfall inputs.

    The decoder holds pred_len learnable query embeddings that attend to the
    encoder's rainfall representation via cross-attention.  This makes the
    mapping purely rainfall → runoff with no dependence on flow state.

    Architecture:
        Encoder : window_len days of rainfall features → contextual representation
        Decoder : pred_len learnable query vectors (no data input)
        Output  : pred_len runoff predictions

    Uses nn.Transformer directly for a compact, efficient implementation.

    Args:
        enc_input_size  : encoder input features (e.g. 3 for rain, rain_30d, rain_90d)
        d_model         : embedding dimension
        nhead           : attention heads (must divide d_model)
        num_enc_layers  : encoder depth
        num_dec_layers  : decoder depth
        dim_feedforward : FF sub-network width
        window_len      : encoder sequence length (days)
        pred_len        : number of output timesteps
        dropout         : dropout rate
    """

    def __init__(self, enc_input_size, d_model, nhead,
                 num_enc_layers, num_dec_layers, dim_feedforward,
                 window_len, pred_len, dropout=0.1):
        super().__init__()
        self.model_name = "RR-Former"
        self.window_len = window_len
        self.pred_len   = pred_len

        # Project rainfall features into the model's embedding space
        self.enc_proj = nn.Linear(enc_input_size, d_model)

        # Learnable position embeddings for the encoder
        self.enc_pos = nn.Embedding(window_len, d_model)

        # Learnable query embeddings for the decoder — one per output timestep.
        # These replace any data-driven decoder input: the decoder learns what
        # to ask the encoder, purely from training, with no flow signal.
        self.dec_queries = nn.Embedding(pred_len, d_model)

        self.drop = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        """
        Args:
            src : (batch, window_len, enc_input_size) — rainfall features
        Returns:
            predictions : (batch, pred_len)
        """
        device  = src.device
        batch   = src.size(0)

        # Encoder: rainfall features + positional embedding
        enc_pos = torch.arange(self.window_len, device=device)
        src_emb = self.drop(self.enc_proj(src) + self.enc_pos(enc_pos))

        # Decoder: fixed learnable queries, same for every sample in the batch
        dec_idx = torch.arange(self.pred_len, device=device)
        tgt_emb = self.dec_queries(dec_idx)                  # (pred_len, d_model)
        tgt_emb = self.drop(tgt_emb.unsqueeze(0).expand(batch, -1, -1))

        out = self.transformer(src_emb, tgt_emb)             # (batch, pred_len, d_model)
        return self.fc_out(out).squeeze(-1)                  # (batch, pred_len)
