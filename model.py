import torch
import torch.nn as nn


class TransformerConfig:
    def __init__(
        self,
        output_size: int = 1,
        max_len: int = 1001,
        num_layers: int = 2,
        hidden_dim: int = 16,
        mlp_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        deterministic: bool = False,
        decode: bool = False,
        causal_x: bool = True,
        physics_decoder: bool = False,
    ):
        self.output_size = output_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic
        self.decode = decode
        self.causal_x = causal_x
        self.physics_decoder = physics_decoder


class AddPositionEmbs(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_len, config.hidden_dim) * 0.02
        )

    def forward(self, inputs):
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but it is: {inputs.ndim}"
        return inputs + self.pos_embedding[:, : inputs.shape[1], :]


class MlpBlock(nn.Module):
    def __init__(self, config: TransformerConfig, out_dim: int = None):
        super().__init__()
        self.config = config
        self.out_dim = out_dim if out_dim is not None else config.hidden_dim
        self.dense1 = nn.Linear(config.hidden_dim, config.mlp_dim)
        self.dense2 = nn.Linear(config.mlp_dim, self.out_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.dropout(x)


class EncoderDecoder1DBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.attention_dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MlpBlock(config)

    def forward(self, inputs, decoder_mask=None):
        x = self.ln1(inputs)
        x, _ = self.attention(x, x, x, attn_mask=decoder_mask)
        x = self.dropout(x)
        x = x + inputs
        z = self.ln2(x)
        z = self.mlp(z)
        return x + z


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_x = nn.Linear(1, config.hidden_dim // 2)
        self.embed_dx = nn.Linear(1, config.hidden_dim // 2)
        self.pos_embed = AddPositionEmbs(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = nn.ModuleList(
            [EncoderDecoder1DBlock(config) for _ in range(config.num_layers)]
        )
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.logits_x = nn.Linear(config.hidden_dim, config.output_size)
        self.logits_dx = nn.Linear(config.hidden_dim, config.output_size)

    def forward(self, inputs, decoder_mask=None):
        x = inputs["x"]
        dx = build_deltas(x)

        x = self.embed_x(x)
        dx = self.embed_dx(dx)
        x = torch.cat([x, dx], dim=-1)

        x = self.dropout(x)
        x = self.pos_embed(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, decoder_mask=decoder_mask)

        x = self.ln(x)
        logits_x = self.logits_x(x)
        logits_dx = self.logits_dx(x)

        return logits_x, logits_dx, None


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.decoder = Decoder(config)

    def forward(self, inputs):
        cfg = self.config
        decoder_mask = None
        if cfg.causal_x:
            decoder_mask = nn.Transformer.generate_square_subsequent_mask(
                inputs["x"].shape[1]
            ).to(inputs["x"].device)

        logits_x, logits_dx, aux = self.decoder(inputs, decoder_mask=decoder_mask)
        return logits_x, logits_dx, aux


def build_deltas(x):
    dx = x[:, 1:, :] - x[:, :-1, :]
    dx = torch.nn.functional.pad(dx, (0, 0, 1, 0), mode='constant', value=0)
    return dx
