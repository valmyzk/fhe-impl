from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from torch import nn
from brevitas import nn as qnn


class GlucoseMLP(nn.Module):
    """Tiny 3-layer MLP for glucose prediction."""

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class QGlucoseMLP(nn.Module):
    """Tiny 3-layer MLP, quantized for FHE compilation."""

    def __init__(self, in_features: int, hidden: int, q_bits: int):
        super().__init__()
        self.net = nn.Sequential(
            qnn.QuantIdentity(
                bit_width=q_bits,
                act_quant=Int8ActPerTensorFloat,
                return_quant_tensor=True,
            ),
            qnn.QuantLinear(
                in_features,
                hidden,
                bias=True,
                weight_bit_width=q_bits,
                weight_quant=Int8WeightPerTensorFloat,
            ),
            qnn.QuantReLU(bit_width=q_bits, return_quant_tensor=True),
            qnn.QuantLinear(
                hidden,
                hidden,
                bias=True,
                weight_bit_width=q_bits,
                weight_quant=Int8WeightPerTensorFloat,
            ),
            qnn.QuantReLU(bit_width=q_bits, return_quant_tensor=True),
            qnn.QuantLinear(
                hidden,
                1,
                bias=True,
                weight_bit_width=q_bits,
                weight_quant=Int8WeightPerTensorFloat,
            ),
        )

    def forward(self, x):
        return self.net(x)


__all__ = ["GlucoseMLP", "QGlucoseMLP"]
