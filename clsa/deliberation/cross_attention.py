"""Gated cross-attention for inter-module communication during deliberation.

Section 4.2: During deliberation, modules observe each other via gated
cross-attention. The gating mechanism is learned and controls how much
each module attends to each other module at each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttention(nn.Module):
    """Cross-attention with a learned gate controlling information flow.

    One module (the "receiver") attends to another module's (the "sender")
    hidden states. A sigmoid gate controls how much of the attended
    information is mixed into the receiver's representation.

    This lets the network learn which inter-module interactions are
    useful and when, rather than hard-coding fixed communication patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Query comes from the receiver module
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Key and value come from the sender module
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learned gate: sigmoid output in [0, 1] controlling how much
        # cross-attended information gets mixed in
        self.gate = nn.Linear(hidden_size * 2, 1, bias=True)
        # Initialize gate bias to a small negative value so modules
        # start mostly independent and learn to communicate
        nn.init.constant_(self.gate.bias, -2.0)

        self.dropout = dropout

    def forward(
        self,
        receiver_states: torch.Tensor,
        sender_states: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend from receiver to sender with gating.

        Args:
            receiver_states: (batch, seq_len, hidden_size) from the receiving module.
            sender_states: (batch, seq_len, hidden_size) from the sending module.

        Returns:
            Updated receiver states of the same shape.
        """
        batch, seq_len, hidden_size = receiver_states.shape

        q = self.q_proj(receiver_states)
        k = self.k_proj(sender_states)
        v = self.v_proj(sender_states)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention (no causal mask needed here
        # since this is cross-attention between module states at the same
        # sequence positions)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        attended = self.o_proj(attn)

        # Compute gate value from concatenation of receiver and attended states
        gate_input = torch.cat([receiver_states, attended], dim=-1)
        gate_value = torch.sigmoid(self.gate(gate_input))

        # Mix: output = receiver + gate * attended
        return receiver_states + gate_value * attended
