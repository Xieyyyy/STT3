import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Modules import ScaledDotProductAttention


class GrapgConv(nn.Module):
    def __init__(self, d_in, d_out, bias):
        super(GrapgConv, self).__init__()
        self.w = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x, adj_mx):
        x = torch.einsum("bnld,nm->bmld", (x, adj_mx))
        return self.w(x)


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.graphconv_q = GrapgConv(d_model, d_model, bias=False)
        self.graphconv_k = GrapgConv(d_model, d_model, bias=False)
        self.graphconv_v = GrapgConv(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.graphconv_q.w.weight)
        nn.init.xavier_uniform_(self.graphconv_k.w.weight)
        nn.init.xavier_uniform_(self.graphconv_v.w.weight)

        self.graphconv_fc = GrapgConv(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.graphconv_fc.w.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, adj_mx, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, sz_node, len_q, len_k, len_v = q.size(0), q.size(1), q.size(2), k.size(2), v.size(2)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.graphconv_q(q, adj_mx).view(sz_b, -1, len_q, n_head, d_k)
        k = self.graphconv_k(k, adj_mx).view(sz_b, -1, len_k, n_head, d_k)
        v = self.graphconv_v(v, adj_mx).view(sz_b, -1, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(2, 3).contiguous().view(sz_b, sz_node, len_q, -1)
        output = self.dropout(self.graphconv_fc(output, adj_mx))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
