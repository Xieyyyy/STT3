import math

import torch
import torch.nn as nn
import transformer.Constants as Constants
from transformer.Layers import EventEncoderLayer, ValueEncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    # [B,L]
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class ValueEncoder(nn.Module):
    def __init__(self, enc_dim, d_model, d_inner,
                 n_layers, n_head, d_k, d_v, dropout, device):
        super(ValueEncoder, self).__init__()
        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        self.value_emb = nn.Linear(enc_dim, d_model)
        self.layer_stack = nn.ModuleList([
            ValueEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, enc_in, event_time):
        tem_enc = self.temporal_enc(event_time)
        enc_output = self.value_emb(enc_in)
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output)
        return enc_output


class EventEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, device):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EventEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # event_type: [B,L]
        # event_time: [B,L]
        # non_pad_mask: [B,L,1]\in{0,1}

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(dim, dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(dim, num_types, bias=True))

    def forward(self, data, non_pad_mask):
        out = self.MLP(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, enc_dim=6, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, device="cpu"):
        super().__init__()

        self.event_encoder = EventEncoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device
        )

        self.value_encoder = ValueEncoder(
            enc_dim=enc_dim,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device
        )

        self.combine_encoder = ValueEncoder(
            enc_dim=d_model * 2,
            d_model=d_model * 2,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model * 2, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        # self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model * 2, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model * 2, num_types)

    def forward(self, event_type, event_time, weather_info):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        # event_type: [B,L]
        non_pad_mask = get_non_pad_mask(event_type)

        # non_pad_mask: [B,L,1]\in{0,1}
        # event_time: [B,L]
        event_enc_output = self.event_encoder(event_type, event_time, non_pad_mask)
        weather_enc_output = self.value_encoder(weather_info, event_time)

        combine_enc_output = self.combine_encoder(torch.cat([event_enc_output, weather_enc_output], dim=-1), event_time)
        # enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(combine_enc_output, non_pad_mask)

        type_prediction = self.type_predictor(combine_enc_output, non_pad_mask)

        return combine_enc_output, (type_prediction, time_prediction)
