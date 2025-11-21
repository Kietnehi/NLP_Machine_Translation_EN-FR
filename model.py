# model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, DEVICE


# ========== 1. Encoder với LSTM ==========

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, src, src_lengths):
        """
        src: [B, src_len]
        src_lengths: [B] (đã sort giảm dần)
        Trả về:
            outputs: [B, src_len, hid_dim]
            (hidden, cell): context vector
        """
        embedded = self.dropout(self.embedding(src))  # [B, src_len, emb_dim]

        packed_embedded = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True
        )  # [B, src_len, hid_dim]

        # context vector = (hidden, cell)
        return outputs, hidden, cell


# ========== 2. Decoder với LSTM ==========

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        """
        input: [B]  - token hiện tại
        hidden, cell: từ encoder hoặc bước trước
        Trả về:
            prediction: [B, output_dim]
            hidden, cell: trạng thái mới
        """
        input = input.unsqueeze(1)  # [B, 1]
        embedded = self.dropout(self.embedding(input))  # [B, 1, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [B, 1, hid_dim]
        prediction = self.fc_out(output.squeeze(1))  # [B, output_dim]

        return prediction, hidden, cell


# ========== 3. Seq2Seq bao Encoder + Decoder ==========

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=DEVICE):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden size Encoder và Decoder phải giống nhau!"
        assert (
            encoder.num_layers == decoder.num_layers
        ), "Số layer LSTM của Encoder và Decoder phải giống nhau!"

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        src: [B, src_len]
        src_lengths: [B]
        trg: [B, trg_len]
        Trả về:
            outputs: [B, trg_len, output_dim]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(
            self.device
        )

        # Encoder
        _, hidden, cell = self.encoder(src, src_lengths)

        # token đầu tiên của Decoder là <sos>
        input = trg[:, 0]  # [B]

        for t in range(1, trg_len):
            # 1 step Decoder
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # lấy token có xác suất lớn nhất
            top1 = output.argmax(1)  # [B]

            # teacher forcing hay không?
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if use_teacher else top1

        return outputs
