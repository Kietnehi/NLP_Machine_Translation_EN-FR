# train_subword_bpe.py
"""
Train SentencePiece BPE model và re-train Seq2Seq với BiLSTM + Attention
"""
import torchtext
torchtext.disable_torchtext_deprecation_warning()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from tqdm import tqdm

# Import config
from config import (
    DEVICE, DATA_ROOT, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
    SPECIAL_TOKENS, MAX_LEN_GENERATE
)

# BPE-specific hyperparameters
EMB_DIM = 512
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.4
MAX_VOCAB_SIZE = 10_000

# Paths
SPM_EN_MODEL = "spm_en_bpe.model"
SPM_FR_MODEL = "spm_fr_bpe.model"


# ========== 1. Train SentencePiece BPE ==========
def train_sentencepiece_bpe():
    """Train SentencePiece BPE models for EN and FR"""
    
    if os.path.exists(SPM_EN_MODEL) and os.path.exists(SPM_FR_MODEL):
        print("✓ SentencePiece models already exist")
        return
    
    print("🚀 Training SentencePiece BPE for EN...")
    spm.SentencePieceTrainer.train(
        input=os.path.join(DATA_ROOT, "train.en"),
        model_prefix="spm_en_bpe",
        vocab_size=MAX_VOCAB_SIZE - len(SPECIAL_TOKENS),
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0
    )
    
    print("🚀 Training SentencePiece BPE for FR...")
    spm.SentencePieceTrainer.train(
        input=os.path.join(DATA_ROOT, "train.fr"),
        model_prefix="spm_fr_bpe",
        vocab_size=MAX_VOCAB_SIZE - len(SPECIAL_TOKENS),
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0
    )
    
    print("✅ SentencePiece BPE training completed")


# ========== 2. Load SentencePiece Processors ==========
def load_spm_tokenizers():
    """Load trained SentencePiece models"""
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(SPM_EN_MODEL)
    
    sp_fr = spm.SentencePieceProcessor()
    sp_fr.load(SPM_FR_MODEL)
    
    def en_tokenizer(text):
        return sp_en.encode(text.lower(), out_type=str)
    
    def fr_tokenizer(text):
        return sp_fr.encode(text.lower(), out_type=str)
    
    return en_tokenizer, fr_tokenizer


# ========== 3. Model Classes (BiLSTM + Attention) ==========
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        assert hid_dim % 2 == 0
        self.rnn = nn.LSTM(
            emb_dim, hid_dim // 2,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
    
    def _cat_directions(self, h):
        h = h.view(self.num_layers, 2, -1, self.hid_dim // 2)
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=2)
        return h
    
    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        hidden = self._cat_directions(hidden)
        cell = self._cat_directions(cell)
        return outputs, hidden, cell


class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        scores = torch.bmm(output, encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        concat = torch.cat((output, context), dim=2)
        prediction = self.fc_out(concat.squeeze(1))
        return prediction, hidden, cell


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        import random
        batch_size, trg_len = src.size(0), trg.size(1)
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs


print("✅ Script loaded successfully. Ready to train SentencePiece BPE model.")
print(f"Device: {DEVICE}")
print(f"Data root: {DATA_ROOT}")
print("\nTo train, run:")
print("  train_sentencepiece_bpe()")
