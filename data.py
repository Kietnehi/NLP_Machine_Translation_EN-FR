# data.py
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from config import (
    DATA_ROOT,
    MAX_VOCAB_SIZE,
    SPECIAL_TOKENS,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    BATCH_SIZE,
    DEVICE,
)

# LƯU Ý:
# - File này lo phần: đọc text, tokenize, build vocab, Dataset, DataLoader.

# Nếu muốn tắt warning deprecation của torchtext, có thể dùng:
# import torchtext
# torchtext.disable_torchtext_deprecation_warning()


# ========== 1. Tokenizer với spaCy ==========
# Cần cài:
#   pip install spacy torch torchtext
#   python -m spacy download en_core_web_sm
#   python -m spacy download fr_core_news_sm

en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")


# ========== 2. Đọc dữ liệu song song en-fr ==========

def read_parallel_file(
    root: str, split: str
) -> List[Tuple[str, str]]:
    """
    Đọc song song: split.en và split.fr
    Ví dụ:
        split = "train" -> train.en, train.fr
    Trả về list các cặp (en_sentence, fr_sentence)
    """
    en_path = os.path.join(root, f"{split}.en")
    fr_path = os.path.join(root, f"{split}.fr")

    assert os.path.exists(en_path), f"Không tìm thấy file: {en_path}"
    assert os.path.exists(fr_path), f"Không tìm thấy file: {fr_path}"

    pairs = []
    with open(en_path, "r", encoding="utf-8") as f_en, open(
        fr_path, "r", encoding="utf-8"
    ) as f_fr:
        for en_line, fr_line in zip(f_en, f_fr):
            en_line = en_line.strip()
            fr_line = fr_line.strip()
            if en_line == "" or fr_line == "":
                continue
            pairs.append((en_line, fr_line))
    return pairs


# ========== 3. Hàm sinh token để build vocab ==========

def yield_tokens_en(pairs):
    """Iterator trả về token tiếng Anh để build vocab."""
    for en, _ in pairs:
        yield en_tokenizer(en)


def yield_tokens_fr(pairs):
    """Iterator trả về token tiếng Pháp để build vocab."""
    for _, fr in pairs:
        yield fr_tokenizer(fr)


# ========== 4. Xây vocab ==========

def build_vocab(pairs):
    """
    Build vocab cho tiếng Anh & Pháp từ dữ liệu train.
    Giới hạn: MAX_VOCAB_SIZE, thêm SPECIAL_TOKENS.
    YÊU CẦU: UNK_TOKEN phải nằm trong SPECIAL_TOKENS.
    """
    from collections import Counter

    def build_with_limit(yield_tokens_fn):
        # 1) Đếm tần suất token
        counter = Counter()
        for tokens in yield_tokens_fn(pairs):
            counter.update(tokens)

        # 2) Lấy các token phổ biến nhất
        most_common = [w for w, _ in counter.most_common(MAX_VOCAB_SIZE)]

        # 3) Iterator dùng cho build_vocab_from_iterator
        def iterator():
            # build_vocab_from_iterator expects an iterator of token lists
            yield most_common

        # 4) Build vocab, thêm SPECIAL_TOKENS vào đầu
        vocab = build_vocab_from_iterator(
            iterator(),
            specials=SPECIAL_TOKENS,      # ví dụ: ["<pad>", "<sos>", "<eos>", "<unk>"]
            special_first=True,
        )

        # 5) Thiết lập index mặc định là UNK
        vocab.set_default_index(vocab[UNK_TOKEN])

        return vocab

    src_vocab = build_with_limit(yield_tokens_en)
    trg_vocab = build_with_limit(yield_tokens_fr)
    return src_vocab, trg_vocab


# ========== 5. Dataset ==========

class TranslationDataset(Dataset):
    """
    Dataset lưu các câu đã tokenize + numericalize.
    Mỗi phần tử:
        src_tensor: [src_len]
        trg_tensor: [trg_len]
    """

    def __init__(self, pairs, src_vocab, trg_vocab):
        self.data = []
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        for en, fr in pairs:
            src_tokens = [SOS_TOKEN] + list(en_tokenizer(en)) + [EOS_TOKEN]
            trg_tokens = [SOS_TOKEN] + list(fr_tokenizer(fr)) + [EOS_TOKEN]

            src_ids = torch.tensor(
                [src_vocab[token] for token in src_tokens],
                dtype=torch.long,
            )
            trg_ids = torch.tensor(
                [trg_vocab[token] for token in trg_tokens],
                dtype=torch.long,
            )

            self.data.append((src_ids, trg_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ========== 6. collate_fn cho DataLoader ==========

def collate_fn(batch, pad_idx: int):
    """
    batch: list[(src_ids, trg_ids)]
    Trả về:
        src_padded: [B, src_len]
        src_lengths: [B]
        trg_padded: [B, trg_len]
        trg_lengths: [B]
    Và đã sort theo src_lengths giảm dần để dùng pack_padded_sequence.
    """
    src_list, trg_list = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    trg_lengths = torch.tensor([len(t) for t in trg_list], dtype=torch.long)

    # sort theo src_len giảm dần
    src_lengths, sort_idx = src_lengths.sort(descending=True)
    src_list = [src_list[i] for i in sort_idx]
    trg_list = [trg_list[i] for i in sort_idx]
    trg_lengths = torch.tensor([len(t) for t in trg_list], dtype=torch.long)

    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    trg_padded = pad_sequence(trg_list, batch_first=True, padding_value=pad_idx)

    return (
        src_padded.to(DEVICE),
        src_lengths.to(DEVICE),
        trg_padded.to(DEVICE),
        trg_lengths.to(DEVICE),
    )


# ========== 7. Hàm build DataLoader đầy đủ ==========

def build_dataloaders():
    """
    - Đọc train/val/test
    - Build vocab từ train
    - Tạo Dataset + DataLoader
    Trả về:
        train_loader, val_loader, test_loader,
        src_vocab, trg_vocab, pad_idx
    """
    # Đọc dữ liệu
    train_pairs = read_parallel_file(DATA_ROOT, "train")
    val_pairs = read_parallel_file(DATA_ROOT, "val")
    test_pairs = read_parallel_file(DATA_ROOT, "test")

    # Build vocab từ train
    src_vocab, trg_vocab = build_vocab(train_pairs)

    pad_idx = src_vocab[PAD_TOKEN]  # PAD theo src_vocab (trg_vocab cũng có)

    train_dataset = TranslationDataset(train_pairs, src_vocab, trg_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, trg_vocab)
    test_dataset = TranslationDataset(test_pairs, src_vocab, trg_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # BLEU từng câu cho dễ
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )

    return train_loader, val_loader, test_loader, src_vocab, trg_vocab, pad_idx
