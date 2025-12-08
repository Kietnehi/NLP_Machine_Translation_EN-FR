"""
So sánh 2 models theo đúng cách Colab notebook
- Model 1 (word-level): sử dụng spaCy tokenizer + raw text files
- Model 2 (subword BPE): sử dụng SentencePiece + DataLoader giống Colab
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm.auto import tqdm

# Import từ các files có sẵn
from config import *
from model import Encoder, Decoder
from compare_two_models import (
    BiLSTMEncoderForSubword, 
    AttnDecoder, 
    Seq2SeqAttn,
    load_sentencepiece_tokenizer,
    bpe_decode,
    translate_from_ids
)

# Tokenizers cho word-level
try:
    import spacy
    en_nlp = spacy.load("en_core_web_sm")
    fr_nlp = spacy.load("fr_core_news_sm")
    
    def word_en_tokenizer(text):
        return [token.text for token in en_nlp.tokenizer(text.lower())]
    
    def word_fr_tokenizer(text):
        return [token.text for token in fr_nlp.tokenizer(text.lower())]
    
    HAS_SPACY = True
except:
    HAS_SPACY = False
    print("spaCy không khả dụng")

# SentencePiece
try:
    import sentencepiece as spm
    HAS_SPM = True
except:
    HAS_SPM = False
    print("SentencePiece không khả dụng")

# BLEU
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ============================================================
# DATASET VÀ DATALOADER CHO SUBWORD BPE (theo Colab notebook)
# ============================================================

class TranslationDataset(Dataset):
    """Dataset cho subword BPE giống Colab"""
    def __init__(self, pairs, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
        self.data = []
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        
        for en, fr in pairs:
            src_tokens = [SOS_TOKEN] + src_tokenizer(en) + [EOS_TOKEN]
            trg_tokens = [SOS_TOKEN] + trg_tokenizer(fr) + [EOS_TOKEN]
            
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


def collate_fn(batch, pad_idx):
    """
    batch: list[(src_ids, trg_ids)]
    Trả về:
        src_padded:  [B, src_len]
        src_lengths: [B]
        trg_padded:  [B, trg_len]
        trg_lengths: [B]
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


def read_parallel_file(root, split):
    """
    Đọc song song split.en và split.fr (raw text)
    Trả về list (en_sentence, fr_sentence)
    """
    en_path = os.path.join(root, f"{split}.en")
    fr_path = os.path.join(root, f"{split}.fr")
    
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


# ============================================================
# TÍNH BLEU CHO SUBWORD MODEL (theo Colab notebook)
# ============================================================

def beam_search_decode(model, encoder_outputs, hidden, cell, trg_vocab, beam_size=5, max_len=50):
    """
    Beam search decoding cho Seq2Seq + Attention
    Giống y chang Suggestion.ipynb
    """
    EOS = trg_vocab[EOS_TOKEN]
    
    # mỗi phần tử beam = (score, seq, hidden, cell)
    beams = [(0.0, [trg_vocab[SOS_TOKEN]], hidden, cell)]  # score = log prob
    
    finished = []
    
    for _ in range(max_len):
        new_beams = []
        
        for score, seq, h, c in beams:
            last_token = seq[-1]
            
            if last_token == EOS:
                finished.append((score, seq))
                continue
            
            input_token = torch.tensor([last_token], dtype=torch.long, device=DEVICE)
            
            with torch.no_grad():
                output, new_h, new_c, _ = model.decoder(
                    input_token, h, c, encoder_outputs
                )
            
            # log-softmax để cộng log-prob
            log_probs = torch.log_softmax(output, dim=1)  # [1, vocab]
            
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=1)
            
            for k in range(beam_size):
                candidate_score = score + topk_log_probs[0][k].item()
                candidate_seq = seq + [topk_ids[0][k].item()]
                new_beams.append((candidate_score, candidate_seq, new_h, new_c))
        
        # giữ lại beam tốt nhất
        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        beams = new_beams
        
        if len(finished) >= beam_size:
            break
    
    if len(finished) == 0:
        finished = beams
    
    # chọn câu có điểm cao nhất
    best = sorted(finished, key=lambda x: x[0], reverse=True)[0]
    best_seq = best[1]
    return best_seq


def translate_from_ids_beam(src_ids, model, src_vocab, trg_vocab, beam_size=5, max_len=50):
    """
    Dịch từ dãy ID src với beam search
    Giống y chang Suggestion.ipynb
    """
    model.eval()
    
    # Chuyển về tensor
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    # Beam search
    best_seq = beam_search_decode(
        model, encoder_outputs, hidden, cell, trg_vocab,
        beam_size=beam_size,
        max_len=max_len
    )
    
    # Chuyển ID -> token
    trg_tokens = [trg_vocab.lookup_token(idx) for idx in best_seq]
    
    # Bỏ SOS, cắt tại EOS
    trg_tokens = trg_tokens[1:]
    if EOS_TOKEN in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index(EOS_TOKEN)]
    
    return trg_tokens  # trả về list token BPE


def compute_bleu_subword(model, test_loader, src_vocab, trg_vocab, beam_size=5):
    """
    Tính BLEU score cho subword model với beam search
    Giống y chang Subword(BPE).ipynb - tính trên BPE tokens, KHÔNG decode
    """
    smoothie = SmoothingFunction().method4
    model.eval()
    scores = []
    
    with torch.no_grad():
        for src, src_len, trg, trg_len in tqdm(test_loader, desc="BLEU (Subword)", total=len(test_loader)):
            batch_size = src.size(0)
            
            for b in range(batch_size):
                # ===== Reference: BPE tokens =====
                trg_ids = trg[b].tolist()
                trg_tokens = [trg_vocab.lookup_token(idx) for idx in trg_ids]
                clean_trg = [
                    tok for tok in trg_tokens
                    if tok not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
                ]
                reference = [clean_trg]  # List of BPE tokens
                
                # ===== Source: lấy đúng length thật, không dùng PAD =====
                true_len = src_len[b].item()
                src_ids = src[b, :true_len].tolist()
                
                # ===== Hypothesis: BPE tokens =====
                hypothesis_tokens = translate_from_ids_beam(
                    src_ids, model, src_vocab, trg_vocab, beam_size=beam_size
                )  # Trả về list BPE tokens
                
                # Tính BLEU trên BPE tokens (không decode)
                bleu = sentence_bleu(reference, hypothesis_tokens, smoothing_function=smoothie)
                scores.append(bleu)
    
    return sum(scores) / len(scores) * 100 if scores else 0.0


# ============================================================
# TÍNH BLEU CHO WORD-LEVEL MODEL (dùng raw text)
# ============================================================

def compute_bleu_wordlevel(model, test_en, test_fr, src_vocab, trg_vocab, en_tokenizer, fr_tokenizer):
    """
    Tính BLEU score cho word-level model
    Sử dụng raw text files và spaCy tokenizer
    """
    from compare_two_models import translate_sentence
    
    model.eval()
    
    references = []
    predictions = []
    
    for src_sentence, trg_sentence in tqdm(zip(test_en, test_fr), total=len(test_en), desc="BLEU (Word-level)"):
        # Dịch câu
        pred_tokens = translate_sentence(src_sentence, model, src_vocab, trg_vocab, en_tokenizer, is_attn_model=False)
        
        # Chuẩn bị reference (tokenize câu target)
        ref_tokens = list(fr_tokenizer(trg_sentence.lower()))
        
        predictions.append(pred_tokens)
        references.append([ref_tokens])
    
    # Tính BLEU score
    bleu = bleu_score(predictions, references)
    return bleu * 100


# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def compare_models():
    """
    So sánh 2 models:
    - Model 1 (word-level): best_model.pth
    - Model 2 (subword BPE): best_model_subword.pth
    """
    print("="*80)
    print("SO SÁNH WORD-LEVEL VS SUBWORD (BPE) MODELS")
    print("="*80)
    
    # Paths
    model1_path = "best_model.pth"
    model2_path = "best_model_subword.pth"
    spm_en_model = "spm_en_bpe.model"
    spm_fr_model = "spm_fr_bpe.model"
    data_root = "./data/multi30k_en_fr"
    
    print(f"SentencePiece EN model: {spm_en_model}")
    print(f"SentencePiece FR model: {spm_fr_model}")
    print()
    
    # ========== LOAD TEST DATA (RAW TEXT) ==========
    test_en_path = os.path.join(data_root, "test.en")
    test_fr_path = os.path.join(data_root, "test.fr")
    
    with open(test_en_path, "r", encoding="utf-8") as f:
        test_en = [line.strip() for line in f]
    
    with open(test_fr_path, "r", encoding="utf-8") as f:
        test_fr = [line.strip() for line in f]
    
    print(f"Test set size: {len(test_en)} sentences")
    print()
    
    # ========== MODEL 1: WORD-LEVEL ==========
    print(f"[1/2] Loading Word-level model from {model1_path}...")
    ckpt1 = torch.load(model1_path, map_location=DEVICE)
    src_vocab1 = ckpt1["src_vocab"]
    trg_vocab1 = ckpt1["trg_vocab"]
    pad_idx1 = ckpt1["pad_idx"]
    
    # Load model architecture
    encoder1 = Encoder(
        input_dim=len(src_vocab1),
        emb_dim=256,
        hid_dim=512,
        num_layers=2,
        dropout=0.3
    )
    decoder1 = Decoder(
        output_dim=len(trg_vocab1),
        emb_dim=256,
        hid_dim=512,
        num_layers=2,
        dropout=0.3
    )
    
    from model import Seq2Seq
    model1 = Seq2Seq(encoder1, decoder1, DEVICE).to(DEVICE)
    model1.load_state_dict(ckpt1["model_state_dict"])
    model1.eval()
    
    print(f"   [OK] Word-level model loaded")
    print(f"   Vocab size: EN={len(src_vocab1)}, FR={len(trg_vocab1)}")
    print()
    
    # Calculate BLEU for Model 1
    print(f"   Calculating BLEU for Word-level model...")
    bleu1 = compute_bleu_wordlevel(
        model1, test_en, test_fr, 
        src_vocab1, trg_vocab1,
        word_en_tokenizer, word_fr_tokenizer
    )
    print(f"   [OK] Word-level BLEU: {bleu1:.2f}%")
    print()
    
    # ========== MODEL 2: SUBWORD BPE ==========
    print(f"[2/2] Loading Subword BPE model from {model2_path}...")
    ckpt2 = torch.load(model2_path, map_location=DEVICE)
    src_vocab2 = ckpt2["src_vocab"]
    trg_vocab2 = ckpt2["trg_vocab"]
    pad_idx2 = ckpt2["pad_idx"]
    
    # Load SentencePiece tokenizers
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(spm_en_model)
    
    sp_trg = spm.SentencePieceProcessor()
    sp_trg.load(spm_fr_model)
    
    def src_bpe_tokenizer(text):
        return sp_src.encode(text, out_type=str)
    
    def trg_bpe_tokenizer(text):
        return sp_trg.encode(text, out_type=str)
    
    # Build DataLoader
    test_pairs = read_parallel_file(data_root, "test")
    test_dataset = TranslationDataset(test_pairs, src_vocab2, trg_vocab2, src_bpe_tokenizer, trg_bpe_tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx2),
    )
    
    # Load model architecture
    encoder2 = BiLSTMEncoderForSubword(
        input_dim=len(src_vocab2),
        emb_dim=512,
        hid_dim=512,
        num_layers=2,
        dropout=0.4
    )
    decoder2 = AttnDecoder(
        output_dim=len(trg_vocab2),
        emb_dim=512,
        hid_dim=512,
        num_layers=2,
        dropout=0.4
    )
    
    model2 = Seq2SeqAttn(encoder2, decoder2, device=DEVICE).to(DEVICE)
    model2.load_state_dict(ckpt2["model_state_dict"])
    model2.eval()
    
    print(f"   [OK] Subword BPE model loaded")
    print(f"   Vocab size: EN={len(src_vocab2)}, FR={len(trg_vocab2)}")
    print()
    
    # Calculate BLEU for Model 2
    print(f"   Calculating BLEU for Subword BPE model with beam search (beam_size=5)...")
    bleu2 = compute_bleu_subword(model2, test_loader, src_vocab2, trg_vocab2, beam_size=5)
    print(f"   [OK] Subword BPE BLEU: {bleu2:.2f}%")
    print()
    
    # ========== COMPARISON SUMMARY ==========
    print("="*80)
    print("KẾT QUẢ SO SÁNH")
    print("="*80)
    print(f"Word-level model       : BLEU = {bleu1:6.2f}%")
    print(f"Subword BPE model      : BLEU = {bleu2:6.2f}%")
    print("-"*80)
    
    if bleu2 > bleu1:
        diff = bleu2 - bleu1
        print(f"🏆 Subword BPE model tốt hơn Word-level với chênh lệch {diff:.2f} BLEU points")
    elif bleu1 > bleu2:
        diff = bleu1 - bleu2
        print(f"🏆 Word-level model tốt hơn Subword BPE với chênh lệch {diff:.2f} BLEU points")
    else:
        print(f"[TIED] Cả hai models có hiệu suất tương đương")
    print("="*80)
    print()
    
    print(f"[OK] Hoàn thành so sánh!")
    print()
    print(f"Tóm tắt:")
    print(f"   - Word-level model: {bleu1:.2f}% BLEU")
    print(f"   - Subword BPE model: {bleu2:.2f}% BLEU")
    print(f"   - Test set: {len(test_en)} câu")


if __name__ == "__main__":
    compare_models()
