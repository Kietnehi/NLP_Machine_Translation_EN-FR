# compare_two_models.py
"""
So sánh 2 models:
1. best_model.pth: Word-level tokenizer (spaCy)
2. best_model_subword.pth: Subword BPE tokenizer (SentencePiece)
"""
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torch
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import os

from config import DEVICE, SOS_TOKEN, EOS_TOKEN, MAX_LEN_GENERATE

# Word-level tokenizer (spaCy)
from data import en_tokenizer as word_en_tokenizer, fr_tokenizer as word_fr_tokenizer
from model import Encoder, Decoder, Seq2Seq

# Subword tokenizer (SentencePiece)
try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    print("Warning: sentencepiece not installed. Subword model comparison will fail.")


# ========== Model classes for Subword (BiLSTM + Attention) ==========
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoderForSubword(nn.Module):
    """
    BiLSTM Encoder for subword model (matching Colab Subword(BPE).ipynb)
    - Bidirectional LSTM with hid_dim // 2 per direction
    - Concatenates forward and backward to get hid_dim output
    """
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        assert hid_dim % 2 == 0, "HID_DIM must be even for BiLSTM"
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

    def _cat_directions(self, h):
        """
        h: [num_layers * 2, B, hid_dim // 2]
        Returns: [num_layers, B, hid_dim]
        """
        num_layers_times_2, batch_size, hid_half = h.size()
        assert num_layers_times_2 == self.num_layers * 2

        h = h.view(self.num_layers, 2, batch_size, hid_half)
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=2)
        return h

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        hidden = self._cat_directions(hidden)
        cell = self._cat_directions(cell)

        return outputs, hidden, cell


class AttnDecoder(nn.Module):
    """
    Luong attention decoder for subword model
    """
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

        return prediction, hidden, cell, attn_weights.squeeze(1)


class Seq2SeqAttn(nn.Module):
    """Seq2Seq with attention for subword model"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            top1 = output.argmax(1)

            import random
            if random.random() < teacher_forcing_ratio:
                input = trg[:, t]
            else:
                input = top1

        return outputs



def load_model_from_checkpoint(ckpt_path, is_subword=False):
    """
    Load model + vocab từ checkpoint
    is_subword: True nếu là subword model (BiLSTM + Attention)
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    src_vocab = ckpt["src_vocab"]
    trg_vocab = ckpt["trg_vocab"]
    pad_idx = ckpt["pad_idx"]

    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)

    # Load hyperparameters from checkpoint if available
    if "hyperparameters" in ckpt:
        hp = ckpt["hyperparameters"]
        emb_dim = hp.get("emb_dim", 256)
        hid_dim = hp.get("hid_dim", 512)
        num_layers = hp.get("num_layers", 2)
        dropout = hp.get("dropout", 0.3)
    else:
        # Infer from model architecture
        if is_subword:
            emb_dim = 512
            hid_dim = 512
            num_layers = 2
            dropout = 0.4
        else:
            from config import EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT
            emb_dim = EMB_DIM
            hid_dim = HID_DIM
            num_layers = NUM_LAYERS
            dropout = DROPOUT

    if is_subword:
        # BiLSTM Encoder + Attention Decoder (matching Colab architecture)
        encoder = BiLSTMEncoderForSubword(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        decoder = AttnDecoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = Seq2SeqAttn(encoder, decoder, device=DEVICE).to(DEVICE)
    else:
        # Regular LSTM Encoder + Decoder
        encoder = Encoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        decoder = Decoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = Seq2Seq(encoder, decoder, device=DEVICE).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, src_vocab, trg_vocab, pad_idx


def translate_sentence(sentence, model, src_vocab, trg_vocab, en_tokenizer, is_attn_model=False, max_len=MAX_LEN_GENERATE):
    """
    Dịch một câu tiếng Anh sang tiếng Pháp
    en_tokenizer: function để tokenize câu input
    is_attn_model: True nếu model có attention (cần encoder_outputs)
    """
    model.eval()

    tokens_en = [SOS_TOKEN] + list(en_tokenizer(sentence.lower())) + [EOS_TOKEN]
    src_indexes = [src_vocab[token] for token in tokens_en]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_length = torch.tensor([len(src_indexes)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_length)

    trg_indexes = [trg_vocab[SOS_TOKEN]]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            if is_attn_model:
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            else:
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab[EOS_TOKEN]:
            break

    trg_tokens = [trg_vocab.lookup_token(idx) for idx in trg_indexes]
    trg_tokens = trg_tokens[1:]
    if EOS_TOKEN in trg_tokens:
        trg_tokens = trg_tokens[: trg_tokens.index(EOS_TOKEN)]

    return trg_tokens


def translate_from_ids(src_ids, model, src_vocab, trg_vocab, max_len=MAX_LEN_GENERATE):
    """
    Dịch từ dãy ID src (đã là BPE + SOS/EOS/PAD) -> list token trg (BPE)
    Giống hệt như trong Colab notebook
    src_ids: list[int]
    """
    model.eval()
    
    # Chuyển về tensor
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    # Bắt đầu với SOS
    trg_indexes = [trg_vocab[SOS_TOKEN]]
    
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            # Decoder với attention (trả về 4 values)
            output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == trg_vocab[EOS_TOKEN]:
            break
    
    # Chuyển ID -> token
    trg_tokens = [trg_vocab.lookup_token(idx) for idx in trg_indexes]
    
    # Bỏ SOS, cắt tại EOS
    trg_tokens = trg_tokens[1:]
    if EOS_TOKEN in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index(EOS_TOKEN)]
    
    return trg_tokens  # trả về list token BPE



def load_test_data(test_en_path="./data/multi30k_en_fr/test.en", 
                   test_fr_path="./data/multi30k_en_fr/test.fr"):
    """
    Load test data
    """
    with open(test_en_path, "r", encoding="utf-8") as f:
        test_en = [line.strip() for line in f]
    
    with open(test_fr_path, "r", encoding="utf-8") as f:
        test_fr = [line.strip() for line in f]
    
    return test_en, test_fr


def bpe_decode(tokens):
    """
    Decode BPE tokens to normal text
    tokens: list of BPE subword tokens like ['▁hello', '▁world']
    returns: 'hello world'
    """
    s = "".join(tokens)
    s = s.replace("\u2581", " ").strip()  # \u2581 is Unicode for ▁
    return s


def calculate_bleu_from_loader(model, test_loader, src_vocab, trg_vocab):
    """
    Tính BLEU score cho subword model giống Colab notebook
    Sử dụng DataLoader và compute theo từng batch
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from config import SPECIAL_TOKENS, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    
    smoothie = SmoothingFunction().method4
    model.eval()
    scores = []
    
    with torch.no_grad():
        for src, src_len, trg, trg_len in tqdm(test_loader, desc="Translating", total=len(test_loader)):
            batch_size = src.size(0)
            
            for b in range(batch_size):
                # Lấy reference target (FR REF)
                trg_ids = trg[b].tolist()
                trg_tokens = [trg_vocab.lookup_token(idx) for idx in trg_ids]
                clean_trg = [tok for tok in trg_tokens if tok not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
                
                # Decode BPE to words
                fr_ref = bpe_decode(clean_trg)
                reference = [fr_ref.split()]
                
                # Lấy source thật (không có PAD)
                true_len = src_len[b].item()
                src_ids = src[b, :true_len].tolist()
                
                # Dịch bằng model
                hyp_bpe_tokens = translate_from_ids(src_ids, model, src_vocab, trg_vocab)
                fr_hyp = bpe_decode(hyp_bpe_tokens)
                hypothesis = fr_hyp.split()
                
                # Tính BLEU
                bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
                scores.append(bleu)
    
    return sum(scores) / len(scores) * 100 if scores else 0.0


def calculate_bleu(model, src_vocab, trg_vocab, test_en, test_fr, en_tokenizer, fr_tokenizer, is_attn_model=False, use_bpe_decode=False):
    """
    Tính BLEU score trên tập test cho word-level model
    en_tokenizer: tokenizer cho input EN
    fr_tokenizer: tokenizer cho reference FR
    is_attn_model: True nếu model có attention
    """
    model.eval()
    
    references = []
    predictions = []
    
    for src_sentence, trg_sentence in tqdm(zip(test_en, test_fr), total=len(test_en), desc="Translating"):
        # Dịch câu
        pred_tokens = translate_sentence(src_sentence, model, src_vocab, trg_vocab, en_tokenizer, is_attn_model)
        
        # Chuẩn bị reference (tokenize câu target)
        ref_tokens = list(fr_tokenizer(trg_sentence.lower()))
        
        predictions.append(pred_tokens)
        references.append([ref_tokens])  # BLEU expects list of lists
    
    # Tính BLEU score
    bleu = bleu_score(predictions, references)
    return bleu * 100, predictions, references


def load_sentencepiece_tokenizer(model_path):
    """
    Load SentencePiece tokenizer
    """
    if not HAS_SPM:
        raise ImportError("sentencepiece is required for subword model")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    def tokenizer(text):
        return sp.encode_as_pieces(text.lower())
    
    return tokenizer


def create_vocab_based_tokenizer(vocab):
    """
    Tạo tokenizer đơn giản dựa trên vocab có sẵn
    Sử dụng khi không có SentencePiece model
    """
    # Lấy tất cả tokens trong vocab (trừ special tokens)
    from config import SPECIAL_TOKENS
    all_tokens = []
    for i in range(len(vocab)):
        token = vocab.lookup_token(i)
        if token not in SPECIAL_TOKENS:
            all_tokens.append(token)
    
    # Sort theo độ dài giảm dần để match từ dài trước
    all_tokens_sorted = sorted(all_tokens, key=len, reverse=True)
    
    def tokenizer(text):
        text = text.lower().strip()
        tokens = []
        words = text.split()
        
        for word in words:
            # Thử match với các subword tokens
            remaining = word
            word_tokens = []
            
            while remaining:
                matched = False
                for token in all_tokens_sorted:
                    clean_token = token.replace('▁', '')
                    if remaining.startswith(clean_token):
                        word_tokens.append(token)
                        remaining = remaining[len(clean_token):]
                        matched = True
                        break
                
                if not matched:
                    # Nếu không match được, bỏ ký tự đầu và tiếp tục
                    if remaining:
                        remaining = remaining[1:]
                    break
            
            if word_tokens:
                tokens.extend(word_tokens)
        
        return tokens
    
    return tokenizer



def compare_models(model1_path, model2_path, model1_name, model2_name, 
                   spm_en_model=None, spm_fr_model=None):
    """
    So sanh 2 models tren tap test
    - Model 1 (word-level): su dung spaCy tokenizer
    - Model 2 (subword): su dung SentencePiece tokenizer
    """
    print("="*80)
    print(f"SO SANH HIEU SUAT HAI MODELS TREN TAP TEST")
    print("="*80)
    print()
    
    # Load test data
    print("Loading test data...")
    test_en, test_fr = load_test_data()
    print(f"Test set size: {len(test_en)} sentences")
    print()
    
    # ========== MODEL 1: Word-level ==========
    print(f"Loading {model1_name} from {model1_path}...")
    model1, src_vocab1, trg_vocab1, pad_idx1 = load_model_from_checkpoint(model1_path, is_subword=False)
    print(f"[OK] {model1_name} loaded successfully")
    print(f"  - Source vocab size: {len(src_vocab1)}")
    print(f"  - Target vocab size: {len(trg_vocab1)}")
    print(f"  - Tokenizer: spaCy (word-level)")
    print()
    
    # Calculate BLEU for Model 1 (word-level tokenizer)
    print(f"Calculating BLEU score for {model1_name}...")
    bleu1, preds1, refs1 = calculate_bleu(
        model1, src_vocab1, trg_vocab1, test_en, test_fr,
        en_tokenizer=word_en_tokenizer,
        fr_tokenizer=word_fr_tokenizer,
        is_attn_model=False
    )
    print(f"[OK] {model1_name} BLEU Score: {bleu1:.2f}")
    print()
    
    # ========== MODEL 2: Subword BPE ==========
    print(f"Loading {model2_name} from {model2_path}...")
    model2, src_vocab2, trg_vocab2, pad_idx2 = load_model_from_checkpoint(model2_path, is_subword=True)
    print(f"[OK] {model2_name} loaded successfully")
    print(f"  - Source vocab size: {len(src_vocab2)}")
    print(f"  - Target vocab size: {len(trg_vocab2)}")
    print(f"  - Tokenizer: SentencePiece BPE (subword)")
    print()
    
    # Load SentencePiece tokenizers
    if spm_en_model is None or spm_fr_model is None:
        print("[WARNING]  Warning: SentencePiece model paths not provided. Trying default paths...")
        spm_en_model = "spm_en_bpe.model"
        spm_fr_model = "spm_fr_bpe.model"
    
    if os.path.exists(spm_en_model) and os.path.exists(spm_fr_model) and HAS_SPM:
        print(f"Loading SentencePiece models...")
        print(f"  - EN: {spm_en_model}")
        print(f"  - FR: {spm_fr_model}")
        
        bpe_en_tokenizer = load_sentencepiece_tokenizer(spm_en_model)
        bpe_fr_tokenizer = load_sentencepiece_tokenizer(spm_fr_model)
        print(f"[OK] SentencePiece tokenizers loaded")
    else:
        print(f"[WARNING]  SentencePiece models not found or sentencepiece not installed.")
        print(f"   Creating fallback tokenizers from vocab...")
        
        bpe_en_tokenizer = create_vocab_based_tokenizer(src_vocab2)
        bpe_fr_tokenizer = create_vocab_based_tokenizer(trg_vocab2)
        
        print(f"[OK] Fallback tokenizers created from vocab")
        print(f"   Note: This is less accurate than true SentencePiece BPE")
    print()
    
    # Calculate BLEU for Model 2 (BPE tokenizer)
    print(f"Calculating BLEU score for {model2_name}...")
    bleu2, preds2, refs2 = calculate_bleu(
        model2, src_vocab2, trg_vocab2, test_en, test_fr,
        en_tokenizer=bpe_en_tokenizer,
        fr_tokenizer=bpe_fr_tokenizer,
        is_attn_model=True,
        use_bpe_decode=True  # CRITICAL: Decode BPE before BLEU calculation
    )
    print(f"[OK] {model2_name} BLEU Score: {bleu2:.2f}")
    print()
    
    # Comparison Summary
    print("="*80)
    print("KẾT QUẢ SO SÁNH")
    print("="*80)
    print(f"{model1_name:40s}: BLEU = {bleu1:6.2f}")
    print(f"{model2_name:40s}: BLEU = {bleu2:6.2f}")
    print("-"*80)
    
    if bleu1 > bleu2:
        diff = bleu1 - bleu2
        print(f"🏆 {model1_name} tốt hơn {model2_name} với chênh lệch {diff:.2f} BLEU points")
    elif bleu2 > bleu1:
        diff = bleu2 - bleu1
        print(f"🏆 {model2_name} tốt hơn {model1_name} với chênh lệch {diff:.2f} BLEU points")
    else:
        print(f"[TIED]  Cả hai models có hiệu suất tương đương")
    print("="*80)
    print()
    
    # Show some example translations
    print("VÍ DỤ DỊCH (5 câu đầu tiên):")
    print("="*80)
    for i in range(min(5, len(test_en))):
        print(f"\n[{i+1}] EN: {test_en[i]}")
        print(f"    FR (reference): {test_fr[i]}")
        print(f"    FR ({model1_name}): {' '.join(preds1[i])}")
        print(f"    FR ({model2_name}): {' '.join(preds2[i])}")
    print("="*80)
    
    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "test_size": len(test_en)
    }



if __name__ == "__main__":
    import sys
    
    # Kiểm tra arguments cho đường dẫn SentencePiece models
    spm_en = sys.argv[1] if len(sys.argv) > 1 else "spm_en_bpe.model"
    spm_fr = sys.argv[2] if len(sys.argv) > 2 else "spm_fr_bpe.model"
    
    print("="*80)
    print("SO SÁNH WORD-LEVEL VS SUBWORD (BPE) MODELS")
    print("="*80)
    print(f"SentencePiece EN model: {spm_en}")
    print(f"SentencePiece FR model: {spm_fr}")
    print()
    
    # So sánh best_model.pth và best_model_subword.pth
    results = compare_models(
        model1_path="best_model.pth",
        model2_path="best_model_subword.pth",
        model1_name="Best Model (Word-level tokenizer)",
        model2_name="Best Model (Subword BPE tokenizer)",
        spm_en_model=spm_en,
        spm_fr_model=spm_fr
    )
    
    if results:
        print("\n[OK] Hoàn thành so sánh!")
        print(f"\n📊 Tóm tắt:")
        print(f"   - {results['model1_name']}: {results['bleu1']:.2f} BLEU")
        print(f"   - {results['model2_name']}: {results['bleu2']:.2f} BLEU")
        print(f"   - Test set: {results['test_size']} câu")
    else:
        print("\n[FAILED] So sánh thất bại!")

