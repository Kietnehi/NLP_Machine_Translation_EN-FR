# inference.py
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torch

from config import DEVICE, SOS_TOKEN, EOS_TOKEN, MAX_LEN_GENERATE
from data import en_tokenizer, fr_tokenizer  # chỉ dùng tokenizer
from model import Encoder, Decoder, Seq2Seq


def load_model_from_checkpoint(ckpt_path="best_model.pth"):
    """
    Load model + vocab từ checkpoint để suy luận.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    src_vocab = ckpt["src_vocab"]
    trg_vocab = ckpt["trg_vocab"]
    pad_idx = ckpt["pad_idx"]

    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)

    # THAM SỐ phải giống lúc train
    from config import EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT

    encoder = Encoder(
        input_dim=input_dim,
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model = Seq2Seq(encoder, decoder, device=DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, src_vocab, trg_vocab, pad_idx


def _translate_internal(sentence: str, model, src_vocab, trg_vocab, max_len=MAX_LEN_GENERATE):
    """
    Hàm dịch nội bộ: giống translate_sentence trong train_eval.py
    """
    model.eval()

    tokens_en = [SOS_TOKEN] + list(en_tokenizer(sentence.lower())) + [EOS_TOKEN]
    src_indexes = [src_vocab[token] for token in tokens_en]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_length = torch.tensor([len(src_indexes)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor, src_length)

    trg_indexes = [trg_vocab[SOS_TOKEN]]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab[EOS_TOKEN]:
            break

    trg_tokens = [trg_vocab.lookup_token(idx) for idx in trg_indexes]
    trg_tokens = trg_tokens[1:]
    if EOS_TOKEN in trg_tokens:
        trg_tokens = trg_tokens[: trg_tokens.index(EOS_TOKEN)]

    return " ".join(trg_tokens)


# ===== Hàm bắt buộc theo đề: translate(sentence: str) -> str =====

def translate(sentence: str) -> str:
    """
    Hàm public để nộp đồ án.
    - Lần đầu gọi: load best_model.pth
    - Các lần sau dùng lại model đã load (không load lại cho nhanh).
    """
    # Dùng biến global để cache
    global _MT_MODEL
    global _SRC_VOCAB
    global _TRG_VOCAB
    global _PAD_IDX

    if "_MT_MODEL" not in globals():
        _MT_MODEL, _SRC_VOCAB, _TRG_VOCAB, _PAD_IDX = load_model_from_checkpoint()

    return _translate_internal(sentence, _MT_MODEL, _SRC_VOCAB, _TRG_VOCAB)


if __name__ == "__main__":
    # Demo nhanh: in ra 5 ví dụ để kiểm tra chức năng translate()
    examples = [
        "A man is riding a bicycle on the street.",
        "A woman is holding a baby in her arms.",
        "Two dogs are playing in the park.",
        "A group of people are sitting at a table eating.",
        "The plane is flying over the mountains.",
    ]

    print("Demo translate (5 examples):")
    for s in examples:
        print("EN:", s)
        try:
            print("FR:", translate(s))
        except Exception as e:
            print("FR: <error>", e)
        print()
