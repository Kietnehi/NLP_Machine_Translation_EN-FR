# train_eval.py
import torchtext
torchtext.disable_torchtext_deprecation_warning()

import math
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import nltk

from tqdm.auto import tqdm  # <- thêm tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    EMB_DIM,
    CLIP,
    TEACHER_FORCING_RATIO,
    EARLY_STOPPING_PATIENCE,
    MAX_LEN_GENERATE,
    SOS_TOKEN,
    HID_DIM,
    NUM_LAYERS,
    DROPOUT,
    PAD_TOKEN,
    UNK_TOKEN,
    EOS_TOKEN,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
)

from data import build_dataloaders, en_tokenizer, fr_tokenizer
from model import Encoder, Decoder, Seq2Seq

nltk.download("punkt", quiet=True)


# ========== 1. Hàm train & evaluate ==========

def train_one_epoch(model, iterator, optimizer, criterion, pad_idx):
    model.train()
    epoch_loss = 0

    # dùng tqdm để hiển thị tiến độ train
    for src, src_len, trg, trg_len in tqdm(
        iterator, desc="Đang train", total=len(iterator)
    ):
        # trg: [B, trg_len]
        optimizer.zero_grad()

        output = model(
            src,
            src_len,
            trg,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        )
        # output: [B, trg_len, output_dim]

        # Bỏ cột đầu (t=0) vì đó là <sos>, không cần tính loss
        output_dim = output.shape[-1]
        output = output[:, 1:, :].contiguous().view(-1, output_dim)  # [(B*(trg_len-1)), output_dim]
        trg = trg[:, 1:].contiguous().view(-1)                        # [(B*(trg_len-1))]

        loss = criterion(output, trg)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, pad_idx):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        # dùng tqdm để hiển thị tiến độ eval
        for src, src_len, trg, trg_len in tqdm(
            iterator, desc="Đang evaluate", total=len(iterator)
        ):
            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# ========== 2. Hàm translate 1 câu (inference) ==========

def translate_sentence(
    sentence: str,
    model: Seq2Seq,
    src_vocab,
    trg_vocab,
    max_len: int = MAX_LEN_GENERATE,
):
    """
    Dịch 1 câu tiếng Anh -> tiếng Pháp (greedy decoding)
    """
    model.eval()

    # Tokenize English
    tokens_en = [SOS_TOKEN] + list(en_tokenizer(sentence.lower())) + [EOS_TOKEN]
    src_indexes = [src_vocab[token] for token in tokens_en]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_length = torch.tensor([len(src_indexes)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor, src_length)

    # Decoder
    trg_indexes = [trg_vocab[SOS_TOKEN]]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab[EOS_TOKEN]:
            break

    # Chuyển id -> token
    trg_tokens = [trg_vocab.lookup_token(idx) for idx in trg_indexes]
    # Bỏ <sos> và <eos>
    trg_tokens = trg_tokens[1:]
    if EOS_TOKEN in trg_tokens:
        trg_tokens = trg_tokens[: trg_tokens.index(EOS_TOKEN)]

    return " ".join(trg_tokens)


# ========== 3. Tính BLEU trên tập test ==========

def compute_bleu(model, test_loader, src_vocab, trg_vocab, max_samples=None):
    """
    Tính BLEU trên tập test
    max_samples: None = tính hết, hoặc số nguyên để giới hạn
    """
    smoothie = SmoothingFunction().method4
    model.eval()
    scores = []

    # Ngược lookup: id -> token
    inv_trg_vocab = trg_vocab

    with torch.no_grad():
        for i, (src, src_len, trg, trg_len) in enumerate(tqdm(
            test_loader, desc="Tính BLEU", total=len(test_loader)
        )):
            if max_samples is not None and i >= max_samples:
                break

            # Câu gốc tiếng Pháp (target)
            trg_ids = trg[0].tolist()
            trg_tokens = [inv_trg_vocab.lookup_token(idx) for idx in trg_ids]
            # bỏ <sos>, <eos>, <pad>
            clean_trg = []
            for tok in trg_tokens:
                if tok in [SOS_TOKEN, EOS_TOKEN, "<pad>"]:
                    continue
                clean_trg.append(tok)
            reference = [clean_trg]  # sentence_bleu yêu cầu list[list[str]]

            # Lấy câu tiếng Anh gốc từ src -> token
            src_ids = src[0].tolist()
            src_tokens = [src_vocab.lookup_token(idx) for idx in src_ids]
            clean_src_tokens = []
            for tok in src_tokens:
                if tok in [SOS_TOKEN, EOS_TOKEN, "<pad>"]:
                    continue
                clean_src_tokens.append(tok)
            src_sentence = " ".join(clean_src_tokens)

            # Dịch
            hypothesis_sentence = translate_sentence(
                src_sentence, model, src_vocab, trg_vocab
            )
            hypothesis_tokens = hypothesis_sentence.split()

            bleu = sentence_bleu(reference, hypothesis_tokens, smoothing_function=smoothie)
            scores.append(bleu)

    if len(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)


# ========== 4. Hàm main: train + eval + save model ==========

def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs


def print_device_info():
    """
    In ra đang dùng GPU hay CPU dựa trên DEVICE trong config.
    """
    # DEVICE trong config thường là torch.device(...)
    if isinstance(DEVICE, torch.device):
        device_type = DEVICE.type
    else:
        device_type = str(DEVICE)

    if device_type == "cuda" and torch.cuda.is_available():
        print("Thiết bị: GPU (CUDA)")
    else:
        print("Thiết bị: CPU")


def main():
    # In thông tin thiết bị trước
    print_device_info()

    # 1) DataLoader + vocab
    (
        train_loader,
        val_loader,
        test_loader,
        src_vocab,
        trg_vocab,
        pad_idx,
    ) = build_dataloaders()

    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)

    # 2) Model
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
    print(model)

    # 3) Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: giảm LR khi val_loss không giảm
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,  
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
        # verbose parameter deprecated in newer torch versions; omit it
    )
    
    # 4) Train loop + early stopping + checkpoint
    best_valid_loss = float("inf")
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    checkpoint_path = "last_checkpoint.pth"
    start_epoch = 0  # epoch index (0-based) để bắt đầu train

    # ====== LOAD CHECKPOINT NẾU CÓ ======
    if os.path.exists(checkpoint_path):
        print(f"=> Phát hiện {checkpoint_path}, đang load checkpoint để tiếp tục train...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        # Load scheduler nếu có
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # các thông tin khác để tiếp tục
        last_epoch = ckpt.get("epoch", -1)  # 0-based
        start_epoch = last_epoch + 1

        best_valid_loss = ckpt.get("best_valid_loss", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])

        print(f"=> Đã load checkpoint từ epoch {last_epoch + 1}. Sẽ train tiếp từ epoch {start_epoch + 1}.")
    else:
        print("=> Không tìm thấy last_checkpoint.pth, train từ đầu.")

    # ====== VÒNG LẶP TRAIN ======
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()

        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, pad_idx)
        valid_loss = evaluate(model, val_loader, criterion, pad_idx)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | PPL: {math.exp(train_loss):.3f}")
        print(f"\tVal   Loss: {valid_loss:.3f} | PPL: {math.exp(valid_loss):.3f}")
        
        # Scheduler step: giảm LR nếu val_loss không giảm
        scheduler.step(valid_loss)
        # Lấy learning rate hiện tại theo API mới
        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            # fallback nếu phiên bản torch cũ không hỗ trợ get_last_lr
            current_lr = optimizer.param_groups[0]["lr"]
        print(f"\tCurrent LR: {current_lr:.6f}")

        # lưu best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab,
                    "trg_vocab": trg_vocab,
                    "pad_idx": pad_idx,
                },
                "best_model.pth",
            )
            print("=> Lưu best_model.pth (model tốt nhất hiện tại)")
        else:
            epochs_no_improve += 1

        # ====== LƯU CHECKPOINT CUỐI MỖI EPOCH ======
        torch.save(
            {
                "epoch": epoch,  # 0-based
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "src_vocab": src_vocab,
                "trg_vocab": trg_vocab,
                "pad_idx": pad_idx,
                "best_valid_loss": best_valid_loss,
                "epochs_no_improve": epochs_no_improve,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            checkpoint_path,
        )
        print("=> Đã lưu last_checkpoint.pth (checkpoint cuối cùng).")

        # early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping do val_loss không giảm nữa.")
            break

    # 5) Vẽ biểu đồ loss (để chụp cho báo cáo)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train/Val Loss")
    plt.savefig("loss_curve.png")
    print("=> Đã lưu loss_curve.png (chèn vào báo cáo).")

    # 6) Load best model và tính BLEU trên test
    print("Đánh giá BLEU trên tập test...")
    ckpt = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    src_vocab = ckpt["src_vocab"]
    trg_vocab = ckpt["trg_vocab"]
    pad_idx = ckpt["pad_idx"]

    # Tính BLEU trên toàn bộ tập test (bỏ max_samples hoặc để None)
    bleu = compute_bleu(model, test_loader, src_vocab, trg_vocab, max_samples=None)
    print("Kết quả BLEU:")
    print(f"  - Dạng số thực (0 → 1): {bleu:.4f}")
    print(f"  - Dạng phần trăm:       {bleu * 100:.2f}%")


if __name__ == "__main__":
    main()
