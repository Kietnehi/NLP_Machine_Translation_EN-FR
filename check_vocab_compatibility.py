import torch
import sentencepiece as spm

# Check checkpoint
print('=== CHECKPOINT INFO ===')
ckpt = torch.load('best_model_subword.pth', map_location='cpu')
print('Keys:', list(ckpt.keys()))

src_vocab = ckpt['src_vocab']
trg_vocab = ckpt['trg_vocab']

print(f'Source vocab size: {len(src_vocab)}')
print(f'Target vocab size: {len(trg_vocab)}')

# Sample tokens to check BPE
print('\n=== SAMPLE VOCAB TOKENS ===')
print('Source tokens (50-80):')
src_sample = [src_vocab.lookup_token(i) for i in range(50, 80)]
print(src_sample)

print('\nTarget tokens (50-80):')
trg_sample = [trg_vocab.lookup_token(i) for i in range(50, 80)]
print(trg_sample)

# Check for BPE markers
has_bpe_src = any('\u2581' in str(token) for token in src_sample)  # Unicode for ▁
has_bpe_trg = any('\u2581' in str(token) for token in trg_sample)
print(f'\nSource has BPE markers: {has_bpe_src}')
print(f'Target has BPE markers: {has_bpe_trg}')

# Check SentencePiece models
print('\n=== SENTENCEPIECE MODELS ===')
sp_en = spm.SentencePieceProcessor()
sp_en.load('spm_en_bpe.model')
print(f'EN SentencePiece vocab size: {sp_en.get_piece_size()}')
print(f'EN SentencePiece sample: {sp_en.encode_as_pieces("hello world")}')

sp_fr = spm.SentencePieceProcessor()
sp_fr.load('spm_fr_bpe.model')
print(f'FR SentencePiece vocab size: {sp_fr.get_piece_size()}')
print(f'FR SentencePiece sample: {sp_fr.encode_as_pieces("bonjour monde")}')

# CRITICAL: Check if checkpoint vocab matches SentencePiece
print('\n=== VOCAB COMPATIBILITY CHECK ===')
# Encode sample and check if tokens exist in checkpoint vocab
test_en = sp_en.encode_as_pieces('a man is walking')
test_fr = sp_fr.encode_as_pieces('un homme marche')

print(f'SP EN tokens: {test_en}')
print(f'SP FR tokens: {test_fr}')

# Check if these tokens are in checkpoint vocab
en_in_vocab = [token in src_vocab for token in test_en]
fr_in_vocab = [token in trg_vocab for token in test_fr]

print(f'EN tokens in checkpoint vocab: {en_in_vocab}')
print(f'FR tokens in checkpoint vocab: {fr_in_vocab}')

missing_en = [token for token, in_v in zip(test_en, en_in_vocab) if not in_v]
missing_fr = [token for token, in_v in zip(test_fr, fr_in_vocab) if not in_v]

if missing_en or missing_fr:
    print('\n[PROBLEM] SentencePiece models DO NOT match checkpoint vocab!')
    print('   -> Checkpoint was trained with DIFFERENT SentencePiece models!')
    if missing_en:
        print(f'   Missing EN tokens: {missing_en}')
    if missing_fr:
        print(f'   Missing FR tokens: {missing_fr}')
else:
    print('\n[OK] SentencePiece models match checkpoint vocab!')
