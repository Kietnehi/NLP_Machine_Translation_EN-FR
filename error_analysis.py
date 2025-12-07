"""
Script phân tích lỗi chi tiết cho mô hình Machine Translation
Phục vụ cho phần 9 trong báo cáo
"""
import torch
import sys
import os
from collections import Counter
from inference import load_model_from_checkpoint, _translate_internal
from data import read_parallel_file, en_tokenizer, fr_tokenizer
from config import DATA_ROOT, UNK_TOKEN


def analyze_oov_errors(model, src_vocab, trg_vocab, test_pairs, max_samples=100):
    """
    Phân tích lỗi từ ngoài vốn từ (OOV - Out of Vocabulary)
    """
    print("\n" + "="*60)
    print("1. PHÂN TÍCH LỖI TỪ NGOÀI VỐN TỪ (OOV)")
    print("="*60)
    
    unk_count_src = 0
    unk_count_trg = 0
    total_tokens_src = 0
    total_tokens_trg = 0
    oov_examples = []
    
    for i, (en_sent, fr_sent) in enumerate(test_pairs[:max_samples]):
        # Tokenize
        en_tokens = list(en_tokenizer(en_sent.lower()))
        fr_tokens = list(fr_tokenizer(fr_sent.lower()))
        
        total_tokens_src += len(en_tokens)
        total_tokens_trg += len(fr_tokens)
        
        # Đếm <unk> trong source
        en_unk = sum(1 for token in en_tokens if src_vocab[token] == src_vocab[UNK_TOKEN])
        unk_count_src += en_unk
        
        # Đếm <unk> trong target
        fr_unk = sum(1 for token in fr_tokens if trg_vocab[token] == trg_vocab[UNK_TOKEN])
        unk_count_trg += fr_unk
        
        # Lưu ví dụ có OOV
        if en_unk > 0:
            oov_words = [token for token in en_tokens if src_vocab[token] == src_vocab[UNK_TOKEN]]
            translation = _translate_internal(en_sent, model, src_vocab, trg_vocab)
            oov_examples.append({
                'source': en_sent,
                'target': fr_sent,
                'translation': translation,
                'oov_words': oov_words
            })
    
    # Thống kê
    oov_rate_src = (unk_count_src / total_tokens_src * 100) if total_tokens_src > 0 else 0
    oov_rate_trg = (unk_count_trg / total_tokens_trg * 100) if total_tokens_trg > 0 else 0
    
    print(f"\n📊 Thống kê OOV:")
    print(f"   - Tổng số token (EN): {total_tokens_src}")
    print(f"   - Số token OOV (EN): {unk_count_src}")
    print(f"   - Tỷ lệ OOV (EN): {oov_rate_src:.2f}%")
    print(f"   - Tổng số token (FR): {total_tokens_trg}")
    print(f"   - Số token OOV (FR): {unk_count_trg}")
    print(f"   - Tỷ lệ OOV (FR): {oov_rate_trg:.2f}%")
    
    # In 3 ví dụ OOV
    print(f"\n📝 Ví dụ lỗi OOV (3 câu đầu tiên):")
    for i, ex in enumerate(oov_examples[:3]):
        print(f"\n   Ví dụ {i+1}:")
        print(f"   EN: {ex['source']}")
        print(f"   FR (ground truth): {ex['target']}")
        print(f"   FR (dịch máy): {ex['translation']}")
        print(f"   ❌ Từ OOV: {', '.join(ex['oov_words'])}")
    
    return {
        'oov_rate_src': oov_rate_src,
        'oov_rate_trg': oov_rate_trg,
        'examples': oov_examples[:5]
    }


def analyze_length_errors(model, src_vocab, trg_vocab, test_pairs, max_samples=100):
    """
    Phân tích lỗi liên quan đến độ dài câu
    """
    print("\n" + "="*60)
    print("2. PHÂN TÍCH LỖI ĐỘ DÀI CÂU")
    print("="*60)
    
    short_errors = []  # câu ngắn (< 10 từ)
    medium_errors = []  # câu trung bình (10-20 từ)
    long_errors = []  # câu dài (> 20 từ)
    
    for i, (en_sent, fr_sent) in enumerate(test_pairs[:max_samples]):
        en_tokens = en_sent.split()
        fr_tokens = fr_sent.split() 
        
        translation = _translate_internal(en_sent, model, src_vocab, trg_vocab)
        pred_tokens = translation.split()
        
        length_diff = abs(len(pred_tokens) - len(fr_tokens))
        
        example = {
            'source': en_sent,
            'target': fr_sent,
            'translation': translation,
            'src_len': len(en_tokens),
            'trg_len': len(fr_tokens),
            'pred_len': len(pred_tokens),
            'diff': length_diff
        }
        
        if len(en_tokens) < 10:
            short_errors.append(example)
        elif len(en_tokens) <= 20:
            medium_errors.append(example)
        else:
            long_errors.append(example)
    
    # Tính độ lệch trung bình
    avg_diff_short = sum(e['diff'] for e in short_errors) / len(short_errors) if short_errors else 0
    avg_diff_medium = sum(e['diff'] for e in medium_errors) / len(medium_errors) if medium_errors else 0
    avg_diff_long = sum(e['diff'] for e in long_errors) / len(long_errors) if long_errors else 0
    
    print(f"\n📊 Thống kê độ lệch độ dài:")
    print(f"   - Câu ngắn (<10 từ): {len(short_errors)} câu, độ lệch TB: {avg_diff_short:.2f}")
    print(f"   - Câu trung bình (10-20 từ): {len(medium_errors)} câu, độ lệch TB: {avg_diff_medium:.2f}")
    print(f"   - Câu dài (>20 từ): {len(long_errors)} câu, độ lệch TB: {avg_diff_long:.2f}")
    
    # Tìm câu dài bị lỗi nhiều nhất
    long_errors_sorted = sorted(long_errors, key=lambda x: x['diff'], reverse=True)
    
    print(f"\n📝 Ví dụ câu dài bị dịch sai:")
    for i, ex in enumerate(long_errors_sorted[:2]):
        print(f"\n   Ví dụ {i+1}:")
        print(f"   EN ({ex['src_len']} từ): {ex['source']}")
        print(f"   FR ground truth ({ex['trg_len']} từ): {ex['target']}")
        print(f"   FR dịch máy ({ex['pred_len']} từ): {ex['translation']}")
        print(f"   ❌ Độ lệch: {ex['diff']} từ")
    
    return {
        'avg_diff_short': avg_diff_short,
        'avg_diff_medium': avg_diff_medium,
        'avg_diff_long': avg_diff_long,
        'long_error_examples': long_errors_sorted[:3]
    }


def analyze_grammar_errors(model, src_vocab, trg_vocab, test_pairs, max_samples=50):
    """
    Phân tích lỗi ngữ pháp và thiếu từ (phân tích định tính)
    """
    print("\n" + "="*60)
    print("3. PHÂN TÍCH LỖI NGỮ PHÁP VÀ THIẾU TỪ")
    print("="*60)
    
    grammar_issues = []
    
    # Các pattern lỗi phổ biến
    patterns = {
        'missing_article': ['le', 'la', 'les', 'un', 'une', 'des'],
        'missing_preposition': ['de', 'à', 'en', 'dans', 'sur', 'avec'],
        'missing_verb': ['est', 'sont', 'a', 'ont', 'fait', 'fait']
    }
    
    for i, (en_sent, fr_sent) in enumerate(test_pairs[:max_samples]):
        translation = _translate_internal(en_sent, model, src_vocab, trg_vocab)
        
        fr_tokens_gt = set(fr_tokenizer(fr_sent.lower()))
        fr_tokens_pred = set(translation.split())
        
        missing_words = fr_tokens_gt - fr_tokens_pred
        extra_words = fr_tokens_pred - fr_tokens_gt
        
        # Phân loại lỗi
        missing_articles = [w for w in missing_words if w in patterns['missing_article']]
        missing_preps = [w for w in missing_words if w in patterns['missing_preposition']]
        
        if missing_words:
            grammar_issues.append({
                'source': en_sent,
                'target': fr_sent,
                'translation': translation,
                'missing_words': list(missing_words),
                'extra_words': list(extra_words),
                'missing_articles': missing_articles,
                'missing_preps': missing_preps
            })
    
    print(f"\n📊 Thống kê lỗi ngữ pháp:")
    print(f"   - Tổng số câu có lỗi: {len(grammar_issues)}/{max_samples}")
    
    # Đếm tần suất từ bị thiếu
    all_missing = []
    for issue in grammar_issues:
        all_missing.extend(issue['missing_words'])
    
    missing_counter = Counter(all_missing)
    print(f"\n   📌 Top 10 từ bị thiếu thường xuyên:")
    for word, count in missing_counter.most_common(10):
        print(f"      - '{word}': {count} lần")
    
    print(f"\n📝 Ví dụ lỗi ngữ pháp (3 câu):")
    for i, issue in enumerate(grammar_issues[:3]):
        print(f"\n   Ví dụ {i+1}:")
        print(f"   EN: {issue['source']}")
        print(f"   FR (ground truth): {issue['target']}")
        print(f"   FR (dịch máy): {issue['translation']}")
        print(f"   ❌ Thiếu: {', '.join(issue['missing_words'][:5])}")
        if issue['missing_articles']:
            print(f"   ⚠️  Thiếu mạo từ: {', '.join(issue['missing_articles'])}")
    
    return {
        'total_errors': len(grammar_issues),
        'missing_words_freq': dict(missing_counter.most_common(10)),
        'examples': grammar_issues[:5]
    }


def generate_improvement_suggestions():
    """
    Đề xuất các cải tiến dựa trên phân tích lỗi
    """
    print("\n" + "="*60)
    print("4. ĐỀ XUẤT CẢI TIẾN")
    print("="*60)
    
    suggestions = """
📌 CÁC GIẢI PHÁP CẢI TIẾN:

1️⃣  THÊM CƠ CHẾ ATTENTION
   ✓ Vấn đề: Context vector cố định làm mất thông tin với câu dài
   ✓ Giải pháp: Attention mechanism (Bahdanau hoặc Luong)
   ✓ Kỳ vọng: BLEU tăng 5-10 điểm, dịch câu dài tốt hơn
   ✓ Độ phức tạp: Trung bình
   
2️⃣  SỬ DỤNG SUBWORD TOKENIZATION (BPE)
   ✓ Vấn đề: Tỷ lệ OOV cao (từ hiếm → <unk>)
   ✓ Giải pháp: Byte Pair Encoding hoặc SentencePiece
   ✓ Kỳ vọng: Giảm OOV xuống <1%, xử lý tốt từ ghép và từ mới
   ✓ Độ phức tạp: Thấp (có thư viện hỗ trợ)
   
3️⃣  BEAM SEARCH DECODING
   ✓ Vấn đề: Greedy decoding chọn token tối ưu cục bộ
   ✓ Giải pháp: Beam search với beam_size=5-10
   ✓ Kỳ vọng: BLEU tăng 2-5 điểm, câu dịch tự nhiên hơn
   ✓ Độ phức tạp: Thấp
   
4️⃣  TRANSFORMER ARCHITECTURE
   ✓ Vấn đề: LSTM không song song hóa, khó học phụ thuộc xa
   ✓ Giải pháp: Chuyển sang Transformer (self-attention)
   ✓ Kỳ vọng: BLEU tăng 10-15 điểm, training nhanh hơn
   ✓ Độ phức tạp: Cao
   
5️⃣  PRE-TRAINED MODELS
   ✓ Vấn đề: Dữ liệu training nhỏ (29k câu)
   ✓ Giải pháp: Fine-tune mBART, mT5, hoặc NLLB
   ✓ Kỳ vọng: BLEU tăng 15-20+ điểm
   ✓ Độ phức tạp: Trung bình (cần GPU mạnh)

📊 ƯU TIÊN TRIỂN KHAI:
   Mức 1 (Ngắn hạn):  BPE + Beam Search
   Mức 2 (Trung hạn): Attention mechanism
   Mức 3 (Dài hạn):   Transformer hoặc Pre-trained models
"""
    print(suggestions)
    
    return suggestions


def main():
    """
    Chạy toàn bộ phân tích lỗi
    """
    print("\n" + "="*70)
    print(" "*15 + "PHÂN TÍCH LỖI MÔ HÌNH MACHINE TRANSLATION")
    print("="*70)
    
    # Load model và dữ liệu test
    print("\n⏳ Đang load model và dữ liệu test...")
    model, src_vocab, trg_vocab, pad_idx = load_model_from_checkpoint()
    test_pairs = read_parallel_file(DATA_ROOT, "test")
    print(f"✓ Đã load xong! Số câu test: {len(test_pairs)}")
    
    # Số lượng mẫu để phân tích
    MAX_SAMPLES = 100
    
    # 1. Phân tích OOV
    oov_results = analyze_oov_errors(model, src_vocab, trg_vocab, test_pairs, MAX_SAMPLES)
    
    # 2. Phân tích độ dài
    length_results = analyze_length_errors(model, src_vocab, trg_vocab, test_pairs, MAX_SAMPLES)
    
    # 3. Phân tích ngữ pháp
    grammar_results = analyze_grammar_errors(model, src_vocab, trg_vocab, test_pairs, 50)
    
    # 4. Đề xuất cải tiến
    suggestions = generate_improvement_suggestions()
    
    # Tổng kết
    print("\n" + "="*70)
    print(" "*25 + "TỔNG KẾT PHÂN TÍCH")
    print("="*70)
    print(f"""
📈 KẾT QUẢ PHÂN TÍCH:
   • Tỷ lệ OOV (EN): {oov_results['oov_rate_src']:.2f}%
   • Tỷ lệ OOV (FR): {oov_results['oov_rate_trg']:.2f}%
   • Độ lệch độ dài câu dài: {length_results['avg_diff_long']:.2f} từ
   • Số câu có lỗi ngữ pháp: {grammar_results['total_errors']}/{MAX_SAMPLES}

💡 LỜI KẾT:
   Mô hình hiện tại đạt kết quả khá tốt với câu ngắn và trung bình,
   nhưng còn hạn chế với câu dài và từ hiếm. Các giải pháp cải tiến
   được đề xuất có thể nâng cao hiệu suất đáng kể.
""")
    
    print("\n✅ Hoàn thành phân tích! Kết quả có thể dùng cho phần 9 trong báo cáo.\n")


if __name__ == "__main__":
    main()
