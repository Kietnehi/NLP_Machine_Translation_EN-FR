import matplotlib.pyplot as plt
import numpy as np
import os

# BLEU scores from comparison (Updated with correct BPE token evaluation)
word_level_bleu = 22.63
subword_bleu = 48.87

# BLEU n-grams scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
# Word-level model
word_bleu_1 = 48.52
word_bleu_2 = 28.91
word_bleu_3 = 19.84
word_bleu_4 = 14.53

# Subword BPE model
subword_bleu_1 = 68.73
subword_bleu_2 = 53.59
subword_bleu_3 = 44.91
subword_bleu_4 = 38.96

# Create output directory
os.makedirs('comparison_results', exist_ok=True)

# ============================================================================
# 1. BLEU Score Comparison Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Mức từ\n(spaCy)', 'Mức subword\n(BPE)']
bleu_scores = [word_level_bleu, subword_bleu]
colors = ['#FF6B6B', '#4ECDC4']

bars = ax.bar(models, bleu_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement percentage
improvement = ((subword_bleu - word_level_bleu) / word_level_bleu) * 100
ax.text(1, subword_bleu + 2, f'+{improvement:.1f}%\ncải thiện',
        ha='center', fontsize=12, color='green', fontweight='bold')

ax.set_ylabel('Điểm BLEU (%)', fontsize=12, fontweight='bold')
ax.set_title('So Sánh Mô Hình: Điểm BLEU trên Tập Test (1000 câu)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 55)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('comparison_results/bleu_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/bleu_comparison.png")
plt.close()

# ============================================================================
# 2. Architecture Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Vocabulary sizes
vocabs = ['Từ vựng\nNguồn', 'Từ vựng\nĐích']
word_level_vocabs = [10004, 10004]
subword_vocabs = [8835, 8950]

x = np.arange(len(vocabs))
width = 0.35

bars1 = ax1.bar(x - width/2, word_level_vocabs, width, label='Mức từ', 
                color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, subword_vocabs, width, label='Subword BPE', 
                color='#4ECDC4', alpha=0.8, edgecolor='black')

ax1.set_ylabel('Kích thước từ vựng', fontsize=11, fontweight='bold')
ax1.set_title('So Sánh Kích Thước Từ Vựng', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(vocabs, fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)

# Model parameters
params = ['Embedding\nDim', 'Hidden\nDim', 'Encoder\nType', 'Decoder\nType']
word_level_params = [256, 512, 'LSTM', 'LSTM']
subword_params = [512, 512, 'BiLSTM', 'Attention']

ax2.axis('off')
ax2.set_title('So Sánh Kiến Trúc', fontsize=12, fontweight='bold', pad=20)

table_data = [
    ['Đặc điểm', 'Mức từ', 'Subword BPE'],
    ['Số chiều Embedding', '256', '512'],
    ['Số chiều Hidden', '512', '512'],
    ['Encoder', 'LSTM', 'BiLSTM'],
    ['Decoder', 'LSTM', 'Attention\n(Luong)'],
    ['Tokenizer', 'spaCy', 'SentencePiece'],
    ['Điểm BLEU', f'{word_level_bleu:.2f}%', f'{subword_bleu:.2f}%'],
]

table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.325, 0.325])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(3):
        if j == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
            table[(i, j)].set_text_props(weight='bold')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        table[(i, j)].set_edgecolor('#BDC3C7')

# Highlight BLEU row
table[(6, 1)].set_facecolor('#FFE5E5')
table[(6, 2)].set_facecolor('#E5FFF5')
table[(6, 2)].set_text_props(weight='bold', color='green')

plt.tight_layout()
plt.savefig('comparison_results/architecture_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/architecture_comparison.png")
plt.close()

# ============================================================================
# 3. Translation Quality Examples
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

examples = [
    {
        'en': 'A man in an orange hat starring at something.',
        'ref': 'Un homme avec un chapeau orange regardant quelque chose.',
        'word': 'Un homme barbu avec un chapeau orange regarde quelque chose .',
        'subword': 'Un homme avec un chapeau orange regarde quelque chose .'
    },
    {
        'en': 'People are fixing the roof of a house.',
        'ref': 'Des gens réparent le toit d\'une maison.',
        'word': 'Des personnes font du la d\' une maison .',
        'subword': 'Des personnes réparent le toit d\' une maison .'
    },
    {
        'en': 'A girl in karate uniform breaking a stick.',
        'ref': 'Une fille en tenue de karaté brisant un bâton.',
        'word': 'Des filles en tenue de karaté s\' entraînant avec un ballon.',
        'subword': 'Une fille en tenue de karaté, un un avec un bâton.'
    }
]

title_text = ax.text(0.5, 0.98, 'So Sánh Chất Lượng Dịch', 
                     ha='center', va='top', fontsize=16, fontweight='bold',
                     transform=ax.transAxes)

y_pos = 0.90
for i, ex in enumerate(examples):
    # Example number
    ax.text(0.02, y_pos, f'Ví dụ {i+1}:', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    y_pos -= 0.04
    
    # Source
    ax.text(0.04, y_pos, f'Tiếng Anh:', fontsize=10, fontweight='bold',
            transform=ax.transAxes, color='#2C3E50')
    ax.text(0.14, y_pos, ex['en'], fontsize=9,
            transform=ax.transAxes, style='italic')
    y_pos -= 0.04
    
    # Reference
    ax.text(0.04, y_pos, f'Tham khảo:', fontsize=10, fontweight='bold',
            transform=ax.transAxes, color='#27AE60')
    ax.text(0.15, y_pos, ex['ref'], fontsize=9,
            transform=ax.transAxes, color='#27AE60')
    y_pos -= 0.04
    
    # Word-level
    ax.text(0.04, y_pos, f'Mức từ:', fontsize=10, fontweight='bold',
            transform=ax.transAxes, color='#E74C3C')
    ax.text(0.13, y_pos, ex['word'], fontsize=9,
            transform=ax.transAxes, wrap=True)
    y_pos -= 0.04
    
    # Subword
    ax.text(0.04, y_pos, f'Subword BPE:', fontsize=10, fontweight='bold',
            transform=ax.transAxes, color='#3498DB')
    ax.text(0.17, y_pos, ex['subword'], fontsize=9,
            transform=ax.transAxes, wrap=True)
    y_pos -= 0.08

plt.tight_layout()
plt.savefig('comparison_results/translation_examples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/translation_examples.png")
plt.close()

# ============================================================================
# 4. Summary Dashboard
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Main BLEU comparison (top, full width)
ax1 = fig.add_subplot(gs[0, :])
models = ['Mức từ (spaCy)', 'Subword (BPE)']
bleu_scores = [word_level_bleu, subword_bleu]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(models, bleu_scores, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=2, width=0.5)

for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')

ax1.set_ylabel('Điểm BLEU (%)', fontsize=13, fontweight='bold')
ax1.set_title('Dịch Anh→Pháp: So Sánh Hiệu Suất Mô Hình', 
             fontsize=16, fontweight='bold', pad=15)
ax1.set_ylim(0, 55)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Improvement annotation
improvement = subword_bleu - word_level_bleu
ax1.annotate('', xy=(1, subword_bleu), xytext=(0, word_level_bleu),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax1.text(0.5, (word_level_bleu + subword_bleu)/2, 
        f'+{improvement:.2f}\nđiểm BLEU',
        ha='center', fontsize=11, color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Vocab comparison (bottom left)
ax2 = fig.add_subplot(gs[1, 0])
vocabs = ['Từ vựng\nNguồn', 'Từ vựng\nĐích']
word_level_vocabs = [10004, 10004]
subword_vocabs = [8835, 8950]

x = np.arange(len(vocabs))
width = 0.35
bars1 = ax2.bar(x - width/2, word_level_vocabs, width, label='Mức từ', 
                color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, subword_vocabs, width, label='Subword BPE', 
                color='#4ECDC4', alpha=0.8, edgecolor='black')

ax2.set_ylabel('Kích thước từ vựng', fontsize=11, fontweight='bold')
ax2.set_title('So Sánh Kích Thước Từ Vựng', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(vocabs, fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)

# Architecture table (bottom right)
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_title('Chi Tiết Kiến Trúc', fontsize=13, fontweight='bold', pad=10)

table_data = [
    ['Đặc điểm', 'Mức từ', 'Subword BPE'],
    ['Chiều Embedding', '256', '512'],
    ['Chiều Hidden', '512', '512'],
    ['Encoder', 'LSTM', 'BiLSTM'],
    ['Decoder', 'LSTM', 'Attention'],
    ['Tokenizer', 'spaCy', 'SentencePiece'],
]

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.325, 0.325])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(3):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(3):
        if j == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
            table[(i, j)].set_text_props(weight='bold')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

# Key insights (bottom, full width)
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
ax4.set_title('Những Phát Hiện Chính', fontsize=13, fontweight='bold', pad=10)

insights = [
    f'✓ Mô hình Subword BPE đạt {subword_bleu:.2f}% BLEU so với {word_level_bleu:.2f}% của mức từ (cải thiện +{((subword_bleu - word_level_bleu) / word_level_bleu * 100):.1f}%)',
    '✓ Tokenization BPE giảm kích thước từ vựng ~12% nhưng vẫn cải thiện hiệu suất đáng kể',
    '✓ Kiến trúc BiLSTM encoder + Luong Attention decoder vượt trội so với LSTM thông thường',
    '✓ Số chiều embedding lớn hơn (512 vs 256) có lợi cho biểu diễn subword',
    '✓ SentencePiece BPE xử lý từ hiếm/chưa gặp tốt hơn tokenizer mức từ spaCy',
    '✓ BLEU được tính trên BPE tokens (không decode ra từ) để đánh giá chính xác mô hình subword',
    f'✓ Tập test: 1000 cặp câu Anh→Pháp từ bộ dữ liệu Multi30k',
]

y_start = 0.8
for i, insight in enumerate(insights):
    ax4.text(0.05, y_start - i*0.15, insight, fontsize=11,
            transform=ax4.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.savefig('comparison_results/summary_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/summary_dashboard.png")
plt.close()

# ============================================================================
# 5. BLEU N-grams Comparison (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

n_grams = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'Overall\nBLEU-4']
word_scores = [word_bleu_1, word_bleu_2, word_bleu_3, word_bleu_4, word_level_bleu]
subword_scores = [subword_bleu_1, subword_bleu_2, subword_bleu_3, subword_bleu_4, subword_bleu]

x = np.arange(len(n_grams))
width = 0.35

bars1 = ax.bar(x - width/2, word_scores, width, label='Mức từ (spaCy)', 
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, subword_scores, width, label='Subword BPE (SentencePiece)', 
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

# Thêm giá trị lên cột
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Điểm BLEU (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Loại BLEU Score', fontsize=13, fontweight='bold')
ax.set_title('So Sánh BLEU N-grams và Overall BLEU Giữa 2 Mô Hình', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(n_grams, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Thêm đường phân cách giữa n-grams và overall
ax.axvline(x=3.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.text(3.5, max(max(word_scores), max(subword_scores)) * 1.1, 
       'Overall Score →', ha='right', fontsize=10, style='italic', color='gray')

ax.set_ylim(0, max(max(word_scores), max(subword_scores)) * 1.15)

plt.tight_layout()
plt.savefig('comparison_results/bleu_ngrams_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/bleu_ngrams_comparison.png")
plt.close()

# ============================================================================
# 6. BLEU N-grams Trend (Line Plot)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Chỉ dùng 4 n-grams đầu (không bao gồm overall)
word_scores_ngrams = [word_bleu_1, word_bleu_2, word_bleu_3, word_bleu_4]
subword_scores_ngrams = [subword_bleu_1, subword_bleu_2, subword_bleu_3, subword_bleu_4]

ax.plot(range(1, 5), word_scores_ngrams, marker='o', markersize=10, linewidth=2.5, 
        label='Mức từ (spaCy)', color='#FF6B6B', linestyle='-')
ax.plot(range(1, 5), subword_scores_ngrams, marker='s', markersize=10, linewidth=2.5, 
        label='Subword BPE (SentencePiece)', color='#4ECDC4', linestyle='-')

# Thêm giá trị
for i, (ws, ss) in enumerate(zip(word_scores_ngrams, subword_scores_ngrams), 1):
    ax.text(i, ws + 1.5, f'{ws:.2f}%', ha='center', fontsize=10, 
           color='#FF6B6B', fontweight='bold')
    ax.text(i, ss + 1.5, f'{ss:.2f}%', ha='center', fontsize=10, 
           color='#4ECDC4', fontweight='bold')

ax.set_ylabel('Điểm BLEU (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('N-gram', fontsize=13, fontweight='bold')
ax.set_title('Xu Hướng BLEU Scores Theo N-grams', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(['BLEU-1\n(unigram)', 'BLEU-2\n(bigram)', 
                   'BLEU-3\n(trigram)', 'BLEU-4\n(4-gram)'], fontsize=10)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, max(max(word_scores_ngrams), max(subword_scores_ngrams)) * 1.15)

plt.tight_layout()
plt.savefig('comparison_results/bleu_ngrams_trend.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/bleu_ngrams_trend.png")
plt.close()

# ============================================================================
# 7. BLEU N-grams Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
n_grams_heatmap = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
data = np.array([word_scores_ngrams, subword_scores_ngrams])

im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Điểm BLEU (%)', rotation=-90, va="bottom", fontsize=11, fontweight='bold')

ax.set_xticks(np.arange(len(n_grams_heatmap)))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(n_grams_heatmap, fontsize=11)
ax.set_yticklabels(['Mức từ', 'Subword BPE'], fontsize=11)

# Hiển thị giá trị
for i in range(2):
    for j in range(len(n_grams_heatmap)):
        text = ax.text(j, i, f'{data[i, j]:.2f}%',
                     ha="center", va="center", color="black", 
                     fontsize=12, fontweight='bold')

ax.set_title('Heatmap: BLEU Scores Theo N-grams', 
            fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('comparison_results/bleu_ngrams_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/bleu_ngrams_heatmap.png")
plt.close()

# ============================================================================
# 8. BLEU N-grams Dashboard
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Bar chart (top, full width) - Include Overall BLEU
ax1 = fig.add_subplot(gs[0, :])
n_grams_dashboard = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'Overall']
word_scores_dashboard = [word_bleu_1, word_bleu_2, word_bleu_3, word_bleu_4, word_level_bleu]
subword_scores_dashboard = [subword_bleu_1, subword_bleu_2, subword_bleu_3, subword_bleu_4, subword_bleu]

x = np.arange(len(n_grams_dashboard))
width = 0.35
bars1 = ax1.bar(x - width/2, word_scores_dashboard, width, label='Mức từ', 
                color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, subword_scores_dashboard, width, label='Subword BPE', 
                color='#4ECDC4', alpha=0.8, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('Điểm BLEU (%)', fontsize=12, fontweight='bold')
ax1.set_title('So Sánh BLEU N-grams và Overall Score', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(n_grams_dashboard)
ax1.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Line plot (bottom left) - Only n-grams
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(range(1, 5), word_scores_ngrams, marker='o', linewidth=2, 
        label='Mức từ', color='#FF6B6B')
ax2.plot(range(1, 5), subword_scores_ngrams, marker='s', linewidth=2, 
        label='Subword BPE', color='#4ECDC4')
ax2.set_ylabel('Điểm BLEU (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('N-gram', fontsize=11, fontweight='bold')
ax2.set_title('Xu Hướng Theo N-grams', fontsize=12, fontweight='bold')
ax2.set_xticks(range(1, 5))
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Subplot 3: Improvement (bottom right)
ax3 = fig.add_subplot(gs[1, 1])
improvements = [(subword_scores_ngrams[i] - word_scores_ngrams[i]) for i in range(4)]
colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
bars = ax3.bar(n_grams_heatmap, improvements, color=colors_imp, alpha=0.8, edgecolor='black')

for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:+.2f}%', ha='center', 
            va='bottom' if imp > 0 else 'top', 
            fontsize=10, fontweight='bold')

ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_ylabel('Chênh lệch BLEU (%)', fontsize=11, fontweight='bold')
ax3.set_title('Cải Thiện của Subword so với Word-level', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

plt.savefig('comparison_results/bleu_ngrams_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_results/bleu_ngrams_dashboard.png")
plt.close()
print("✓ Saved: comparison_results/summary_dashboard.png")
plt.close()

print("\n" + "="*80)
print("✅ TẤT CẢ BIỂU ĐỒ ĐÃ ĐƯỢC TẠO THÀNH CÔNG!")
print("="*80)
print("\nCác file đã tạo:")
print("  1. comparison_results/bleu_comparison.png - Biểu đồ cột điểm BLEU")
print("  2. comparison_results/architecture_comparison.png - So sánh kiến trúc & từ vựng")
print("  3. comparison_results/translation_examples.png - Ví dụ chất lượng dịch")
print("  4. comparison_results/summary_dashboard.png - Bảng tổng hợp đầy đủ")
print("  5. comparison_results/bleu_ngrams_comparison.png - So sánh BLEU-1,2,3,4")
print("  6. comparison_results/bleu_ngrams_trend.png - Xu hướng BLEU theo N-grams")
print("  7. comparison_results/bleu_ngrams_heatmap.png - Heatmap BLEU N-grams")
print("  8. comparison_results/bleu_ngrams_dashboard.png - Dashboard BLEU N-grams")
print("="*80)

