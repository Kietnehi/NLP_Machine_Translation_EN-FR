# config.py
import torch

# ============ ĐƯỜNG DẪN DỮ LIỆU ============
# Sửa chỗ này cho đúng với máy bạn!
DATA_ROOT = r"./data/multi30k_en_fr"  # chứa train.en, train.fr, ...

# ============ HYPERPARAMETER ===============
MAX_VOCAB_SIZE = 10_000

EMB_DIM = 256         # có thể tăng lên 512
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
CLIP = 1.0            # gradient clipping

EARLY_STOPPING_PATIENCE = 3

MAX_LEN_GENERATE = 50

# ============ THIẾT BỊ =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ TOKEN ĐẶC BIỆT ===============
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
