import os
import gzip

RAW_DIR = r"C:\Users\ADMIN\Desktop\NLP PROJECT\multi30k-dataset\data\task1\raw"
OUT_DIR = r"C:\Users\ADMIN\Desktop\NLP PROJECT\data\multi30k_en_fr"

os.makedirs(OUT_DIR, exist_ok=True)

FILE_MAP = {
    "train.en.gz": "train.en",
    "train.fr.gz": "train.fr",
    "val.en.gz": "val.en",
    "val.fr.gz": "val.fr",

    # Test set chọn Flickr 2016 (chuẩn 1k câu)
    "test_2016_flickr.en.gz": "test.en",
    "test_2016_flickr.fr.gz": "test.fr",
}

def extract_gz(src_path, dst_path):
    print(f"[*] Giải nén: {src_path} → {dst_path}")
    with gzip.open(src_path, "rt", encoding="utf-8") as fin:
        with open(dst_path, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)

def main():
    print("[*] RAW_DIR:", RAW_DIR)
    print("[*] OUT_DIR:", OUT_DIR)

    for gz_name, out_name in FILE_MAP.items():
        gz_path = os.path.join(RAW_DIR, gz_name)
        out_path = os.path.join(OUT_DIR, out_name)

        if not os.path.exists(gz_path):
            print(f"[X] Không tìm thấy: {gz_path}")
            continue

        extract_gz(gz_path, out_path)

    print("\nHoàn tất! Các file đã tạo trong:")
    print(OUT_DIR)
    print(os.listdir(OUT_DIR))

if __name__ == "__main__":
    main()
