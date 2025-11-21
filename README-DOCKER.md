# Docker Deployment Guide - EN-FR Machine Translation

## 📦 Quick Start

### 1. Build và chạy với Docker Compose (Khuyến nghị)
```bash
# Build image và start container
docker-compose up -d

# Xem logs
docker-compose logs -f

# Stop container
docker-compose down
```

### 2. Build và chạy thủ công
```bash
# Build image
docker build -t nlp-translation-en-fr:latest .

# Run container
docker run -d \
  --name nlp-translation \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/best_model.pth:/app/best_model.pth \
  nlp-translation-en-fr:latest
```

---

## 🔧 Cấu trúc Docker

### Dockerfile
- **Base image**: `python:3.10-slim` (lightweight, tối ưu cho production)
- **Dependencies**: Cài đặt PyTorch, spaCy, NLTK và các thư viện cần thiết
- **Port**: 5000 (Flask web app)
- **Health check**: Tự động kiểm tra container health

### docker-compose.yml
- **Service**: `translation-app`
- **Volumes**: Mount data, model checkpoint, static files và templates
- **Network**: Tạo bridge network riêng cho isolation
- **Restart policy**: `unless-stopped` để tự động khởi động lại

---

## 📋 Yêu cầu

### Trước khi chạy Docker
1. ✅ Đảm bảo file `best_model.pth` đã tồn tại (train model trước)
2. ✅ Folder `data/multi30k_en_fr` có dữ liệu đã xử lý
3. ✅ Đã cài Docker và Docker Compose

### Kiểm tra Docker
```bash
docker --version
docker-compose --version
```

---

## 🚀 Các lệnh Docker hữu ích

### Xem logs
```bash
# Toàn bộ logs
docker-compose logs

# Follow logs (real-time)
docker-compose logs -f

# Logs từ specific container
docker logs nlp-translation-en-fr
```

### Quản lý container
```bash
# Start/Stop
docker-compose start
docker-compose stop

# Restart
docker-compose restart

# Remove container và network
docker-compose down

# Remove kèm volumes
docker-compose down -v
```

### Truy cập container
```bash
# Bash shell
docker exec -it nlp-translation-en-fr bash

# Python shell
docker exec -it nlp-translation-en-fr python

# Chạy inference test
docker exec -it nlp-translation-en-fr python inference.py
```

### Kiểm tra tài nguyên
```bash
# Container stats
docker stats nlp-translation-en-fr

# Inspect container
docker inspect nlp-translation-en-fr
```

---

## 🔍 Troubleshooting

### Container không start
```bash
# Xem logs lỗi
docker-compose logs

# Kiểm tra port conflict
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/Mac
```

### Model không load được
```bash
# Kiểm tra mount volumes
docker exec -it nlp-translation-en-fr ls -la /app

# Verify model file
docker exec -it nlp-translation-en-fr python -c "import torch; print(torch.load('best_model.pth', map_location='cpu').keys())"
```

### Out of memory
```bash
# Giới hạn memory cho container
docker run -d \
  --name nlp-translation \
  --memory="2g" \
  --memory-swap="2g" \
  -p 5000:5000 \
  nlp-translation-en-fr:latest
```

### Rebuild sau khi sửa code
```bash
# Rebuild image
docker-compose build --no-cache

# Recreate container
docker-compose up -d --force-recreate
```

---

## 🌐 Truy cập ứng dụng

Sau khi container đã chạy:
- **Web UI**: http://localhost:5000
- **API**: `POST http://localhost:5000/translate`

### Test API
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "A man is riding a bicycle on the street."}'
```

---

## 📊 Production Deployment

### Tối ưu cho production
1. **Multi-stage build** để giảm image size
2. **Environment variables** cho config
3. **Nginx reverse proxy** cho load balancing
4. **SSL/TLS** với Let's Encrypt
5. **Monitoring** với Prometheus + Grafana

### Docker Hub deployment
```bash
# Tag image
docker tag nlp-translation-en-fr:latest username/nlp-translation-en-fr:latest

# Push to Docker Hub
docker push username/nlp-translation-en-fr:latest

# Pull và run từ Docker Hub
docker pull username/nlp-translation-en-fr:latest
docker run -d -p 5000:5000 username/nlp-translation-en-fr:latest
```

---

## ⚙️ Customization

### Thay đổi port
Sửa trong `docker-compose.yml`:
```yaml
ports:
  - "8080:5000"  # Host:Container
```

### Thêm environment variables
```yaml
environment:
  - PYTHONUNBUFFERED=1
  - FLASK_ENV=production
  - MAX_LEN_GENERATE=100
  - BATCH_SIZE=32
```

### GPU support (nếu có NVIDIA GPU)
```yaml
services:
  translation-app:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

## 📝 Notes

- Image size: ~2-3GB (do PyTorch dependencies)
- Container startup time: ~30-60 giây (load model)
- Memory usage: ~1-2GB (tùy model size)
- Khuyến nghị: Dùng `.dockerignore` để giảm build context

---
