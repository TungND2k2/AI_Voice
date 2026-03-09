<p align="center">
  <h1 align="center">🎙️ AI_voice</h1>
  <p align="center">Vietnamese Text-to-Speech Service với Job Queue & Multi-GPU Worker</p>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-green">
  <img src="https://img.shields.io/badge/CUDA-12.2-blue">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/dangvansam/viet-asr"></a>
</p>

> **AI_voice** là service TTS tiếng Việt được xây dựng trên nền tảng mã nguồn mở [VietTTS](https://github.com/dangvansam/viet-tts) bởi [@dangvansam](https://github.com/dangvansam), bổ sung thêm hệ thống job queue, multi-worker GPU, và dashboard quản lý.

---

## ✨ Tính năng

- **Job Queue** — Mọi request được đẩy vào hàng đợi SQLite (WAL mode), xử lý tuần tự không mất request
- **Multi-Worker GPU** — Thêm/xóa worker ngay trên UI, mỗi worker chạy độc lập trên GPU
- **Queue Dashboard** — Giao diện web real-time: xem jobs, workers, VRAM, disk, tải file về
- **OpenAI-compatible API** — Tương thích `POST /v1/audio/speech` của OpenAI
- **Async API** — `POST /v1/audio/speech/async` trả về `job_id` ngay, poll status sau
- **Voice Cloning** — Upload file audio bất kỳ làm giọng đọc
- **Nhiều format** — mp3, wav, flac, opus, aac
- **Cloudflare Tunnel** — Public URL qua tunnel (không cần mở port)

---

## 🚀 Khởi động nhanh

### Yêu cầu
- Docker + Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit
- CUDA 12.x

### Chạy với Docker Compose

```bash
git clone <repo>
cd viet-tts

# Build image
docker compose build

# Chạy (mặc định 1 worker, có thể thêm qua UI)
docker compose up -d

# Xem log
docker compose logs -f api
```

Service khởi động tại `http://localhost:8298`

### Cấu hình số worker mặc định

Sửa trong `docker-compose.yaml`:
```yaml
environment:
  - TTS_NUM_WORKERS=1   # số worker tự động khởi động
```

Hoặc thêm worker thủ công qua Dashboard → **+ Thêm Worker**

---

## 🖥️ Dashboard

Truy cập `/queue` để xem:

- **Stats** — Pending / Running / Completed / Failed
- **GPU Workers** — Trạng thái từng worker, heartbeat, job đang xử lý
- **Jobs** — Danh sách jobs, filter, tìm kiếm, tải file
- **VRAM & Disk** — Hiển thị usage real-time
- **Thêm/Xóa Worker** — Click `+ Thêm Worker` hoặc `×` trên worker card

---

## 📡 API

### Đồng bộ (chờ kết quả)

```bash
curl http://localhost:8298/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Xin chào Việt Nam.",
    "voice": "nu-nhe-nhang",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

### Bất đồng bộ (queue)

```bash
# Tạo job
curl -X POST http://localhost:8298/v1/audio/speech/async \
  -H "Content-Type: application/json" \
  -d '{"input": "Xin chào", "voice": "nu-nhe-nhang"}'
# → {"job_id": "abc123...", "status": "pending", "position": 1}

# Kiểm tra trạng thái
curl http://localhost:8298/v1/jobs/abc123...

# Tải file khi hoàn thành
curl http://localhost:8298/v1/jobs/abc123.../audio --output result.mp3
```

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8298", api_key="unused")

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="nu-nhe-nhang",
    input="Xin chào Việt Nam.",
    response_format="wav"
) as response:
    response.stream_to_file("output.wav")
```

### Quản lý Workers

```bash
# Danh sách workers
GET /v1/workers

# Thêm worker mới
POST /v1/workers

# Xóa worker
DELETE /v1/workers/worker-0
```

### Storage

```bash
# Thông tin disk & VRAM
GET /v1/storage

# Xóa tất cả file đã xong
POST /v1/storage/cleanup?max_age_hours=0

# Xóa file cũ hơn 24h
POST /v1/storage/cleanup?max_age_hours=24
```

---

## 🎤 Danh sách giọng có sẵn

```bash
curl http://localhost:8298/v1/voices
```

| ID | Voice | Gender |
|----|-------|--------|
| 1 | nsnd-le-chuc | 👨 |
| 2 | speechify_10 | 👩 |
| 3 | atuan | 👨 |
| 4 | speechify_11 | 👩 |
| 5 | cdteam | 👨 |
| 6 | nu-nhe-nhang | 👩 |
| 7 | nguyen-ngoc-ngan | 👩 |
| 8 | son-tung-mtp | 👨 |
| ... | ... | ... |

Thêm giọng mới: đặt file `.wav`/`.mp3` vào thư mục `samples/`

---

## 🏗️ Kiến trúc

```
┌─────────────────────────────────────┐
│  FastAPI Server (server.py)          │
│  - Queue API endpoints               │
│  - Gradio UI (/ui)                   │
│  - Dashboard (/queue)                │
│  - Supervisor thread                 │
└──────────────┬──────────────────────┘
               │ SQLite WAL (queue.db)
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 worker-0   worker-1  worker-N
 (GPU)      (GPU)     (GPU)
```

- Server không load model, chỉ quản lý queue
- Mỗi worker là process độc lập, load model riêng
- SQLite `BEGIN IMMEDIATE` đảm bảo không 2 worker lấy cùng 1 job
- Supervisor tự động restart worker chết, requeue job bị mất

---

## 🙏 Credits

Xây dựng trên nền tảng:
- **[VietTTS](https://github.com/dangvansam/viet-tts)** — Model TTS tiếng Việt mã nguồn mở bởi [@dangvansam](https://github.com/dangvansam)
- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** — Base architecture
- **[silero-vad](https://github.com/snakers4/silero-vad)** — VAD model
- **[Vinorm](https://github.com/v-nhandt21/Vinorm)** — Text normalization

## 📜 License

Apache 2.0 (source code) · CC BY-NC (pretrained models & audio samples)
