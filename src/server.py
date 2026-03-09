import asyncio
import io
import os
import sys
import time
import random
import subprocess
import threading
import tempfile
import shutil
import wave

import requests
import gradio as gr
from loguru import logger
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils.file_utils import load_voices
from src.queue_manager import QueueManager

# ── Config ─────────────────────────────────────────────────────────────────
VOICE_DIR = "samples"
VOICE_MAP = load_voices(VOICE_DIR)
MODEL_DIR = "pretrained-models"
NUM_WORKERS = int(os.environ.get("TTS_NUM_WORKERS", 1))
WORKER_STARTUP_DELAY = 15  # seconds between each worker start

queue_manager = QueueManager()
_worker_procs: dict[int, subprocess.Popen] = {}
_worker_lock = threading.Lock()
_next_worker_idx = 0

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI_voice API",
    description="AI_voice — Vietnamese TTS with job queue (powered by viet-tts: https://github.com/dangvansam/viet-tts)",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────
class OpenAITTSRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = list(VOICE_MAP)[0] if VOICE_MAP else "0"
    response_format: str = "mp3"
    speed: float = 1.0


class AsyncTTSRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = list(VOICE_MAP)[0] if VOICE_MAP else "0"
    response_format: str = "mp3"
    speed: float = 1.0


# ── Helpers ─────────────────────────────────────────────────────────────────
def _resolve_voice(voice: str):
    if voice.isdigit():
        voices = list(VOICE_MAP.values())
        idx = int(voice)
        return voices[idx] if idx < len(voices) else None
    return VOICE_MAP.get(voice)


def _media_type(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/x-flac",
        "opus": "audio/ogg;codec=opus",
        "aac": "audio/aac",
        "pcm": "audio/pcm;rate=24000",
    }.get(fmt, "audio/mpeg")


async def _wait_for_job(job_id: str, timeout: int = 1200):
    """Poll queue until job completes. Raises on failure/timeout."""
    for _ in range(timeout * 2):  # check every 0.5s
        await asyncio.sleep(0.5)
        job = queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] == "completed":
            return job
        if job["status"] == "failed":
            raise HTTPException(500, job.get("error") or "Job failed")
    raise HTTPException(408, "Request timeout waiting for TTS job")


# ── Basic endpoints ─────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "AI_voice API"


@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "AI_voice API is running..."


@app.get("/voices")
@app.get("/v1/voices")
async def show_voices():
    return list(VOICE_MAP.keys())


# ── Sync TTS (OpenAI-compatible) — submits to queue and waits ───────────────
@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def openai_api_tts(req: OpenAITTSRequest):
    logger.info(f"[API] /audio/speech voice={req.voice} fmt={req.response_format}")
    voice_file = _resolve_voice(req.voice)
    if not voice_file:
        raise HTTPException(404, f"Voice '{req.voice}' not found")

    job_id = queue_manager.add_job(
        text=req.input,
        voice=req.voice,
        voice_path=voice_file,
        speed=req.speed,
        response_format=req.response_format,
    )
    job = await _wait_for_job(job_id)

    with open(job["output_path"], "rb") as f:
        content = f.read()
    return StreamingResponse(
        content=iter([content]),
        media_type=_media_type(req.response_format),
    )


# ── Extended TTS (form-data with voice upload) ──────────────────────────────
@app.post("/tts")
@app.post("/v1/tts")
async def tts(
    text: str = Form(...),
    voice: str = Form("0"),
    speed: float = Form(1.0),
    audio_url: str = Form(None),
    audio_file: UploadFile = File(None),
):
    voice_file = None
    temp_path = None

    if audio_file:
        suffix = "." + audio_file.filename.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        shutil.copyfileobj(audio_file.file, tmp)
        tmp.close()
        audio_file.file.close()
        temp_path = tmp.name
        voice_file = temp_path
        voice = "custom"
    elif audio_url:
        suffix = "." + audio_url.lower().split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        resp = requests.get(audio_url, stream=True)
        if resp.status_code != 200:
            raise HTTPException(400, "Failed to fetch audio from URL")
        shutil.copyfileobj(resp.raw, tmp)
        tmp.close()
        resp.close()
        temp_path = tmp.name
        voice_file = temp_path
        voice = "custom"
    else:
        voice_file = _resolve_voice(voice)
        if not voice_file:
            raise HTTPException(404, "Voice not found")

    if not voice_file or not os.path.exists(voice_file):
        raise HTTPException(400, "No valid voice file")

    job_id = queue_manager.add_job(
        text=text,
        voice=voice,
        voice_path=voice_file,
        speed=speed,
        response_format="mp3",
    )
    try:
        job = await _wait_for_job(job_id)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    return FileResponse(
        path=job["output_path"],
        media_type="audio/mpeg",
        filename=f"tts_{job_id[:8]}.mp3",
    )


# ── Async TTS — returns job_id immediately ───────────────────────────────────
@app.post("/v1/audio/speech/async")
async def async_openai_tts(req: AsyncTTSRequest):
    voice_file = _resolve_voice(req.voice)
    if not voice_file:
        raise HTTPException(404, f"Voice '{req.voice}' not found")
    job_id = queue_manager.add_job(
        text=req.input,
        voice=req.voice,
        voice_path=voice_file,
        speed=req.speed,
        response_format=req.response_format,
    )
    return {"job_id": job_id, "status": "pending",
            "position": queue_manager.get_position(job_id)}


# ── Job endpoints ───────────────────────────────────────────────────────────
@app.get("/v1/jobs/stats")
async def jobs_stats():
    return queue_manager.stats()


@app.get("/v1/jobs")
async def list_jobs(limit: int = 100):
    return queue_manager.list_jobs(limit=limit)


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "pending":
        job["position"] = queue_manager.get_position(job_id)
    return job


@app.get("/v1/jobs/{job_id}/audio")
async def get_job_audio(job_id: str):
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(425, f"Job status is '{job['status']}', not completed")
    path = job.get("output_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Audio file not found on disk")
    fmt = job.get("response_format", "mp3")
    return FileResponse(
        path=path,
        media_type=_media_type(fmt),
        filename=f"{job_id}.{fmt}",
    )


# ── Worker endpoints ────────────────────────────────────────────────────────
@app.get("/v1/workers")
async def list_workers():
    return queue_manager.list_workers()


def _spawn_worker(idx: int, delay: int = 0) -> subprocess.Popen:
    cwd = os.getcwd()
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.worker",
         "--worker-index", str(idx),
         "--model-dir", MODEL_DIR,
         "--startup-delay", str(delay)],
        cwd=cwd,
    )
    return proc


@app.post("/v1/workers")
async def add_worker():
    global _next_worker_idx
    with _worker_lock:
        idx = _next_worker_idx
        _next_worker_idx += 1
    proc = _spawn_worker(idx, delay=0)
    with _worker_lock:
        _worker_procs[idx] = proc
    logger.info(f"[API] Spawned worker-{idx} (PID {proc.pid}) via API")
    return {"worker_id": f"worker-{idx}", "pid": proc.pid, "status": "loading"}


@app.delete("/v1/workers/{worker_id}")
async def remove_worker(worker_id: str):
    try:
        idx = int(worker_id.replace("worker-", ""))
    except ValueError:
        raise HTTPException(400, "Invalid worker_id format")
    with _worker_lock:
        proc = _worker_procs.pop(idx, None)
    if proc is None:
        raise HTTPException(404, f"Worker '{worker_id}' not found")
    try:
        proc.terminate()
    except Exception:
        pass
    queue_manager.delete_worker(worker_id)
    logger.info(f"[API] Terminated {worker_id} via API")
    return {"worker_id": worker_id, "status": "terminated"}


# ── Storage endpoints ────────────────────────────────────────────────────────
@app.get("/v1/storage")
async def storage_info():
    return queue_manager.storage_info()


@app.post("/v1/storage/cleanup")
async def trigger_cleanup(max_age_hours: int = 24):
    return queue_manager.cleanup_old_files(max_age_hours=max_age_hours)


# ── Queue / Worker monitor UI ───────────────────────────────────────────────
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")

QUEUE_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI_voice — Queue Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a; --border: #2d3148;
  --text: #e2e8f0; --muted: #64748b; --accent: #6366f1;
  --pending-bg:#2d2410;--pending-fg:#fbbf24;--pending-dot:#f59e0b;
  --running-bg:#0d1f3c;--running-fg:#60a5fa;--running-dot:#3b82f6;
  --done-bg:#0d2318;--done-fg:#4ade80;--done-dot:#22c55e;
  --failed-bg:#2d0f0f;--failed-fg:#f87171;--failed-dot:#ef4444;
  --idle-bg:#1a1d27;--idle-fg:#94a3b8;
  --radius:12px; --shadow:0 4px 24px rgba(0,0,0,.4);
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}

/* Header */
.header{background:var(--surface);border-bottom:1px solid var(--border);padding:0 32px;
  display:flex;align-items:center;justify-content:space-between;height:60px;
  position:sticky;top:0;z-index:10;}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:1rem;}
.logo-icon{width:32px;height:32px;background:var(--accent);border-radius:8px;
  display:grid;place-items:center;font-size:1rem;}
.live-label{font-size:.75rem;color:var(--done-fg);font-weight:500;display:flex;align-items:center;gap:6px;}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--done-dot);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}

.main{padding:28px 32px;max-width:1600px;margin:0 auto;}

/* Stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 22px;display:flex;align-items:center;gap:14px;
  transition:transform .15s,box-shadow .15s;}
.stat-card:hover{transform:translateY(-2px);box-shadow:var(--shadow);}
.stat-icon{width:46px;height:46px;border-radius:10px;display:grid;place-items:center;font-size:1.2rem;flex-shrink:0;}
.s-pending .stat-icon{background:var(--pending-bg);}
.s-running .stat-icon{background:var(--running-bg);}
.s-done    .stat-icon{background:var(--done-bg);}
.s-failed  .stat-icon{background:var(--failed-bg);}
.stat-num{font-size:1.9rem;font-weight:700;line-height:1;}
.s-pending .stat-num{color:var(--pending-fg);}
.s-running .stat-num{color:var(--running-fg);}
.s-done    .stat-num{color:var(--done-fg);}
.s-failed  .stat-num{color:var(--failed-fg);}
.stat-lbl{font-size:.75rem;color:var(--muted);margin-top:3px;font-weight:500;
  text-transform:uppercase;letter-spacing:.05em;}

/* Workers */
.section-title{font-weight:600;font-size:.85rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:12px;}
.workers{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:12px;margin-bottom:24px;}
.worker-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:16px;transition:border-color .2s;}
.worker-card.busy{border-color:rgba(59,130,246,.4);}
.worker-card.idle{border-color:var(--border);}
.worker-card.loading{border-color:rgba(251,191,36,.3);}
.worker-card.offline{opacity:.5;}
.wk-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.wk-name{font-weight:600;font-size:.9rem;}
.wk-badge{display:inline-flex;align-items:center;gap:5px;
  padding:2px 9px;border-radius:20px;font-size:.7rem;font-weight:600;}
.wk-badge.idle{background:var(--idle-bg);color:var(--idle-fg);}
.wk-badge.busy{background:var(--running-bg);color:var(--running-fg);}
.wk-badge.loading{background:var(--pending-bg);color:var(--pending-fg);}
.wk-badge.offline{background:#1a1a1a;color:#555;}
.wk-dot{width:6px;height:6px;border-radius:50%;}
.wk-badge.idle .wk-dot{background:var(--idle-fg);}
.wk-badge.busy .wk-dot{background:var(--running-dot);animation:pulse 1s infinite;}
.wk-badge.loading .wk-dot{background:var(--pending-dot);}
.wk-badge.offline .wk-dot{background:#555;}
.wk-meta{font-size:.75rem;color:var(--muted);}
.wk-job{font-size:.75rem;color:var(--running-fg);margin-top:6px;
  font-family:'JetBrains Mono',monospace;}
.wk-progress{height:3px;background:var(--border);border-radius:2px;margin-top:10px;overflow:hidden;}
.wk-progress-fill{height:100%;background:linear-gradient(90deg,var(--running-dot),var(--accent));
  animation:progress-anim 1.5s ease-in-out infinite alternate;border-radius:2px;}
@keyframes progress-anim{from{width:15%}to{width:95%}}

/* Table */
.table-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;}
.table-header{display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;border-bottom:1px solid var(--border);}
.table-title{font-weight:600;font-size:.9rem;}
.refresh-info{font-size:.75rem;color:var(--muted);display:flex;align-items:center;gap:6px;}
.spinner{width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}

table{width:100%;border-collapse:collapse;}
thead th{padding:10px 14px;text-align:left;font-size:.7rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  border-bottom:1px solid var(--border);white-space:nowrap;}
tbody tr{border-bottom:1px solid var(--border);transition:background .1s;}
tbody tr:last-child{border-bottom:none;}
tbody tr:hover{background:var(--surface2);}
tbody td{padding:11px 14px;font-size:.82rem;vertical-align:middle;}

.badge{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;
  border-radius:20px;font-size:.7rem;font-weight:600;white-space:nowrap;}
.badge-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.b-pending{background:var(--pending-bg);color:var(--pending-fg);}
.b-pending .badge-dot{background:var(--pending-dot);}
.b-running{background:var(--running-bg);color:var(--running-fg);}
.b-running .badge-dot{background:var(--running-dot);animation:pulse 1.2s infinite;}
.b-completed{background:var(--done-bg);color:var(--done-fg);}
.b-completed .badge-dot{background:var(--done-dot);}
.b-failed{background:var(--failed-bg);color:var(--failed-fg);}
.b-failed .badge-dot{background:var(--failed-dot);}

.text-clip{max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);}
.mono{font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--muted);}

.btn-dl{display:inline-flex;align-items:center;gap:5px;background:var(--done-bg);
  color:var(--done-fg);border:1px solid rgba(74,222,128,.2);
  padding:4px 11px;border-radius:6px;font-size:.72rem;font-weight:600;
  text-decoration:none;transition:background .15s,transform .1s;}
.btn-dl:hover{background:rgba(34,197,94,.15);transform:scale(1.03);}
.err-text{color:var(--failed-fg);font-size:.72rem;max-width:160px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

.empty-state{text-align:center;padding:60px 20px;color:var(--muted);}
.empty-state .icon{font-size:2.5rem;margin-bottom:10px;opacity:.4;}

@media(max-width:900px){
  .main{padding:16px;}
  .stats{grid-template-columns:repeat(2,1fr);}
  .header{padding:0 16px;}
}
</style>
</head>
<body>
<div class="header">
  <div class="logo">
    <div class="logo-icon">🎙</div>
    <span>AI_voice</span>
    <span style="color:var(--muted);font-weight:400">/ Queue Monitor</span>
  </div>
  <div class="live-label"><div class="live-dot"></div>Live</div>
</div>

<div class="main">
  <!-- Stats -->
  <div class="stats">
    <div class="stat-card s-pending">
      <div class="stat-icon">⏳</div>
      <div><div class="stat-num" id="n-pending">–</div><div class="stat-lbl">Đang chờ</div></div>
    </div>
    <div class="stat-card s-running">
      <div class="stat-icon">⚡</div>
      <div><div class="stat-num" id="n-running">–</div><div class="stat-lbl">Đang xử lý</div></div>
    </div>
    <div class="stat-card s-done">
      <div class="stat-icon">✅</div>
      <div><div class="stat-num" id="n-completed">–</div><div class="stat-lbl">Hoàn thành</div></div>
    </div>
    <div class="stat-card s-failed">
      <div class="stat-icon">❌</div>
      <div><div class="stat-num" id="n-failed">–</div><div class="stat-lbl">Thất bại</div></div>
    </div>
  </div>

  <!-- Workers -->
  <div class="section-title">Workers GPU</div>
  <div class="workers" id="workers-grid">
    <div class="worker-card loading">
      <div class="wk-header"><span class="wk-name">–</span>
        <span class="wk-badge loading"><span class="wk-dot"></span>Đang tải...</span>
      </div>
    </div>
  </div>

  <!-- Jobs table -->
  <div class="table-card">
    <div class="table-header">
      <span class="table-title">Danh sách Jobs</span>
      <div class="refresh-info"><div class="spinner"></div><span id="refresh-ts">Đang tải...</span></div>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Job ID</th><th>Trạng thái</th><th>Worker</th>
            <th>Giọng</th><th>Văn bản</th><th>Tốc độ</th><th>Format</th>
            <th>Tạo lúc</th><th>Xử lý</th><th>Thao tác</th>
          </tr>
        </thead>
        <tbody id="tbody">
          <tr><td colspan="10"><div class="empty-state"><div class="icon">📭</div><p>Chưa có job nào</p></div></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

function timeAgo(iso) {
  if (!iso) return '–';
  const d = new Date(iso.endsWith('Z') ? iso : iso+'Z');
  const s = Math.floor((Date.now()-d)/1000);
  if (s<60) return s+'g trước';
  if (s<3600) return Math.floor(s/60)+'ph trước';
  return d.toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit'});
}
function elapsed(a, b) {
  if (!a) return '–';
  const end = b ? new Date((b.endsWith('Z')?b:b+'Z')) : new Date();
  return ((end-new Date((a.endsWith('Z')?a:a+'Z')))/1000).toFixed(1)+'s';
}
const statusLabel = s=>({pending:'Chờ',running:'Đang xử lý',completed:'Hoàn thành',failed:'Thất bại'}[s]||s);
const wkLabel = s=>({idle:'Sẵn sàng',busy:'Đang xử lý',loading:'Tải model',offline:'Offline'}[s]||s);

function setNum(id, val) {
  const el = document.getElementById(id);
  if (el && el.textContent != String(val)) el.textContent = val;
}

async function refresh() {
  try {
    const [jr, sr, wr] = await Promise.all([
      fetch('/v1/jobs'), fetch('/v1/jobs/stats'), fetch('/v1/workers')
    ]);
    const jobs = await jr.json();
    const s = await sr.json();
    const workers = await wr.json();

    setNum('n-pending',   s.pending);
    setNum('n-running',   s.running);
    setNum('n-completed', s.completed);
    setNum('n-failed',    s.failed);

    // Workers grid
    const wHtml = workers.length ? workers.map(w => {
      const st = w.status || 'offline';
      const jobLabel = w.current_job_id
        ? `<div class="wk-job">▶ ${w.current_job_id.slice(0,8)}… ${elapsed(w.started_at||'',null)}</div>` : '';
      const progress = st==='busy'
        ? `<div class="wk-progress"><div class="wk-progress-fill"></div></div>` : '';
      const hb = w.heartbeat_at ? timeAgo(w.heartbeat_at) : '–';
      return `<div class="worker-card ${st}">
        <div class="wk-header">
          <span class="wk-name">${esc(w.id)}</span>
          <span class="wk-badge ${st}"><span class="wk-dot"></span>${wkLabel(st)}</span>
        </div>
        <div class="wk-meta">PID ${w.pid||'–'} · HB ${hb}</div>
        ${jobLabel}${progress}
      </div>`;
    }).join('') : '<div class="worker-card offline"><div class="wk-header"><span class="wk-name">–</span><span class="wk-badge offline"><span class="wk-dot"></span>Không có worker</span></div></div>';
    document.getElementById('workers-grid').innerHTML = wHtml;

    // Jobs table
    const rows = jobs.map(j => {
      const pos = (j.status==='pending'&&j.position) ? ` <span style="opacity:.6">#${j.position}</span>` : '';
      const badge = `<span class="badge b-${j.status}"><span class="badge-dot"></span>${statusLabel(j.status)}${pos}</span>`;
      const wkCell = j.worker_id
        ? `<span class="mono" style="font-size:.7rem">${esc(j.worker_id)}</span>` : '–';
      let action = '';
      if (j.status==='completed')
        action = `<a class="btn-dl" href="/v1/jobs/${j.id}/audio">⬇ Tải</a>`;
      else if (j.status==='failed')
        action = `<span class="err-text" title="${esc(j.error)}">${esc(j.error||'Lỗi')}</span>`;
      else if (j.status==='running')
        action = `<span style="color:var(--running-fg);font-size:.72rem">⚡ ${elapsed(j.started_at,'')}</span>`;
      return `<tr>
        <td><span class="mono" title="${j.id}">${j.id.slice(0,8)}…</span></td>
        <td>${badge}</td>
        <td>${wkCell}</td>
        <td><span style="font-size:.8rem">${esc(j.voice||'–')}</span></td>
        <td><div class="text-clip" title="${esc(j.text)}">${esc(j.text)}</div></td>
        <td><span class="mono">${j.speed}×</span></td>
        <td><span class="mono" style="text-transform:uppercase;font-size:.68rem">${j.response_format||'mp3'}</span></td>
        <td><span title="${j.created_at||''}" style="font-size:.78rem;color:var(--muted)">${timeAgo(j.created_at)}</span></td>
        <td><span class="mono" style="font-size:.78rem">${elapsed(j.started_at,j.completed_at)}</span></td>
        <td>${action}</td>
      </tr>`;
    }).join('');
    document.getElementById('tbody').innerHTML = rows ||
      `<tr><td colspan="10"><div class="empty-state"><div class="icon">📭</div><p>Chưa có job nào. Gửi request đến <code>/v1/audio/speech/async</code> để bắt đầu.</p></div></td></tr>`;

    const now = new Date().toLocaleTimeString('vi-VN');
    document.getElementById('refresh-ts').textContent = `Cập nhật ${now}`;
  } catch(e) {
    document.getElementById('refresh-ts').textContent = '⚠ Lỗi kết nối';
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


@app.get("/queue", response_class=HTMLResponse)
async def queue_monitor():
    # Serve from web/queue.html if exists (hot-reload friendly, no restart needed)
    html_path = os.path.join(_WEB_DIR, "queue.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return QUEUE_HTML


# ── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _next_worker_idx
    # Clear stale worker records from previous run
    queue_manager.clear_workers()
    # Launch NUM_WORKERS external worker processes (staggered)
    for i in range(NUM_WORKERS):
        delay = i * WORKER_STARTUP_DELAY
        proc = _spawn_worker(i, delay=delay)
        with _worker_lock:
            _worker_procs[i] = proc
        logger.info(f"[Server] Launched worker-{i} (PID {proc.pid}, delay={delay}s)")
    _next_worker_idx = NUM_WORKERS

    # Supervisor: restart dead workers + cleanup stale jobs + periodic file cleanup
    def _supervisor():
        last_cleanup = 0
        while True:
            time.sleep(10)
            queue_manager.cleanup_stale_jobs()
            if time.time() - last_cleanup > 3600:
                queue_manager.cleanup_old_files(max_age_hours=24)
                last_cleanup = time.time()
            with _worker_lock:
                snapshot = list(_worker_procs.items())
            dead = [(idx, proc) for idx, proc in snapshot if proc.poll() is not None]
            for restart_idx, (idx, proc) in enumerate(dead):
                if restart_idx > 0:
                    time.sleep(WORKER_STARTUP_DELAY)
                logger.warning(f"[Server] worker-{idx} died (exit={proc.returncode}), restarting...")
                queue_manager.reset_worker_jobs(f"worker-{idx}")
                new_proc = _spawn_worker(idx, delay=0)
                with _worker_lock:
                    _worker_procs[idx] = new_proc
                logger.info(f"[Server] Restarted worker-{idx} (PID {new_proc.pid})")

    threading.Thread(target=_supervisor, daemon=True).start()

    # Gradio UI — submits jobs to queue and polls for result
    def synthesize(text, voice, speed, audio_file):
        if not text or not text.strip():
            return None, "⚠️ Vui lòng nhập văn bản"
        if audio_file:
            voice_file = audio_file
            voice_name = "custom"
        else:
            voice_file = VOICE_MAP.get(voice)
            voice_name = voice
            if not voice_file:
                return None, f"⚠️ Không tìm thấy giọng: {voice}"

        job_id = queue_manager.add_job(
            text=text, voice=voice_name, voice_path=voice_file,
            speed=speed, response_format="wav",
        )
        logger.info(f"[Gradio] Queued job {job_id[:8]} voice={voice_name}")

        # Poll (blocking — runs in threadpool via Gradio)
        for _ in range(2400):  # max 20 min
            time.sleep(0.5)
            job = queue_manager.get_job(job_id)
            if job["status"] == "completed":
                pos = queue_manager.get_position(job_id)
                return job["output_path"], f"✅ Hoàn thành ({job.get('worker_id','?')})"
            if job["status"] == "failed":
                return None, f"❌ Lỗi: {job.get('error','unknown')}"

        return None, "⏱ Timeout"

    with gr.Blocks(title="AI_voice", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ AI_voice — Tổng hợp tiếng Việt")
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Văn bản", placeholder="Nhập văn bản cần đọc...", lines=8
                )
                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        choices=list(VOICE_MAP.keys()),
                        value=list(VOICE_MAP.keys())[0] if VOICE_MAP else None,
                        label="Giọng đọc",
                    )
                    speed_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Tốc độ"
                    )
                audio_upload = gr.Audio(
                    label="Hoặc upload giọng riêng (tuỳ chọn)",
                    type="filepath", sources=["upload"],
                )
                submit_btn = gr.Button("🔊 Tổng hợp", variant="primary")
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Kết quả", type="filepath")
                status_output = gr.Textbox(label="Trạng thái", interactive=False)

        submit_btn.click(
            fn=synthesize,
            inputs=[text_input, voice_dropdown, speed_slider, audio_upload],
            outputs=[audio_output, status_output],
        )

    global app
    app = gr.mount_gradio_app(app, demo, path="/ui")
