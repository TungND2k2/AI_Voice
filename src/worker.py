"""Standalone TTS worker process.

Launch via:
    python -m viettts.worker --worker-index 0 --model-dir pretrained-models
"""
import os
import sys
import time
import signal
import sqlite3
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger

OUTPUT_DIR = Path("/tmp/tts_outputs")
QUEUE_DB = str(OUTPUT_DIR / "queue.db")


# ── DB helpers ──────────────────────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(QUEUE_DB, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _register(worker_id, pid):
    conn = _db()
    conn.execute("""
        INSERT OR REPLACE INTO workers (id, pid, status, started_at, heartbeat_at)
        VALUES (?, ?, 'loading', ?, ?)
    """, (worker_id, pid, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def _set_idle(worker_id):
    conn = _db()
    conn.execute(
        "UPDATE workers SET status='idle', current_job_id=NULL, heartbeat_at=? WHERE id=?",
        (datetime.utcnow().isoformat(), worker_id)
    )
    conn.commit()
    conn.close()


def _set_offline(worker_id):
    conn = _db()
    conn.execute(
        "UPDATE workers SET status='offline', current_job_id=NULL WHERE id=?",
        (worker_id,)
    )
    conn.commit()
    conn.close()


def _heartbeat(worker_id):
    conn = _db()
    conn.execute(
        "UPDATE workers SET heartbeat_at=? WHERE id=?",
        (datetime.utcnow().isoformat(), worker_id)
    )
    conn.commit()
    conn.close()


def _claim_job(worker_id):
    """Atomically claim next pending job. Returns job dict or None."""
    conn = _db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row:
            job = dict(row)
            now = datetime.utcnow().isoformat()
            conn.execute(
                "UPDATE jobs SET status='running', worker_id=?, started_at=? WHERE id=?",
                (worker_id, now, job["id"])
            )
            conn.execute(
                "UPDATE workers SET status='busy', current_job_id=?, heartbeat_at=? WHERE id=?",
                (job["id"], now, worker_id)
            )
            conn.commit()
            return job
        conn.commit()
        return None
    except Exception as e:
        conn.rollback()
        logger.error(f"claim_job error: {e}")
        return None
    finally:
        conn.close()


def _finish_job(worker_id, job_id, output_path=None, error=None):
    conn = _db()
    now = datetime.utcnow().isoformat()
    if error:
        conn.execute(
            "UPDATE jobs SET status='failed', completed_at=?, error=? WHERE id=?",
            (now, str(error), job_id)
        )
    else:
        conn.execute(
            "UPDATE jobs SET status='completed', completed_at=?, output_path=? WHERE id=?",
            (now, output_path, job_id)
        )
    conn.execute(
        "UPDATE workers SET status='idle', current_job_id=NULL, heartbeat_at=? WHERE id=?",
        (now, worker_id)
    )
    conn.commit()
    conn.close()


# ── TTS processing ──────────────────────────────────────────────────────────

def _build_ffmpeg(fmt, output_path):
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-f", "f32le", "-ar", "24000", "-ac", "1", "-i", "-"]
    if fmt == "mp3":
        cmd += ["-f", "mp3", "-c:a", "libmp3lame", "-ab", "64k"]
    elif fmt == "wav":
        cmd += ["-f", "wav", "-c:a", "pcm_s16le"]
    elif fmt == "flac":
        cmd += ["-f", "flac", "-c:a", "flac"]
    elif fmt == "opus":
        cmd += ["-f", "ogg", "-c:a", "libopus"]
    elif fmt == "aac":
        cmd += ["-f", "adts", "-c:a", "aac", "-ab", "64k"]
    else:
        cmd += ["-f", "mp3", "-c:a", "libmp3lame", "-ab", "64k"]
    return cmd + [output_path]


def _update_progress(job_id, progress):
    conn = _db()
    conn.execute("UPDATE jobs SET progress=? WHERE id=?", (progress, job_id))
    conn.commit()
    conn.close()


def _process(tts, job, worker_id):
    from src.utils.file_utils import load_prompt_speech_from_file

    prompt = load_prompt_speech_from_file(
        filepath=job["voice_path"], min_duration=3, max_duration=5
    )
    sentences = tts.frontend.preprocess_text(job["text"], split=True)
    total = len(sentences)
    logger.info(f"[{worker_id}] job {job['id'][:8]} — {total} câu")

    raw_audio = b""
    for idx, sentence in enumerate(sentences, 1):
        _heartbeat(worker_id)
        progress = round(idx / total * 100, 1)
        _update_progress(job["id"], progress)
        logger.info(f"[{worker_id}] câu {idx}/{total} ({progress}%)")
        model_input = tts.frontend.frontend_tts(sentence, prompt)
        for out in tts.model.tts(**model_input, stream=False, speed=float(job["speed"])):
            raw_audio += out["tts_speech"].numpy().tobytes()

    fmt = job.get("response_format", "mp3")
    output_path = str(OUTPUT_DIR / f"{job['id']}.{fmt}")
    result = subprocess.run(
        _build_ffmpeg(fmt, output_path),
        input=raw_audio,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg: {result.stderr.decode()}")
    return output_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VietTTS worker process")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--model-dir", default="pretrained-models")
    parser.add_argument("--startup-delay", type=int, default=0,
                        help="Seconds to wait before loading model (stagger start)")
    args = parser.parse_args()

    worker_id = f"worker-{args.worker_index}"
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.startup_delay > 0:
        logger.info(f"[{worker_id}] Staggered start, waiting {args.startup_delay}s...")
        time.sleep(args.startup_delay)

    _register(worker_id, os.getpid())

    logger.info(f"[{worker_id}] Loading TTS model from '{args.model_dir}'...")
    from src.tts import TTS
    tts = TTS(model_dir=args.model_dir)
    logger.success(f"[{worker_id}] Model ready, entering job loop")

    _set_idle(worker_id)

    # Signal handlers for clean shutdown
    def _shutdown(sig, frame):
        logger.info(f"[{worker_id}] Shutting down")
        _set_offline(worker_id)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    MAX_CONSECUTIVE_ERRORS = 5
    consecutive_errors = 0
    last_hb = 0

    while True:
        try:
            now = time.time()
            if now - last_hb > 5:
                _heartbeat(worker_id)
                last_hb = now

            job = _claim_job(worker_id)
            if job:
                logger.info(f"[{worker_id}] ▶ {job['id'][:8]}")
                try:
                    output_path = _process(tts, job, worker_id)
                    _finish_job(worker_id, job["id"], output_path=output_path)
                    logger.success(f"[{worker_id}] ✓ {job['id'][:8]}")
                    consecutive_errors = 0
                except Exception as e:
                    logger.error(f"[{worker_id}] ✗ {job['id'][:8]}: {e}")
                    _finish_job(worker_id, job["id"], error=str(e))
                    consecutive_errors += 1
            else:
                time.sleep(0.5)
                continue

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"[{worker_id}] {consecutive_errors} consecutive job failures, "
                    f"exiting to release GPU memory (supervisor will restart)"
                )
                _set_offline(worker_id)
                sys.exit(1)

        except KeyboardInterrupt:
            _shutdown(None, None)
        except Exception as e:
            logger.error(f"[{worker_id}] loop error: {e}")
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(f"[{worker_id}] too many loop errors, exiting")
                _set_offline(worker_id)
                sys.exit(1)
            time.sleep(2)


if __name__ == "__main__":
    main()
