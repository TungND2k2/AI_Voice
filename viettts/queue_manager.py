import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from loguru import logger

OUTPUT_DIR = Path("/tmp/tts_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
QUEUE_DB = str(OUTPUT_DIR / "queue.db")


def _connect(timeout=30):
    conn = sqlite3.connect(QUEUE_DB, check_same_thread=False, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


class QueueManager:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = _connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'pending',
                text TEXT NOT NULL,
                voice TEXT,
                voice_path TEXT,
                speed REAL DEFAULT 1.0,
                response_format TEXT DEFAULT 'mp3',
                worker_id TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                output_path TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY,
                pid INTEGER,
                status TEXT DEFAULT 'loading',
                current_job_id TEXT,
                started_at TEXT,
                heartbeat_at TEXT
            );
        """)
        # Migration: add worker_id column if missing
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN worker_id TEXT")
            conn.commit()
        except Exception:
            pass
        conn.close()

    def add_job(self, text, voice, voice_path, speed=1.0, response_format="mp3"):
        job_id = str(uuid.uuid4())
        conn = _connect()
        conn.execute("""
            INSERT INTO jobs (id, text, voice, voice_path, speed, response_format, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (job_id, text, voice, voice_path, speed, response_format,
              datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return job_id

    def get_job(self, job_id):
        conn = _connect()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_jobs(self, limit=100):
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def stats(self):
        conn = _connect()
        result = {}
        for s in ["pending", "running", "completed", "failed"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (s,)
            ).fetchone()[0]
            result[s] = count
        conn.close()
        return result

    def get_position(self, job_id):
        conn = _connect()
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        conn.close()
        for i, r in enumerate(rows, 1):
            if r["id"] == job_id:
                return i
        return None

    def clear_workers(self):
        """Remove all worker records on server startup (stale from previous run)."""
        conn = _connect()
        conn.execute("DELETE FROM workers")
        conn.execute("""
            UPDATE jobs SET status='pending', worker_id=NULL, started_at=NULL
            WHERE status='running'
        """)
        conn.commit()
        conn.close()

    def list_workers(self):
        conn = _connect()
        rows = conn.execute("SELECT * FROM workers ORDER BY id ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_worker(self, worker_id):
        """Remove worker record and requeue its jobs."""
        conn = _connect()
        conn.execute("""
            UPDATE jobs SET status='pending', worker_id=NULL, started_at=NULL
            WHERE worker_id=? AND status='running'
        """, (worker_id,))
        conn.execute("DELETE FROM workers WHERE id=?", (worker_id,))
        conn.commit()
        conn.close()

    def reset_worker_jobs(self, worker_id):
        """Requeue running jobs from a dead worker back to pending."""
        conn = _connect()
        conn.execute("""
            UPDATE jobs SET status='pending', worker_id=NULL, started_at=NULL
            WHERE worker_id=? AND status='running'
        """, (worker_id,))
        conn.execute(
            "UPDATE workers SET status='offline', current_job_id=NULL WHERE id=?",
            (worker_id,)
        )
        conn.commit()
        conn.close()

    def storage_info(self):
        """Return disk + VRAM usage stats."""
        import shutil as _shutil
        import subprocess as _sp
        total, used, free = _shutil.disk_usage(str(OUTPUT_DIR))
        files = list(OUTPUT_DIR.glob("*.mp3")) + list(OUTPUT_DIR.glob("*.wav")) + \
                list(OUTPUT_DIR.glob("*.flac")) + list(OUTPUT_DIR.glob("*.opus")) + \
                list(OUTPUT_DIR.glob("*.aac"))
        audio_size = sum(f.stat().st_size for f in files if f.exists())

        # VRAM via nvidia-smi
        vram_used_mb = vram_total_mb = 0
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=_sp.DEVNULL, timeout=3
            ).decode().strip()
            nums = [int(x.strip()) for x in out.split(",")]
            vram_used_mb, vram_total_mb = nums[0], nums[1]
        except Exception:
            pass

        return {
            "disk_total_gb": round(total / 1e9, 1),
            "disk_used_gb":  round(used  / 1e9, 1),
            "disk_free_gb":  round(free  / 1e9, 1),
            "disk_pct":      round(used / total * 100, 1),
            "audio_files":   len(files),
            "audio_size_mb": round(audio_size / 1e6, 1),
            "output_dir":    str(OUTPUT_DIR),
            "vram_used_mb":  vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "vram_pct":      round(vram_used_mb / vram_total_mb * 100, 1) if vram_total_mb else 0,
        }

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Delete audio files and DB records for jobs older than max_age_hours."""
        import glob as _glob
        cutoff = datetime.utcnow().timestamp() - max_age_hours * 3600
        conn = _connect()
        rows = conn.execute("""
            SELECT id, output_path FROM jobs
            WHERE status IN ('completed','failed')
              AND completed_at IS NOT NULL
              AND datetime(completed_at) < datetime('now', ? || ' hours')
        """, (f"-{max_age_hours}",)).fetchall()

        deleted_files = 0
        freed_bytes   = 0
        for row in rows:
            path = row["output_path"]
            if path and os.path.exists(path):
                try:
                    freed_bytes += os.path.getsize(path)
                    os.unlink(path)
                    deleted_files += 1
                except Exception:
                    pass
            conn.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))

        # Also clean up Gradio /tmp/tmp*.wav files older than 1 hour
        import glob as _glob
        import time as _time
        now = _time.time()
        for f in _glob.glob("/tmp/tmp*.wav") + _glob.glob("/tmp/tmp*.mp3"):
            try:
                if now - os.path.getmtime(f) > 3600:
                    os.unlink(f)
            except Exception:
                pass

        conn.commit()
        conn.close()
        if deleted_files:
            logger.info(f"[Cleanup] Removed {deleted_files} files, freed {freed_bytes/1e6:.1f} MB")
        return {"deleted_files": deleted_files, "freed_mb": round(freed_bytes/1e6, 1)}

    def cleanup_stale_jobs(self):
        """Reset running jobs from workers with stale heartbeats."""
        conn = _connect()
        rows = conn.execute(
            "SELECT id, current_job_id, heartbeat_at FROM workers WHERE status='busy'"
        ).fetchall()
        now = datetime.utcnow().timestamp()
        for w in rows:
            if not w["heartbeat_at"]:
                continue
            try:
                hb_ts = datetime.fromisoformat(w["heartbeat_at"]).timestamp()
                if now - hb_ts > 600:
                    logger.warning(f"[Queue] Worker {w['id']} stale, requeueing job")
                    if w["current_job_id"]:
                        conn.execute("""
                            UPDATE jobs SET status='pending', worker_id=NULL, started_at=NULL
                            WHERE id=? AND status='running'
                        """, (w["current_job_id"],))
                    conn.execute(
                        "UPDATE workers SET status='offline', current_job_id=NULL WHERE id=?",
                        (w["id"],)
                    )
            except Exception as e:
                logger.error(f"cleanup_stale_jobs error: {e}")
        conn.commit()
        conn.close()
