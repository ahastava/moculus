"""
MoCoLUS Web Server — FastAPI + WebSocket Replacement for noVNC
==============================================================
Serves a native web GUI that replaces the PyQt6 + noVNC pipeline.
Real-time B-mode/M-mode frames stream over WebSocket as base64 JPEG.

Usage:
    python -m uvicorn src.web_server:app --host 0.0.0.0 --port 8000
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import asyncio
import base64
import io
import json
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .poc_image_stack import (
    SCENARIOS,
    LungZone,
    PatientCase,
    POCImageStackGenerator,
    ProbeReading,
    StackConfig,
    generate_patient_case,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_root = Path(__file__).resolve().parent.parent
app = FastAPI(title="MoCoLUS POCUS Trainer")
app.mount("/static", StaticFiles(directory=str(_root / "static")), name="static")

# Thread pool for CPU-bound generation (training, dataset)
_executor = ThreadPoolExecutor(max_workers=2)

# Active training/dataset jobs
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIAGNOSIS_OPTIONS = [
    "Normal lung exam",
    "Pneumothorax (left)",
    "Pneumothorax (right)",
    "Pneumothorax with lung point",
    "Pulmonary edema (cardiogenic)",
    "ARDS / white lung",
    "Pneumonia (left)",
    "Pneumonia (right)",
    "Bilateral pneumonia / viral",
    "Pleural effusion (right)",
    "Pleural effusion (bilateral)",
    "Hemothorax (trauma)",
    "Pneumonia with effusion",
    "COPD / asthma exacerbation",
    "CHF with bilateral effusions",
]


def encode_frame_jpeg(frame: np.ndarray, quality: int = 92) -> str:
    """Encode [H,W] float32 in [0,1] → base64 JPEG string."""
    gray = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fuzzy_diagnosis_match(selected: str, correct: str) -> bool:
    """Match diagnosis with fuzzy term overlap (mirrors PyQt6 GUI logic)."""
    if selected.lower().strip() == correct.lower().strip():
        return True
    sel_terms = set(selected.lower().replace("(", "").replace(")", "").split())
    cor_terms = set(correct.lower().replace("(", "").replace(")", "").split())
    overlap = sel_terms & cor_terms
    return len(overlap) >= len(cor_terms) * 0.5


# ---------------------------------------------------------------------------
# WebSocket Simulator Session
# ---------------------------------------------------------------------------

_FRAME_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "frame_cache.npz"
_frame_cache = None

def _load_frame_cache():
    """Load pre-generated frame cache if available (GPU-rendered, CPU-fast)."""
    global _frame_cache
    if _frame_cache is not None:
        return _frame_cache
    if _FRAME_CACHE_PATH.exists():
        try:
            _frame_cache = dict(np.load(str(_FRAME_CACHE_PATH), allow_pickle=True))
            logger.info(f"Loaded pre-generated frame cache ({len(_frame_cache)} entries)")
            return _frame_cache
        except Exception as e:
            logger.warning(f"Could not load frame cache: {e}")
    return None


class WebSimSession:
    """Per-client simulator session state."""

    def __init__(self):
        self.patient_case: Optional[PatientCase] = None
        self.exam_cache: Dict[str, Dict] = {}
        self.active_zone: Optional[str] = None
        self.frame_idx: int = 0
        self.playing: bool = False
        self.frozen: bool = False
        self.testing_mode: bool = False
        self.examined_zones: set = set()
        self.answer_revealed: bool = False
        self.imu_yaw: float = 0.0
        self.imu_pitch: float = 0.0
        self.imu_roll: float = 0.0
        self._gen: Optional[POCImageStackGenerator] = None
        # Free probe positioning
        self.probe_nx: float = 0.5
        self.probe_ny: float = 0.25
        self.zone_positions: Dict[str, Dict[str, float]] = {}
        self._interp_cache_key: Optional[str] = None
        self._interp_encoded: Optional[list] = None

    def load_case(self, scenario_key: Optional[str] = None):
        """Load a patient case. Uses pre-generated cache if available,
        otherwise generates frames live (slower, but works without GPU)."""
        self.patient_case = generate_patient_case(scenario_key=scenario_key)
        pc = self.patient_case

        self._gen = POCImageStackGenerator(
            scenario=pc.scenario_key,
            stack_config=StackConfig(n_frames=32, image_size=(256, 256)),
        )
        self.exam_cache.clear()
        self.examined_zones.clear()
        self.active_zone = None
        self.frame_idx = 0
        self.playing = False
        self.frozen = False
        self.answer_revealed = False
        self._interp_cache_key = None
        self._interp_encoded = None

        # Try pre-generated cache first (instant load, GPU-quality)
        cache = _load_frame_cache()
        if cache and self._load_from_cache(cache, pc.scenario_key):
            logger.info(f"Loaded case '{pc.scenario_key}' from pre-generated cache")
            return

        # Fallback: generate live (CPU, slower, no diffusion)
        logger.info(f"Generating case '{pc.scenario_key}' live (no cache)")
        rng = np.random.default_rng(pc.seed)
        for zone in LungZone:
            zone_seed = int(rng.integers(0, 2**31))
            anchor = self._gen.zone_resolver.get_anchor(zone)
            probe = ProbeReading(x_m=anchor.x_m, y_m=anchor.y_m, pressure=1.0)
            result = self._gen.generate(probe, seed=zone_seed)
            self.exam_cache[zone.name] = result

    def _load_from_cache(self, cache: dict, scenario_key: str) -> bool:
        """Load all 8 zones from pre-generated cache for a scenario."""
        for zone in LungZone:
            key = f"{scenario_key}/{zone.name}"
            bmode_key = f"{key}/bmode"
            if bmode_key not in cache:
                return False  # Scenario not in cache
            bmode = cache[bmode_key].astype(np.float32)
            mmode = cache[f"{key}/mmode"].astype(np.float32)
            pathology = str(cache[f"{key}/pathology"])
            pathology_class = int(cache[f"{key}/pathology_class"])
            sliding = bool(cache[f"{key}/sliding"])
            mmode_pattern = str(cache[f"{key}/mmode_pattern"])

            self.exam_cache[zone.name] = {
                "bmode_stack": bmode,
                "mmode": mmode,
                "pathology": pathology,
                "pathology_class": pathology_class,
                "lung_sliding": sliding,
                "mmode_pattern": mmode_pattern,
            }
        return True

    def get_case_info(self) -> Dict:
        """Return patient info + scenario list for the client."""
        pc = self.patient_case
        if not pc:
            return {}
        patient = {
            "age": pc.age,
            "sex": pc.sex,
            "chief_complaint": pc.chief_complaint,
            "history": pc.history,
            "vitals": pc.vitals,
            "vitals_text": pc.vitals_text,
        }
        # Zone positions (normalized 0-1 for SVG rendering)
        zones = {}
        for zone in LungZone:
            anchor = self._gen.zone_resolver.get_anchor(zone)
            # Normalize: x in [-0.20, 0.20] → [0, 1], y in [0, 0.30] → [0, 1]
            nx = (anchor.x_m + 0.20) / 0.40
            ny = anchor.y_m / 0.30
            zones[zone.name] = {"nx": round(nx, 3), "ny": round(ny, 3)}
        self.zone_positions = zones
        return {
            "patient": patient,
            "scenario_key": pc.scenario_key if not self.testing_mode else None,
            "scenario_name": pc.scenario.name if not self.testing_mode else "Hidden",
            "scenario_description": pc.scenario.description if not self.testing_mode else "Examine all zones and submit your diagnosis.",
            "zones": zones,
            "diagnosis_options": DIAGNOSIS_OPTIONS,
            "n_frames": 32,
        }

    def get_frame_data(self) -> Optional[Dict]:
        """Get current frame as base64 JPEG + metadata.
        Caches encoded JPEGs to avoid re-encoding on every playback frame."""
        if not self.active_zone or self.active_zone not in self.exam_cache:
            return None
        result = self.exam_cache[self.active_zone]
        n_frames = result["bmode_stack"].shape[0]
        idx = max(0, min(self.frame_idx, n_frames - 1))

        # Lazy-cache encoded frames
        cache_key = f"{self.active_zone}_encoded"
        if cache_key not in self.exam_cache:
            bmode_stack = result["bmode_stack"]
            self.exam_cache[cache_key] = {
                "bmode": [encode_frame_jpeg(bmode_stack[i]) for i in range(n_frames)],
                "mmode": encode_frame_jpeg(result["mmode"]),
            }
        encoded = self.exam_cache[cache_key]

        return {
            "type": "frame",
            "bmode": encoded["bmode"][idx],
            "mmode": encoded["mmode"],
            "zone": self.active_zone,
            "pathology": result["pathology"],
            "pathology_class": int(result["pathology_class"]),
            "sliding": bool(result["lung_sliding"]),
            "mmode_pattern": result["mmode_pattern"],
            "frame_idx": idx,
            "n_frames": n_frames,
            "frozen": self.frozen,
            "examined_zones": list(self.examined_zones),
        }

    def get_interpolated_frame_data(self, nx: float, ny: float) -> Optional[Dict]:
        """Get distance-weighted blended frame based on free probe position.

        Blends B-mode frames from nearest zones using a Gaussian kernel,
        so anatomy transitions smoothly as the probe sweeps across the torso.
        """
        if not self.zone_positions or not self.exam_cache:
            return None

        # Calculate distances to all zone anchors
        sigma = 0.14  # Gaussian falloff (normalized coords)
        weights: Dict[str, float] = {}
        for zone_key, pos in self.zone_positions.items():
            if zone_key not in self.exam_cache:
                continue
            dist = math.sqrt((nx - pos["nx"]) ** 2 + (ny - pos["ny"]) ** 2)
            w = math.exp(-(dist ** 2) / (2 * sigma ** 2))
            if w > 0.005:  # skip negligible contributions
                weights[zone_key] = w

        if not weights:
            return None

        # Normalize weights
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

        # Find dominant zone (highest weight) for metadata
        dominant_zone = max(weights, key=weights.get)
        dominant_result = self.exam_cache[dominant_zone]
        n_frames = dominant_result["bmode_stack"].shape[0]
        idx = max(0, min(self.frame_idx, n_frames - 1))

        # Mark zones near the probe as examined
        for zone_key, w in weights.items():
            if w > 0.3:
                self.examined_zones.add(zone_key)

        # Check if we can reuse cached interpolated encoding
        cache_key = f"interp_{nx:.3f}_{ny:.3f}"
        if self._interp_cache_key == cache_key and self._interp_encoded:
            encoded_bmode = self._interp_encoded
        else:
            # Blend B-mode stacks from contributing zones
            blended_stack = np.zeros_like(dominant_result["bmode_stack"])
            for zone_key, w in weights.items():
                blended_stack += w * self.exam_cache[zone_key]["bmode_stack"].astype(np.float32)

            # Encode all blended frames
            encoded_bmode = [encode_frame_jpeg(blended_stack[i]) for i in range(n_frames)]
            self._interp_cache_key = cache_key
            self._interp_encoded = encoded_bmode

        # M-mode: use dominant zone's (blending M-mode strips is confusing)
        dom_enc_key = f"{dominant_zone}_encoded"
        if dom_enc_key not in self.exam_cache:
            self.exam_cache[dom_enc_key] = {
                "bmode": [encode_frame_jpeg(dominant_result["bmode_stack"][i]) for i in range(n_frames)],
                "mmode": encode_frame_jpeg(dominant_result["mmode"]),
            }
        mmode_encoded = self.exam_cache[dom_enc_key]["mmode"]

        self.active_zone = dominant_zone
        return {
            "type": "frame",
            "bmode": encoded_bmode[idx],
            "mmode": mmode_encoded,
            "zone": dominant_zone,
            "pathology": dominant_result["pathology"],
            "pathology_class": int(dominant_result["pathology_class"]),
            "sliding": bool(dominant_result["lung_sliding"]),
            "mmode_pattern": dominant_result["mmode_pattern"],
            "frame_idx": idx,
            "n_frames": n_frames,
            "frozen": self.frozen,
            "examined_zones": list(self.examined_zones),
            "interpolated": True,
            "weights": {k: round(v, 3) for k, v in sorted(weights.items(), key=lambda x: -x[1])[:3]},
        }

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/simulator")
async def simulator_ws(websocket: WebSocket):
    await websocket.accept()
    session = WebSimSession()

    # Load initial case
    session.load_case(scenario_key="normal")
    await websocket.send_json({
        "type": "case_loaded",
        **session.get_case_info(),
    })

    playback_task = None

    async def playback_loop():
        """Server-driven playback at ~30fps, using interpolated or zone frames."""
        while session.playing and not session.frozen:
            n_frames = 32
            if session.active_zone:
                n_frames = session.exam_cache.get(session.active_zone, {}).get(
                    "bmode_stack", np.zeros((1,))
                ).shape[0]
            session.frame_idx = (session.frame_idx + 1) % n_frames
            data = session.get_interpolated_frame_data(session.probe_nx, session.probe_ny)
            if not data:
                data = session.get_frame_data()
            if data:
                try:
                    await websocket.send_json(data)
                except Exception:
                    return
            await asyncio.sleep(0.033)  # ~30fps

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "select_zone":
                zone_key = msg.get("zone", "")
                if zone_key in session.exam_cache:
                    session.active_zone = zone_key
                    session.examined_zones.add(zone_key)
                    session.frame_idx = 0
                    # Update probe position to zone anchor
                    if zone_key in session.zone_positions:
                        session.probe_nx = session.zone_positions[zone_key]["nx"]
                        session.probe_ny = session.zone_positions[zone_key]["ny"]
                    data = session.get_frame_data()
                    if data:
                        await websocket.send_json(data)

            elif msg_type == "probe_position":
                nx = float(msg.get("nx", 0.5))
                ny = float(msg.get("ny", 0.25))
                session.probe_nx = nx
                session.probe_ny = ny
                if not session.playing:
                    data = session.get_interpolated_frame_data(nx, ny)
                    if data:
                        await websocket.send_json(data)

            elif msg_type == "set_frame":
                session.frame_idx = int(msg.get("frame", 0))
                if not session.playing:
                    data = session.get_interpolated_frame_data(
                        session.probe_nx, session.probe_ny
                    )
                    if not data:
                        data = session.get_frame_data()
                    if data:
                        await websocket.send_json(data)

            elif msg_type == "play":
                if not session.playing:
                    session.playing = True
                    playback_task = asyncio.create_task(playback_loop())

            elif msg_type == "pause":
                session.playing = False
                if playback_task:
                    playback_task.cancel()
                    playback_task = None

            elif msg_type == "freeze":
                session.frozen = True
                session.playing = False
                if playback_task:
                    playback_task.cancel()
                    playback_task = None
                data = session.get_frame_data()
                if data:
                    await websocket.send_json(data)

            elif msg_type == "unfreeze":
                session.frozen = False
                data = session.get_frame_data()
                if data:
                    await websocket.send_json(data)

            elif msg_type == "set_scenario":
                scenario = msg.get("scenario", "normal")
                session.testing_mode = False
                session.load_case(scenario_key=scenario)
                await websocket.send_json({
                    "type": "case_loaded",
                    **session.get_case_info(),
                })

            elif msg_type == "new_case":
                mode = msg.get("mode", "practice")
                session.testing_mode = (mode == "test")
                scenario = msg.get("scenario") if mode == "practice" else None
                session.load_case(scenario_key=scenario)
                await websocket.send_json({
                    "type": "case_loaded",
                    **session.get_case_info(),
                })

            elif msg_type == "set_mode":
                mode = msg.get("mode", "practice")
                session.testing_mode = (mode == "test")
                scenario = None if session.testing_mode else msg.get("scenario", "normal")
                session.load_case(scenario_key=scenario)
                await websocket.send_json({
                    "type": "case_loaded",
                    **session.get_case_info(),
                })

            elif msg_type == "submit_diagnosis":
                diagnosis = msg.get("diagnosis", "")
                if session.patient_case:
                    correct = session.patient_case.diagnosis
                    is_correct = fuzzy_diagnosis_match(diagnosis, correct)
                    session.answer_revealed = True
                    await websocket.send_json({
                        "type": "diagnosis_result",
                        "correct": is_correct,
                        "selected": diagnosis,
                        "answer": correct,
                        "explanation": session.patient_case.diagnosis_explanation,
                        "blue_profile": session.patient_case.blue_profile,
                        "scenario_name": session.patient_case.scenario.name,
                    })

            elif msg_type == "reveal_answer":
                if session.patient_case:
                    session.answer_revealed = True
                    await websocket.send_json({
                        "type": "diagnosis_result",
                        "correct": None,
                        "selected": None,
                        "answer": session.patient_case.diagnosis,
                        "explanation": session.patient_case.diagnosis_explanation,
                        "blue_profile": session.patient_case.blue_profile,
                        "scenario_name": session.patient_case.scenario.name,
                    })

            elif msg_type == "imu_update":
                session.imu_yaw = float(msg.get("yaw", 0))
                session.imu_pitch = float(msg.get("pitch", 0))
                session.imu_roll = float(msg.get("roll", 0))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        session.playing = False
        if playback_task:
            playback_task.cancel()


# ---------------------------------------------------------------------------
# REST API — Scenarios
# ---------------------------------------------------------------------------

@app.get("/api/scenarios")
async def list_scenarios():
    return {
        key: {"name": s.name, "description": s.description}
        for key, s in SCENARIOS.items()
    }


@app.get("/api/system-info")
async def system_info():
    """Report system capabilities — used by frontend to gate Training Studio."""
    import socket
    gpu_available = False
    gpu_name = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "hostname": socket.gethostname(),
    }


# ---------------------------------------------------------------------------
# REST API — Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    epochs: int = 100
    batch_size: int = 4
    n_per_class: int = 200
    image_size: int = 256
    from_scratch: bool = False
    resume_path: Optional[str] = None


@app.post("/api/train")
async def start_training(req: TrainRequest):
    async with _jobs_lock:
        active = [j for j in _jobs.values() if j["status"] == "running" and j["type"] == "train"]
        if active:
            return JSONResponse(
                status_code=409,
                content={"error": "A training job is already running", "job_id": active[0]["id"]},
            )

    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id,
        "type": "train",
        "status": "running",
        "progress": [],
        "params": req.model_dump(),
        "cancelled": False,
    }

    async with _jobs_lock:
        _jobs[job_id] = job

    def _run_training():
        try:
            from .train_zea_diffusion import ZeaDiffusionTrainer
            trainer = ZeaDiffusionTrainer(
                image_size=req.image_size,
                n_per_class=req.n_per_class,
                batch_size=req.batch_size,
                n_epochs=req.epochs,
                use_pretrained=not req.from_scratch,
                resume_path=req.resume_path,
            )
            trainer.train(progress_callback=lambda info: job["progress"].append(info))
            job["status"] = "completed"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"Training job {job_id} failed: {e}")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_training)

    return {"job_id": job_id, "status": "running"}


@app.get("/api/train/{job_id}/status")
async def train_status(job_id: str):
    """SSE stream of training progress."""
    if job_id not in _jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    async def event_stream():
        seen = 0
        while True:
            job = _jobs.get(job_id)
            if not job:
                break
            while seen < len(job["progress"]):
                data = json.dumps(job["progress"][seen])
                yield f"data: {data}\n\n"
                seen += 1
            if job["status"] in ("completed", "failed", "cancelled"):
                yield f"data: {json.dumps({'status': job['status'], 'error': job.get('error')})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/train/{job_id}/cancel")
async def cancel_training(job_id: str):
    if job_id in _jobs:
        _jobs[job_id]["cancelled"] = True
        _jobs[job_id]["status"] = "cancelled"
        return {"status": "cancelled"}
    return JSONResponse(status_code=404, content={"error": "Job not found"})


# ---------------------------------------------------------------------------
# REST API — Dataset Generation
# ---------------------------------------------------------------------------

class DatasetRequest(BaseModel):
    n_per_class: int = 500
    output_path: str = "data/lung_us_moculus.h5"
    vehicle_types: Optional[list] = None
    seed: int = 42


@app.post("/api/dataset")
async def start_dataset(req: DatasetRequest):
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id,
        "type": "dataset",
        "status": "running",
        "progress": [],
        "params": req.model_dump(),
    }

    async with _jobs_lock:
        _jobs[job_id] = job

    def _run_dataset():
        try:
            from .lung_us_dataset import LungUSDatasetBuilder, DatasetSpec
            spec = DatasetSpec(
                n_samples_per_class=req.n_per_class,
                output_path=req.output_path,
            )
            builder = LungUSDatasetBuilder(spec)
            paths = builder.build_splits(
                vehicle_types=req.vehicle_types,
                seed=req.seed,
                progress_callback=lambda info: job["progress"].append(info),
            )
            job["status"] = "completed"
            job["result"] = paths
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"Dataset job {job_id} failed: {e}")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_dataset)

    return {"job_id": job_id, "status": "running"}


@app.get("/api/dataset/{job_id}/status")
async def dataset_status(job_id: str):
    if job_id not in _jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    async def event_stream():
        seen = 0
        while True:
            job = _jobs.get(job_id)
            if not job:
                break
            while seen < len(job["progress"]):
                data = json.dumps(job["progress"][seen])
                yield f"data: {data}\n\n"
                seen += 1
            if job["status"] in ("completed", "failed"):
                yield f"data: {json.dumps({'status': job['status'], 'result': job.get('result'), 'error': job.get('error')})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# REST API — Preview Generation
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    type: str = "grid"  # "grid", "scenario", "bmode_mmode"
    pathology: Optional[str] = None
    scenario: Optional[str] = None


@app.post("/api/preview")
async def generate_preview(req: PreviewRequest):
    def _render():
        from .training_preview import TrainingPreviewRenderer
        import matplotlib
        matplotlib.use("Agg")
        renderer = TrainingPreviewRenderer()
        out_dir = _root / "data" / "training_previews"
        out_dir.mkdir(parents=True, exist_ok=True)

        if req.type == "grid":
            path = out_dir / "pathology_grid.png"
            renderer.render_comparison_grid(str(path))
        elif req.type == "scenario" and req.scenario:
            path = out_dir / f"scenario_{req.scenario}.png"
            renderer.render_scenario(req.scenario, str(path))
        elif req.type == "bmode_mmode":
            path = out_dir / "bmode_mmode_grid.png"
            renderer.render_bmode_mmode_grid(str(path))
        else:
            path = out_dir / "pathology_grid.png"
            renderer.render_comparison_grid(str(path))
        return str(path)

    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(_executor, _render)
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# REST API — Checkpoints
# ---------------------------------------------------------------------------

@app.get("/api/checkpoints")
async def list_checkpoints():
    ckpt_dir = _root / "checkpoints" / "zea_lung_pocus"
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "checkpoints": [
            {"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1), "path": str(f)}
            for f in files
        ]
    }


# ---------------------------------------------------------------------------
# REST API — Jobs list
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
async def list_jobs():
    return {jid: {"type": j["type"], "status": j["status"]} for jid, j in _jobs.items()}


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(_root / "static" / "index.html"))
