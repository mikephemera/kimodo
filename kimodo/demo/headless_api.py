# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless HTTP API for kimodo_demo.

Provides a FastAPI server that shares the Demo instance with the Viser
frontend, enabling programmatic motion generation while the 3D viewer
stays open for monitoring.

Two generation paths:
  - **Hybrid**: when ``client_id`` is specified and that Viser client is
    connected, the motion is generated via ``Demo.generate()`` and appears
    in the 3D view automatically.
  - **Headless**: when no ``client_id`` is given (or the client is not
    connected), the model is called directly and only the raw motion data
    is returned.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from kimodo.constraints import load_constraints_lst
from kimodo.exports.motion_io import kimodo_npz_to_bytes
from kimodo.model.cfg import CFG_TYPES
from kimodo.tools import seed_everything

# Viser PROMPT_COLORS for timeline sync (hybrid mode)
try:
    from viser._timeline_api import PROMPT_COLORS
except ImportError:
    PROMPT_COLORS = [(90, 160, 255)]  # fallback blue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

CFG_TYPE_VALUES = list(CFG_TYPES)
OUTPUT_FORMATS = ("npz", "json")
AUTO_SAVE_FORMATS = ("NPZ", "CSV", "BVH")


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""

    prompt: Union[str, List[str]] = Field(
        ...,
        description=(
            "Text prompt(s) describing the desired motion. "
            "A single string generates one motion; a list of strings with "
            "multi_prompt=true treats them as sequential segments."
        ),
        examples=["a person walks forward"],
    )
    duration: Union[float, List[float]] = Field(
        default=5.0,
        description=(
            "Duration in seconds. A single float applies to the whole motion; "
            "a list of floats maps 1:1 to prompt segments (multi_prompt mode)."
        ),
        examples=[5.0],
    )
    num_samples: int = Field(default=1, ge=1, le=64, description="Number of motion samples to generate.")
    diffusion_steps: int = Field(default=100, ge=10, le=1000, description="Number of DDIM denoising steps.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    multi_prompt: bool = Field(
        default=False,
        description="If true, treat prompt list as sequential segments stitched together.",
    )
    cfg_type: str = Field(
        default="separated",
        description="Classifier-free guidance mode: nocfg, regular, or separated.",
    )
    cfg_weight: List[float] = Field(
        default=[2.0, 2.0],
        description="CFG scale(s). One float for regular, two for separated [text, constraint].",
    )
    constraints: Optional[List[dict]] = Field(
        default=None,
        description="Optional list of constraint dicts (same format as constraints.json).",
    )
    post_processing: bool = Field(
        default=True,
        description="Apply foot-skate cleanup and constraint enforcement after generation.",
    )
    num_transition_frames: int = Field(
        default=5, ge=0, le=60, description="Overlapping frames for multi-prompt transitions."
    )
    first_heading_angle: float = Field(default=0.0, description="Initial body heading in radians (0 = facing +Z).")
    root_margin: float = Field(
        default=0.04, ge=0.0, description="Horizontal margin (m) for post-process root correction."
    )
    client_id: Optional[int] = Field(
        default=None,
        description=(
            "Viser client ID to render on. When provided and the client is "
            "connected, the motion appears in the 3D viewer (hybrid mode). "
            "When absent, raw data is returned without 3D rendering (headless mode)."
        ),
    )
    output_format: str = Field(
        default="npz",
        description="Response format: 'npz' (binary) or 'json'.",
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save generated motion to auto_save_dir.",
    )
    auto_save_dir: str = Field(
        default="/mnt/datafiles/Work-syncfree/unitree_sim2x/assets/motions/g1_29dof/kimodo_autosave",
        description="Directory for auto-saved motion files.",
    )
    auto_save_format: str = Field(
        default="CSV",
        description="Auto-save file format: 'NPZ', 'CSV', or 'BVH'.",
    )

    @field_validator("auto_save_format")
    @classmethod
    def _check_auto_save_format(cls, v: str) -> str:
        if v.upper() not in AUTO_SAVE_FORMATS:
            raise ValueError(f"auto_save_format must be one of {AUTO_SAVE_FORMATS}, got {v!r}")
        return v.upper()

    @field_validator("cfg_type")
    @classmethod
    def _check_cfg_type(cls, v: str) -> str:
        if v not in CFG_TYPE_VALUES:
            raise ValueError(f"cfg_type must be one of {CFG_TYPE_VALUES}, got {v!r}")
        return v

    @field_validator("output_format")
    @classmethod
    def _check_output_format(cls, v: str) -> str:
        if v not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {v!r}")
        return v

    @field_validator("cfg_weight")
    @classmethod
    def _check_cfg_weight(cls, v: List[float], info) -> List[float]:
        if "cfg_type" in info.data:
            cfg_type = info.data["cfg_type"]
            if cfg_type == "nocfg":
                return v  # ignored
            if cfg_type == "regular" and len(v) != 1:
                raise ValueError("cfg_type='regular' requires exactly one cfg_weight value")
            if cfg_type == "separated" and len(v) != 2:
                raise ValueError("cfg_type='separated' requires exactly two cfg_weight values")
        return v

    @field_validator("duration")
    @classmethod
    def _check_duration_positive(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        values = v if isinstance(v, list) else [v]
        for d in values:
            if d <= 0:
                raise ValueError(f"duration values must be positive, got {d}")
        return v


class GenerateResponse(BaseModel):
    """Metadata returned alongside the motion data."""

    format: str = Field(description="Output format used ('npz' or 'json').")
    num_samples: int
    total_frames: int
    fps: float


class ClientInfo(BaseModel):
    """Summary of a connected Viser client."""

    client_id: int
    model_name: str
    frame_idx: int
    playing: bool


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    cuda_healthy: bool
    loaded_models: list[str]
    connected_clients: int
    headless_api: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_durations(req: GenerateRequest, fps: float) -> list[float]:
    """Normalise duration to a per-segment list."""
    if isinstance(req.duration, list):
        return list(req.duration)
    return [req.duration]


def _resolve_prompts(req: GenerateRequest) -> list[str]:
    """Normalise prompts to a list."""
    if isinstance(req.prompt, str):
        return [req.prompt]
    return list(req.prompt)


def _compute_num_frames(durations_sec: list[float], fps: float) -> list[int]:
    """Convert durations in seconds to frame counts."""
    return [int(d * fps) for d in durations_sec]


def _build_constraint_lst(constraints: Optional[List[dict]], skeleton) -> list:
    """Build constraint objects from JSON-compatible dicts."""
    if not constraints:
        return []
    return load_constraints_lst(constraints, skeleton)


def _resolve_cfg_kwargs(req: GenerateRequest) -> dict:
    """Convert request CFG fields to model kwargs."""
    if req.cfg_type == "nocfg":
        return {"cfg_type": "nocfg"}
    return {"cfg_type": req.cfg_type, "cfg_weight": req.cfg_weight}


def _tensor_to_serializable(obj: Any) -> Any:
    """Recursively convert torch Tensors to lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tensor_to_serializable(v) for v in obj]
    return obj


def _extract_motion_from_output(output: dict, sample_idx: int = 0) -> dict:
    """Extract a single sample from the model output dict.

    The model may return batched tensors [B, T, ...] or single-sample
    tensors [T, ...].  We always return a single-sample dict.
    """
    single: dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if hasattr(value, "ndim") and value.ndim >= 3:
                # Batched: take sample_idx
                single[key] = value[sample_idx]
            else:
                single[key] = value
        else:
            single[key] = value
    return single


def _extract_motion_from_session(session, sample_idx: int = 0) -> dict:
    """Extract motion data from a Viser ClientSession.

    Returns a dict compatible with the model output format, including
    ``local_rot_mats`` and ``root_positions`` needed for CSV/BVH export.
    """
    motion_names = list(session.motions.keys())
    if not motion_names:
        raise RuntimeError("No motions found in session after generation.")
    # Return the first (or only) motion
    name = motion_names[min(sample_idx, len(motion_names) - 1)]
    motion = session.motions[name]
    root_idx = motion.skeleton.root_idx
    return {
        "posed_joints": motion.joints_pos,
        "global_rot_mats": motion.joints_rot,
        "local_rot_mats": motion.joints_local_rot,
        "root_positions": motion.joints_pos[:, root_idx, :],
        "foot_contacts": getattr(motion, "foot_contacts", None),
        "fps": session.model_fps,
    }


def _build_npz_response(output: dict, req: GenerateRequest, auto_saved_path: str | None = None) -> Response:
    """Build a binary NPZ response from model output."""
    # Extract first sample for the NPZ
    single = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            single[key] = value[0].detach().cpu().numpy() if value.ndim >= 3 else value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            single[key] = value[0] if value.ndim >= 3 else value
        else:
            single[key] = value

    npz_bytes = kimodo_npz_to_bytes(single)
    headers = {
        "Content-Disposition": 'attachment; filename="motion.npz"',
        "X-Num-Samples": str(
            output.get("posed_joints", np.empty((0,))).shape[0]
            if hasattr(output.get("posed_joints", np.empty((0,))), "shape")
            else 1
        ),
    }
    if auto_saved_path:
        headers["X-Auto-Saved-Path"] = auto_saved_path
    return Response(
        content=npz_bytes,
        media_type="application/octet-stream",
        headers=headers,
    )


def _build_json_response(output: dict) -> dict:
    """Build a JSON-compatible dict from model output."""
    serialized = _tensor_to_serializable(output)
    # If batched, include all samples
    return serialized


def _model_output_metadata(output: dict, fps: float) -> dict:
    """Extract metadata from model output for the response header/body."""
    posed = output.get("posed_joints")
    if posed is not None:
        if isinstance(posed, torch.Tensor):
            shape = posed.shape
        elif isinstance(posed, np.ndarray):
            shape = posed.shape
        else:
            shape = (1, 1)
    else:
        shape = (1, 1)

    num_samples = shape[0] if len(shape) >= 3 else 1
    total_frames = shape[1] if len(shape) >= 3 else (shape[0] if len(shape) >= 2 else 1)
    return {"num_samples": num_samples, "total_frames": total_frames, "fps": fps}


def _build_filename_slug(prompts: list[str]) -> str:
    """Build a filesystem-safe slug from the first prompt text."""
    if not prompts:
        return "motion"
    raw = prompts[0][:30].strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")
    return slug or "motion"


def _auto_save_motion(
    output: dict,
    prompts: list[str],
    save_dir: str,
    fmt: str,
    skeleton,
    device: str,
    fps: float,
) -> str | None:
    """Save generated motion to disk in the requested format.

    Args:
        output: Motion data dict (from model or session extraction).
        prompts: Text prompts used for filename slug.
        save_dir: Target directory (created if missing).
        fmt: Format string: ``"NPZ"``, ``"CSV"``, or ``"BVH"``.
        skeleton: Skeleton instance (needed for BVH/CSV export).
        device: Torch device for GPU-based conversion.
        fps: Frames per second for the output file.

    Returns:
        Path of the saved file, or ``None`` if format is unsupported.
    """
    os.makedirs(save_dir, exist_ok=True)

    slug = _build_filename_slug(prompts)
    timestamp = datetime.now().strftime("%H%M")
    ext_map = {"NPZ": ".npz", "CSV": ".csv", "BVH": ".bvh"}
    ext = ext_map.get(fmt, ".npz")
    base_name = f"{slug}_{timestamp}{ext}"
    save_path = os.path.join(save_dir, base_name)

    # Avoid overwriting existing files
    counter = 1
    while os.path.exists(save_path):
        stem = os.path.splitext(base_name)[0]
        save_path = os.path.join(save_dir, f"{stem}_{counter}{ext}")
        counter += 1

    if fmt == "NPZ":
        _save_npz(output, save_path)
    elif fmt == "CSV":
        _save_csv(output, skeleton, device, save_path)
    elif fmt == "BVH":
        _save_bvh(output, skeleton, fps, save_path)
    else:
        msg = f"Unsupported auto-save format: {fmt}"
        logger.warning(msg)
        print(f"[auto-save] WARNING: {msg}")
        return None

    logger.info("Auto-saved motion to %s", save_path)
    print(f"[auto-save] Saved motion to {save_path}")
    return save_path


def _save_npz(output: dict, save_path: str) -> None:
    """Save motion as NPZ from a dict of numpy arrays."""
    single: dict[str, np.ndarray] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            continue
        # Take first sample if batched [B, T, ...]
        if arr.ndim >= 3 and arr.shape[0] > 1:
            arr = arr[0]
        single[key] = arr
    npz_bytes = kimodo_npz_to_bytes(single)
    with open(save_path, "wb") as f:
        f.write(npz_bytes)


def _save_csv(output: dict, skeleton, device: str, save_path: str) -> None:
    """Save motion as G1 MuJoCo CSV."""
    from kimodo.exports.mujoco import MujocoQposConverter

    # Ensure tensors are on the right device
    data = {}
    for key, value in output.items():
        if isinstance(value, np.ndarray):
            data[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, torch.Tensor):
            data[key] = value.to(device)
        else:
            data[key] = value

    converter = MujocoQposConverter(skeleton)
    qpos = converter.dict_to_qpos(data, device)
    converter.save_csv(qpos, save_path)


def _save_bvh(output: dict, skeleton, fps: float, save_path: str) -> None:
    """Save motion as BVH."""
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.skeleton import global_rots_to_local_rots

    joints_pos = _to_tensor(output.get("posed_joints"))
    joints_rot = _to_tensor(output.get("global_rot_mats"))

    if joints_pos is None or joints_rot is None:
        logger.warning("Cannot save BVH: missing posed_joints or global_rot_mats")
        return

    # Take first sample if batched
    if joints_pos.ndim >= 4:
        joints_pos = joints_pos[0]
    if joints_rot.ndim >= 5:
        joints_rot = joints_rot[0]

    local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
    root_positions = joints_pos[:, skeleton.root_idx, :]
    save_motion_bvh(
        save_path,
        local_rot_mats,
        root_positions,
        skeleton=skeleton,
        fps=fps,
        standard_tpose=False,
    )


def _to_tensor(value: Any) -> torch.Tensor | None:
    """Convert numpy array to torch tensor, or return None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, torch.Tensor):
        return value
    return None


def _sync_timeline_prompts(
    client,
    prompts: list[str],
    durations_sec: list[float],
    fps: float,
) -> None:
    """Update the Viser timeline prompts to match the API request.

    Clears existing prompts and adds new ones matching the API's
    prompt/duration layout so the Viser UI stays in sync.
    """
    timeline = client.timeline
    timeline.clear_prompts()

    start_frame = 0
    fps_int = int(fps)
    for i, (text, dur) in enumerate(zip(prompts, durations_sec)):
        end_frame = start_frame + int(dur * fps_int)
        color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
        timeline.add_prompt(text, start_frame, end_frame, color=color)
        start_frame = end_frame


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(demo: "Demo") -> FastAPI:  # noqa: F821
    """Build a FastAPI application wired to the shared *demo* instance.

    Args:
        demo: The live ``Demo`` instance that holds models, sessions, and
            the generation lock.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Kimodo Headless API",
        description="Programmatic motion generation sharing a Viser demo instance.",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Server health and status."""
        return HealthResponse(
            status="ok" if demo._cuda_healthy else "degraded",
            cuda_healthy=demo._cuda_healthy,
            loaded_models=list(demo.models.keys()),
            connected_clients=len(demo.client_sessions),
        )

    @app.get("/clients", response_model=List[ClientInfo])
    async def list_clients() -> List[ClientInfo]:
        """List connected Viser clients (for targeting with ``client_id``)."""
        result: list[ClientInfo] = []
        for cid, session in demo.client_sessions.items():
            result.append(
                ClientInfo(
                    client_id=cid,
                    model_name=session.model_name,
                    frame_idx=session.frame_idx,
                    playing=session.playing,
                )
            )
        return result

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """Generate motion from text prompts and optional constraints.

        When ``client_id`` references a connected Viser client the motion
        is rendered in the 3D viewer automatically (hybrid mode).
        Otherwise the model runs headless and only raw data is returned.
        """
        # ---- resolve target model bundle ----
        # Use the first non-empty Viser client's model, or the default.
        target_model_name: str | None = None
        target_client = None
        target_session = None

        if req.client_id is not None:
            target_session = demo.client_sessions.get(req.client_id)
            if target_session is not None:
                target_client = target_session.client
                target_model_name = target_session.model_name

        if target_model_name is None:
            # Try any connected client
            for cid, session in demo.client_sessions.items():
                target_client = session.client
                target_session = session
                target_model_name = session.model_name
                break

        if target_model_name is None:
            target_model_name = demo.default_model_name

        model_bundle = demo.load_model(target_model_name)
        model = model_bundle.model
        skeleton = model_bundle.skeleton
        fps = model_bundle.model_fps

        # ---- resolve inputs ----
        durations_sec = _resolve_durations(req, fps)
        prompts = _resolve_prompts(req)
        num_frames = _compute_num_frames(durations_sec, fps)

        cfg_kwargs = _resolve_cfg_kwargs(req)
        constraint_lst = _build_constraint_lst(req.constraints, skeleton)

        # ---- seed ----
        if req.seed is not None:
            seed_everything(req.seed)

        # ---- determine generation path ----
        # Hybrid mode requires a connected Viser client AND no explicit API
        # constraints (the hybrid path reads constraints from Viser tracks;
        # explicit API constraints would conflict).
        use_hybrid = target_client is not None and target_session is not None and not constraint_lst

        print(f"[headless-api] Generating motion: prompt={prompts[0][:50]}..., "
              f"duration={durations_sec}s, samples={req.num_samples}, "
              f"hybrid={use_hybrid}, "
              f"auto_save={'yes' if req.auto_save else 'no'}"
              + (f" ({req.auto_save_format})" if req.auto_save else ""))

        if use_hybrid:
            # ---- hybrid path: render via Viser ----
            # Demo.generate() manages its own _generation_lock internally --
            # we must NOT acquire it here or we deadlock (the lock is non-reentrant).
            #
            # Sync timeline prompts so the Viser UI reflects the API request.
            _sync_timeline_prompts(target_client, prompts, durations_sec, fps)

            # Constraints come from the Viser timeline tracks, set up by the user
            # in the GUI before calling this endpoint.
            model_constraints = None
            if target_session.constraints:
                from .generation import compute_model_constraints_lst

                model_constraints = compute_model_constraints_lst(
                    target_session, model_bundle, sum(num_frames), demo.device
                )

            demo.generate(
                target_client,
                prompts,
                num_frames,
                req.num_samples,
                req.seed or 0,
                req.diffusion_steps,
                cfg_weight=cfg_kwargs.get("cfg_weight", [2.0, 2.0]),
                cfg_type=cfg_kwargs.get("cfg_type", "separated"),
                postprocess_parameters={
                    "post_processing": req.post_processing,
                    "root_margin": req.root_margin,
                },
                transitions_parameters={
                    "num_transition_frames": req.num_transition_frames,
                },
            )
            output = _extract_motion_from_session(target_session)
            output["fps"] = fps
            meta = {
                "num_samples": 1,
                "total_frames": output["posed_joints"].shape[0],
                "fps": fps,
            }

            # Auto-save if requested
            auto_saved_path = None
            if req.auto_save:
                auto_saved_path = _auto_save_motion(
                    output,
                    prompts,
                    req.auto_save_dir,
                    req.auto_save_format,
                    skeleton,
                    demo.device,
                    fps,
                )
            print(f"[headless-api] Hybrid generation complete, "
                  f"response_format={req.output_format}, "
                  f"auto_save={'yes' if req.auto_save else 'no'}"
                  + (f" ({req.auto_save_format} -> {auto_saved_path})" if auto_saved_path else ""))
        else:
            # ---- headless path: call model directly ----
            print("[headless-api] Entering headless path (direct model call)")
            # Serialise GPU access ourselves since we bypass Demo.generate().
            locked = demo._generation_lock.acquire(blocking=False)
            if not locked:
                logger.info("Waiting for GPU lock (another generation in progress)...")
                print("[auto-save] Waiting for GPU lock (another generation in progress)...")
                demo._generation_lock.acquire()

            try:
                use_postprocess = req.post_processing
                if "g1" in target_model_name:
                    # G1 post-processing is not recommended
                    use_postprocess = False

                output = model(
                    prompts,
                    num_frames,
                    num_denoising_steps=req.diffusion_steps,
                    multi_prompt=req.multi_prompt,
                    constraint_lst=constraint_lst,
                    num_samples=req.num_samples,
                    num_transition_frames=req.num_transition_frames,
                    post_processing=use_postprocess,
                    root_margin=req.root_margin,
                    return_numpy=True,
                    first_heading_angle=torch.tensor(req.first_heading_angle)
                    if req.first_heading_angle != 0.0
                    else None,
                    **cfg_kwargs,
                )
                output["fps"] = fps
                meta = _model_output_metadata(output, fps)
            finally:
                demo._generation_lock.release()

            # Auto-save if requested
            auto_saved_path = None
            if req.auto_save:
                auto_saved_path = _auto_save_motion(
                    output,
                    prompts,
                    req.auto_save_dir,
                    req.auto_save_format,
                    skeleton,
                    demo.device,
                    fps,
                )
            print(f"[headless-api] Headless generation complete, "
                  f"response_format={req.output_format}, "
                  f"auto_save={'yes' if req.auto_save else 'no'}"
                  + (f" ({req.auto_save_format} -> {auto_saved_path})" if auto_saved_path else ""))

        # ---- serialise response ----
        if req.output_format == "npz":
            return _build_npz_response(output, req, auto_saved_path)

        # JSON path
        response_body: dict = {
            "meta": meta,
            "motion": _build_json_response(output),
        }
        if auto_saved_path:
            response_body["auto_saved_path"] = auto_saved_path
        return response_body

    return app
