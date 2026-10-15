"""Persistent, lazy per-clip checkpoints for loop-based H3 music videos."""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path

import folder_paths

try:
    from .h3_project_name import normalize_project_name
except ImportError:  # Standalone module loading used by the test suite.
    def normalize_project_name(value):
        name = re.sub(r"[\\/:*?\"<>|]+", "_", str(value or "").strip())
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"_+", "_", name).strip(" ._")
        if not name or name in {".", ".."}:
            raise ValueError("h3_project_name: enter a non-empty project name")
        return name

try:
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover - ComfyUI ships safetensors
    load_file = save_file = None


_LOG = logging.getLogger("h3_motion_context.music_checkpoint")


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    if not name:
        raise ValueError("h3_music_checkpoint: checkpoint_set cannot be empty")
    return name[:120]


def _checkpoint_path(checkpoint_set, clip_number):
    root = Path(folder_paths.get_output_directory()) / "h3_music_video_checkpoints"
    return root / _safe_name(checkpoint_set) / f"clip_{int(clip_number):04d}.safetensors"


def _streams(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("h3_music_checkpoint: expected a MiniMax H3 LATENT dictionary")
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        parts = list(samples)
    else:
        raise ValueError("h3_music_checkpoint: expected an H3 video/audio latent pair")
    if len(parts) < 2:
        raise ValueError("h3_music_checkpoint: H3 checkpoint requires video and audio streams")
    return parts[0], parts[1]


class MiniMaxH3MusicVideoCheckpoint:
    """Load accepted prefix clips and save regenerated clips in fixed slots."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_latent": ("LATENT", {"lazy": True}),
                "clip_number": ("INT", {"forceInput": True}),
                "resume_from": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 9999,
                        "tooltip": (
                            "1 generates every clip. 5 loads checkpoints 1-4 without "
                            "running their samplers, then regenerates clip 5 onward."
                        ),
                    },
                ),
                "checkpoint_set": (
                    "STRING",
                    {
                        "default": "music_video_main",
                        "tooltip": (
                            "Fixed checkpoint set under ComfyUI/output/"
                            "h3_music_video_checkpoints. Use a new name for a new project."
                        ),
                    },
                ),
            },
            "optional": {
                "project_name": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("clip_latent", "resume_from")
    FUNCTION = "checkpoint"
    CATEGORY = "conditioning/minimax/music video"
    DESCRIPTION = (
        "Persistent per-sequence checkpoint. Clips before Resume From are loaded lazily; "
        "that prevents their sampler branch from executing. Later clips are generated and "
        "atomically overwrite their fixed checkpoint slots."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("NaN")

    def check_lazy_status(
        self,
        generated_latent=None,
        clip_number=1,
        resume_from=1,
        checkpoint_set="music_video_main",
        project_name=None,
    ):
        if int(clip_number) < int(resume_from):
            return []
        if generated_latent is None:
            return ["generated_latent"]
        return []

    def checkpoint(
        self,
        generated_latent=None,
        clip_number=1,
        resume_from=1,
        checkpoint_set="music_video_main",
        project_name=None,
    ):
        if load_file is None or save_file is None:
            raise RuntimeError("h3_music_checkpoint: safetensors is unavailable")
        clip_number = int(clip_number)
        if project_name is not None:
            checkpoint_set = normalize_project_name(project_name)
        path = _checkpoint_path(checkpoint_set, clip_number)

        if clip_number < int(resume_from):
            if not path.is_file():
                raise FileNotFoundError(
                    f"h3_music_checkpoint: cannot resume at clip {int(resume_from)}; "
                    f"missing checkpoint for clip {clip_number}: {path}"
                )
            data = load_file(str(path), device="cpu")
            if "video" not in data or "audio" not in data:
                raise ValueError(f"h3_music_checkpoint: invalid H3 checkpoint: {path}")
            _LOG.warning(
                "[H3 MUSIC VIDEO] [OK] SEQUENCE %02d RESTORED FROM CHECKPOINT: %s",
                clip_number,
                path,
            )
            return ({"samples": [data["video"], data["audio"]]}, int(resume_from))

        if generated_latent is None:
            raise RuntimeError("h3_music_checkpoint: generated latent was not resolved")
        video, audio = _streams(generated_latent)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            save_file(
                {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().contiguous()},
                str(temp),
                metadata={"format": "h3_music_video_checkpoint_v1", "clip_number": str(clip_number)},
            )
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        _LOG.warning(
            "[H3 MUSIC VIDEO] [OK] SEQUENCE %02d GENERATED; CHECKPOINT SAVED: %s",
            clip_number,
            path,
        )
        return (generated_latent, int(resume_from))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MusicVideoCheckpoint": MiniMaxH3MusicVideoCheckpoint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MusicVideoCheckpoint": "H3 Music Video Checkpoint / Resume",
}
