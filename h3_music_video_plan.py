"""Compact, loop-friendly clip plans for MiniMax H3 music videos."""

from __future__ import annotations

import json
import logging
import math


FPS = 24.0
MAX_SEED = (1 << 64) - 1
MAX_CLIPS = 64
_LOG = logging.getLogger("h3_motion_context.music_progress")


def _parse_plan(plan_json):
    try:
        value = json.loads(str(plan_json))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"h3_music_plan: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(value, list) or not value:
        raise ValueError("h3_music_plan: the plan must be a non-empty JSON array")

    clips = []
    for zero_index, raw in enumerate(value):
        if isinstance(raw, str):
            raw = {"prompt": raw}
        if not isinstance(raw, dict):
            raise ValueError(
                f"h3_music_plan: clip {zero_index + 1} must be an object or a prompt string"
            )

        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"h3_music_plan: clip {zero_index + 1} has no prompt")

        seed = raw.get("seed", zero_index)
        if isinstance(seed, bool):
            raise ValueError(f"h3_music_plan: clip {zero_index + 1} seed must be an integer")
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"h3_music_plan: clip {zero_index + 1} seed must be an integer"
            ) from exc
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(
                f"h3_music_plan: clip {zero_index + 1} seed must be between 0 and {MAX_SEED}"
            )

        clip = dict(raw)
        clip.update(index=zero_index, clip_number=zero_index + 1, prompt=prompt.strip(), seed=seed)
        if "start_seconds" in clip:
            try:
                clip["start_seconds"] = float(clip["start_seconds"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"h3_music_plan: clip {zero_index + 1} start_seconds must be numeric"
                ) from exc
            if not math.isfinite(clip["start_seconds"]) or clip["start_seconds"] < 0:
                raise ValueError(
                    f"h3_music_plan: clip {zero_index + 1} start_seconds must be finite and >= 0"
                )
        clips.append(clip)
    total = len(clips)
    for clip in clips:
        clip["total_clips"] = total
    return clips


class MiniMaxH3MusicVideoClipPlan:
    """Turn an editable JSON shot list into a native ComfyUI list."""

    DEFAULT_PLAN = json.dumps(
        [
            {"prompt": "Describe music-video clip 1", "seed": 1},
            {"prompt": "Describe music-video clip 2", "seed": 2},
        ],
        indent=2,
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_PLAN,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "JSON array. Each item needs prompt and may set seed or an explicit "
                            "start_seconds. The array length is the number of generated clips."
                        ),
                    },
                ),
                "master_audio": ("AUDIO",),
                "frame_count": ("INT", {"default": 243, "min": 5, "max": 9999}),
                "context_frames": ("INT", {"default": 39, "min": 0, "max": 9999}),
                "tail_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.25,
                        "tooltip": "Extra video after the master audio for the ending fade.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("H3_MUSIC_CLIP_SPEC",)
    RETURN_NAMES = ("clips",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/music video"
    DESCRIPTION = (
        "Build enough loop clips to cover the master audio plus the requested ending tail. "
        "If necessary, the editable shot plan repeats in order."
    )

    def build(
        self,
        plan_json,
        master_audio=None,
        frame_count=243,
        context_frames=39,
        tail_seconds=5.0,
    ):
        base = _parse_plan(plan_json)
        if master_audio is None:
            return (base,)
        frame_count = int(frame_count)
        context_frames = int(context_frames)
        if context_frames < 0 or context_frames >= frame_count:
            raise ValueError("h3_music_plan: context_frames must be >= 0 and < frame_count")
        sample_rate = int(master_audio["sample_rate"])
        waveform = master_audio["waveform"]
        if sample_rate <= 0 or getattr(waveform, "ndim", 0) != 3:
            raise ValueError("h3_music_plan: master_audio must contain [B,C,L] waveform audio")
        samples = int(waveform.shape[-1])
        target_frames = math.ceil((samples / sample_rate + float(tail_seconds)) * FPS)
        advance = frame_count - context_frames
        clip_count = 1 + max(0, math.ceil((target_frames - frame_count) / advance))
        if clip_count > MAX_CLIPS:
            raise ValueError(
                f"h3_music_plan: audio plus tail needs {clip_count} clips; maximum is {MAX_CLIPS}"
            )

        clips = []
        for index in range(clip_count):
            clip = dict(base[index % len(base)])
            clip.pop("start_seconds", None)
            clip.update(index=index, clip_number=index + 1, total_clips=clip_count)
            clips.append(clip)
        return (clips,)

    @classmethod
    def IS_CHANGED(cls, plan_json, master_audio=None, frame_count=243, context_frames=39, tail_seconds=5.0):
        if master_audio is None:
            audio_key = None
        else:
            waveform = master_audio.get("waveform")
            audio_key = (master_audio.get("sample_rate"), tuple(getattr(waveform, "shape", ())))
        return (str(plan_json), audio_key, int(frame_count), int(context_frames), float(tail_seconds))


class MiniMaxH3MusicVideoClipSpec:
    """Expose the current loop item and derive its master-song position."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("H3_MUSIC_CLIP_SPEC", {"forceInput": True}),
                "frame_count": ("INT", {"default": 362, "min": 5, "max": 9999}),
                "context_frames": ("INT", {"default": 39, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("prompt", "seed", "clip_start_seconds", "clip_number")
    FUNCTION = "extract"
    CATEGORY = "conditioning/minimax/music video"
    DESCRIPTION = (
        "Read one H3 music-video loop item. Timeline starts advance by "
        "(frame_count - context_frames) / 24 unless start_seconds is explicit."
    )

    def extract(self, clip, frame_count=362, context_frames=39):
        if not isinstance(clip, dict):
            raise ValueError("h3_music_plan: loop item is not an H3 clip specification")
        frame_count = int(frame_count)
        context_frames = int(context_frames)
        if frame_count < 5:
            raise ValueError("h3_music_plan: frame_count must be >= 5")
        if context_frames < 0 or context_frames >= frame_count:
            raise ValueError("h3_music_plan: context_frames must be >= 0 and < frame_count")
        start = clip.get("start_seconds")
        if start is None:
            start = int(clip["index"]) * (frame_count - context_frames) / FPS
        number = int(clip["clip_number"])
        total = int(clip.get("total_clips", number))
        banner = f"  MINIMAX H3 MUSIC VIDEO  |  SEQUENCE {number:02d} / {total:02d}  "
        border = "=" * len(banner)
        _LOG.warning("\n%s\n%s\n%s", border, banner, border)
        return (
            str(clip["prompt"]),
            int(clip["seed"]),
            float(start),
            number,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MusicVideoClipPlan": MiniMaxH3MusicVideoClipPlan,
    "MiniMaxH3MusicVideoClipSpec": MiniMaxH3MusicVideoClipSpec,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MusicVideoClipPlan": "H3 Music Video Clip Plan",
    "MiniMaxH3MusicVideoClipSpec": "H3 Music Video Current Clip",
}
