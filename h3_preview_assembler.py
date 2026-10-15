"""Assemble the final music video directly from per-sequence review MP4s."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import folder_paths
import soundfile as sf

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


_LOG = logging.getLogger("h3_motion_context.preview_assembler")
FPS = 24


class MiniMaxH3AssembleMusicVideoPreviews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "completed_clips": ("INT", {"forceInput": True}),
                "master_audio": ("AUDIO",),
                "context_frames": ("INT", {"default": 39, "min": 0, "max": 9999}),
                "tail_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 60.0, "step": 0.25}),
                "preview_directory": ("STRING", {"default": "h3_preview/music_video"}),
                "filename_prefix": ("STRING", {"default": "video/h3_music_video_final"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
            },
            "optional": {
                "project_name": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "assemble"
    INPUT_IS_LIST = True
    OUTPUT_NODE = False
    HAS_INTERMEDIATE_OUTPUT = True
    CATEGORY = "conditioning/minimax/music video"
    DESCRIPTION = (
        "Builds the final movie from the already decoded per-sequence MP4 previews. "
        "It removes the H3 context overlap, restores the untouched master audio, "
        "adds the requested silent visual tail and fades the picture to black."
    )

    @staticmethod
    def _clip_count(value):
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if not values:
            raise ValueError("h3_preview_assembler: the loop completed no clips")
        numbers = [int(item) for item in values]
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            raise ValueError(
                "h3_preview_assembler: completed clip numbers must be contiguous "
                f"from 1; got {numbers}"
            )
        return len(numbers)

    @staticmethod
    def _one(value, name):
        if isinstance(value, list):
            if not value:
                raise ValueError(f"h3_preview_assembler: missing {name}")
            return value[0]
        return value

    @staticmethod
    def _latest_preview(directory, number):
        matches = list(directory.glob(f"sequence_{number:04d}_*-audio.mp4"))
        if not matches:
            raise FileNotFoundError(
                f"h3_preview_assembler: missing MP4 preview for sequence {number:02d} "
                f"under {directory}"
            )
        return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def assemble(
        self,
        completed_clips,
        master_audio,
        context_frames=39,
        tail_seconds=5.0,
        preview_directory="h3_preview/music_video",
        filename_prefix="video/h3_music_video_final",
        crf=19,
        project_name=None,
    ):
        count = self._clip_count(completed_clips)
        master_audio = self._one(master_audio, "master_audio")
        context_frames = int(self._one(context_frames, "context_frames"))
        tail_seconds = float(self._one(tail_seconds, "tail_seconds"))
        preview_directory = str(self._one(preview_directory, "preview_directory"))
        filename_prefix = str(self._one(filename_prefix, "filename_prefix"))
        crf = int(self._one(crf, "crf"))
        if project_name is not None:
            project_name = normalize_project_name(self._one(project_name, "project_name"))
            preview_directory = f"h3_preview/{project_name}"
            filename_prefix = f"video/{project_name}"
        output_root = Path(folder_paths.get_output_directory()).resolve()
        preview_root = (output_root / str(preview_directory)).resolve()
        if output_root not in preview_root.parents:
            raise ValueError("h3_preview_assembler: preview directory must stay under ComfyUI output")
        clips = [self._latest_preview(preview_root, i) for i in range(1, count + 1)]

        waveform = master_audio["waveform"]
        if getattr(waveform, "ndim", 0) != 3 or int(waveform.shape[0]) < 1:
            raise ValueError("h3_preview_assembler: master_audio must be shaped [B,C,L]")
        sample_rate = int(master_audio["sample_rate"])
        pcm = waveform[0].detach().to(device="cpu").float().transpose(0, 1).numpy()
        audio_seconds = int(pcm.shape[0]) / float(sample_rate)
        total_seconds = audio_seconds + max(0.0, float(tail_seconds))
        target_frames = int(round(total_seconds * FPS))
        fade_duration = min(max(0.0, float(tail_seconds)), total_seconds)
        fade_start = max(0.0, total_seconds - fade_duration)

        prefix = Path(str(filename_prefix).replace("\\", "/"))
        out_dir = (output_root / prefix.parent).resolve()
        if output_root not in out_dir.parents and out_dir != output_root:
            raise ValueError("h3_preview_assembler: filename prefix must stay under ComfyUI output")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{prefix.name}_{datetime.now():%Y%m%d_%H%M%S}.mp4"

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("h3_preview_assembler: ffmpeg was not found on PATH")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="h3_master_audio_", suffix=".wav", dir=out_dir, delete=False
            ) as handle:
                temp_path = Path(handle.name)
            sf.write(str(temp_path), pcm, sample_rate, subtype="PCM_24")

            command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin"]
            for clip in clips:
                command.extend(["-i", str(clip)])
            command.extend(["-i", str(temp_path)])

            chains = []
            labels = []
            for index in range(count):
                start = 0 if index == 0 else int(context_frames)
                label = f"v{index}"
                chains.append(
                    f"[{index}:v]trim=start_frame={start},setpts=PTS-STARTPTS,fps={FPS}[{label}]"
                )
                labels.append(f"[{label}]")
            chains.append("".join(labels) + f"concat=n={count}:v=1:a=0[vcat]")
            fade = f",fade=t=out:st={fade_start:.6f}:d={fade_duration:.6f}" if fade_duration else ""
            chains.append(f"[vcat]trim=end_frame={target_frames},setpts=PTS-STARTPTS{fade}[vout]")
            audio_index = count
            chains.append(
                f"[{audio_index}:a]apad=pad_dur={max(0.0, float(tail_seconds)):.6f},"
                f"atrim=duration={total_seconds:.6f},asetpts=PTS-STARTPTS[aout]"
            )
            command.extend([
                "-filter_complex", ";".join(chains),
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
                "-movflags", "+faststart", str(output_path),
            ])
            _LOG.info(
                "h3_preview_assembler: joining %d preview MP4s; removing %d context frames from clips 2-%d; target %.3fs",
                count, int(context_frames), count, total_seconds,
            )
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"h3_preview_assembler: ffmpeg failed with exit code {exc.returncode}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        relative = output_path.relative_to(output_root).as_posix()
        _LOG.info("h3_preview_assembler: final video saved: %s", output_path)
        return {"ui": {"gifs": [{"filename": output_path.name, "subfolder": prefix.parent.as_posix(), "type": "output", "format": "video/mp4"}]}, "result": ([relative],)}
