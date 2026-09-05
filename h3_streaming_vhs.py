"""Single-execution low-RAM final H3 video output backed by Video Helper Suite.

The normal ComfyUI IMAGE datatype is a materialized ``torch.Tensor``.  Long H3
runs can therefore require tens of GiB just to hold the final RGB movie before
VHS sees it.  These output nodes keep the frame stream internal instead:

    H3 latent -> decode one clip -> resolve seam -> VHS encoder -> release clip

No custom object is exposed as a ComfyUI IMAGE output. The internal one-shot sequence exists only during the call into VHS_VideoCombine.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime
from pathlib import PurePosixPath
import logging

import torch

# Keep this module importable even in lightweight/mock test environments that
# replace the implementation modules with partial stubs.  Runtime helpers are
# resolved only when a streaming node actually executes.
FPS = 24.0

def _ext():
    from . import existing_video_extension as module
    return module

def _music():
    from . import h3_song_audio_context as module
    return module

def _cfr_index_map(*a, **k): return _ext()._cfr_index_map(*a, **k)
def _decode_h3_audio_cpu(*a, **k): return _ext()._decode_h3_audio_cpu(*a, **k)
def _decode_h3_video_cpu(*a, **k): return _ext()._decode_h3_video_cpu(*a, **k)
def _fit_waveform(*a, **k): return _ext()._fit_waveform(*a, **k)
def _conform_waveform_length(*a, **k): return _ext()._conform_waveform_length(*a, **k)
def _pixel_frames(*a, **k): return _ext()._pixel_frames(*a, **k)
def _release_decode_memory(*a, **k): return _ext()._release_decode_memory(*a, **k)
def _resample_waveform(*a, **k): return _ext()._resample_waveform(*a, **k)
def _resize_images(*a, **k): return _ext()._resize_images(*a, **k)
def _snap_av_context_length(*a, **k): return _ext()._snap_context_length(*a, **k)
def _stereo_first_batch(*a, **k): return _ext()._stereo_first_batch(*a, **k)
def _canonical_audio(*a, **k): return _ext()._canonical_audio(*a, **k)
def _streams_from_latent(*a, **k): return _ext()._streams_from_latent(*a, **k)
def _snap_music_context_length(*a, **k): return _music()._snap_context_length(*a, **k)

def sample_boundary_from_frames(*a, **k):
    from . import h3_timing as module
    return module.sample_boundary_from_frames(*a, **k)

_LOG = logging.getLogger("h3_motion_context.streaming_vhs")


def _rgb_gib(frames, height, width):
    return int(frames) * int(height) * int(width) * 3 * 4 / float(1024 ** 3)


def _resolve_vhs_video_combine():
    """Resolve VHS at execution time without making plugin import order fragile."""
    try:
        import nodes as comfy_nodes
    except Exception as exc:  # pragma: no cover - only reachable in broken Comfy installs
        raise RuntimeError(
            "h3_streaming_vhs: could not import ComfyUI's nodes module"
        ) from exc

    cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("VHS_VideoCombine")
    if cls is None:
        raise RuntimeError(
            "h3_streaming_vhs: VHS_VideoCombine is required. Install/enable "
            "ComfyUI-VideoHelperSuite before using the streaming final output."
        )

    combine = getattr(cls, "combine_video", None)
    if not callable(combine):
        raise RuntimeError(
            "h3_streaming_vhs: the installed VHS_VideoCombine node does not expose "
            "the expected combine_video API. Update ComfyUI-VideoHelperSuite."
        )

    # Fail with a useful message instead of a long TypeError if an older VHS
    # build is missing controls used by the direct-stream nodes. Future/wrapped
    # implementations that accept **kwargs remain compatible.
    try:
        parameters = inspect.signature(combine).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        expected = {
            "frame_rate", "loop_count", "images", "filename_prefix", "format",
            "pingpong", "save_output", "prompt", "extra_pnginfo", "audio",
            "unique_id", "pix_fmt", "crf", "save_metadata", "trim_to_audio",
        }
        missing = sorted(expected.difference(parameters))
        if missing:
            raise RuntimeError(
                "h3_streaming_vhs: the installed VideoHelperSuite is too old for "
                "direct H3 streaming; VHS_VideoCombine.combine_video is missing: "
                + ", ".join(missing)
                + ". Update ComfyUI-VideoHelperSuite."
            )
    return cls


class _OneShotFrameSequence:
    """Sequence facade for VHS that primes once and then streams exactly once.

    VHS currently asks for ``len(images)``, reads ``images[0]`` for output
    dimensions/metadata, and then iterates the image source.  Priming the
    generator in ``__getitem__(0)`` avoids decoding Clip 1 twice.
    """

    def __init__(self, frame_count, generator_factory):
        self._frame_count = int(frame_count)
        self._generator_factory = generator_factory
        self._generator = None
        self._first = None
        self._primed = False
        self._iterated = False

    def __len__(self):
        return self._frame_count

    def _prime(self):
        if self._primed:
            return
        self._generator = iter(self._generator_factory())
        try:
            self._first = next(self._generator).detach().clone()
        except StopIteration as exc:
            raise RuntimeError("h3_streaming_vhs: frame stream produced no frames") from exc
        self._primed = True

    def __getitem__(self, index):
        if int(index) != 0:
            raise IndexError(
                "h3_streaming_vhs: internal frame stream only supports the first-frame probe"
            )
        self._prime()
        return self._first

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("h3_streaming_vhs: frame stream is one-shot")
        self._prime()
        self._iterated = True
        first = self._first
        self._first = None
        yield first
        yield from self._generator


def _yield_segments_and_hold(segments, hold_frames):
    """Yield all but the final ``hold_frames`` and return a detached CPU tail.

    The returned tail is the only RGB history kept for the next seam.  It is
    cloned so it does not keep the much larger decoded clip storage alive.
    """
    segments = [seg for seg in segments if seg is not None and int(seg.shape[0]) > 0]
    total = sum(int(seg.shape[0]) for seg in segments)
    hold = max(0, min(int(hold_frames), total))
    emit = total - hold
    remainders = []

    for seg in segments:
        n = int(seg.shape[0])
        take = min(n, emit)
        if take > 0:
            for frame in seg[:take]:
                yield frame
            emit -= take
        if take < n:
            remainders.append(seg[take:])

    if emit != 0:
        raise RuntimeError("h3_streaming_vhs: internal frame accounting error")
    if hold == 0:
        return None

    if not remainders:
        raise RuntimeError("h3_streaming_vhs: failed to retain requested seam tail")
    if len(remainders) == 1:
        tail = remainders[0].detach().to(device="cpu", dtype=torch.float32).clone()
    else:
        tail = torch.cat(remainders, dim=0).detach().to(device="cpu", dtype=torch.float32)
    if int(tail.shape[0]) != hold:
        raise RuntimeError(
            f"h3_streaming_vhs: retained {int(tail.shape[0])} frames, expected {hold}"
        )
    return tail.contiguous()


def _seam_overlaps(raw_frames, contexts, overlap):
    """Return overlap frames for each seam using the existing assembler rules."""
    overlaps = [0] * len(raw_frames)
    write_frame = int(raw_frames[0])
    overlap = max(0, int(overlap))
    for i in range(1, len(raw_frames)):
        ov = min(overlap, int(contexts[i]), write_frame)
        overlaps[i] = ov
        write_frame += int(raw_frames[i]) - int(contexts[i])
    return overlaps


def _generated_frame_generator(video_vae, videos, raw_frames, contexts, overlap, log_prefix):
    """Stream generated H3 clips with the same seam math as the old full buffer."""
    seam_ovs = _seam_overlaps(raw_frames, contexts, overlap)
    tail = None

    for i, video_latent in enumerate(videos):
        decoded = _decode_h3_video_cpu(video_vae, video_latent)
        if int(decoded.shape[0]) != int(raw_frames[i]):
            raise RuntimeError(
                f"{log_prefix}: Clip {i + 1} video decode produced "
                f"{int(decoded.shape[0])} frames; expected {int(raw_frames[i])}"
            )

        if i == 0:
            segments = [decoded]
        else:
            ctx = int(contexts[i])
            ov = int(seam_ovs[i])
            if ov > 0:
                if tail is None or int(tail.shape[0]) != ov:
                    raise RuntimeError(
                        f"{log_prefix}: seam {i} retained tail mismatch "
                        f"({0 if tail is None else int(tail.shape[0])} != {ov})"
                    )
                dst = decoded[ctx - ov : ctx]
                alpha = torch.linspace(
                    0.0, 1.0, ov + 2, dtype=torch.float32, device="cpu"
                )[1:-1].view(-1, 1, 1, 1)
                tail.mul_(1.0 - alpha).add_(dst * alpha)
                del dst, alpha
                segments = [tail, decoded[ctx:]]
            else:
                segments = [decoded[ctx:]]
                tail = None

        next_hold = int(seam_ovs[i + 1]) if i + 1 < len(videos) else 0
        new_tail = yield from _yield_segments_and_hold(segments, next_hold)
        tail = new_tail
        del decoded, segments, new_tail
        _release_decode_memory()

    if tail is not None:
        raise RuntimeError(f"{log_prefix}: unflushed seam tail after final clip")


def _existing_base_and_generated_extensions_generator(
    video_vae,
    source_frames,
    source_idx,
    width,
    height,
    crop,
    extension_videos,
    raw_frames,
    contexts,
    overlap,
):
    """Stream an existing source followed by generated extension clips."""
    # Build the same seam-overlap plan as [base, ext1, ext2, ...].
    all_frames = [int(source_idx.numel())] + [int(x) for x in raw_frames]
    all_contexts = [0] + [int(x) for x in contexts]
    seam_ovs = _seam_overlaps(all_frames, all_contexts, overlap)
    first_hold = int(seam_ovs[1]) if len(all_frames) > 1 else 0

    # Stream the CFR-resampled/resized source in small chunks.  Only its final
    # seam tail is copied out of the source batch.
    base_frames = int(source_idx.numel())
    emit_upto = base_frames - first_hold
    tail_parts = []
    chunk = 32
    for start in range(0, base_frames, chunk):
        end = min(base_frames, start + chunk)
        ids = source_idx[start:end]
        part = source_frames.index_select(0, ids)
        part = _resize_images(part, width, height, crop).detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        local_emit = max(0, min(int(part.shape[0]), emit_upto - start))
        if local_emit > 0:
            for frame in part[:local_emit]:
                yield frame
        if local_emit < int(part.shape[0]):
            tail_parts.append(part[local_emit:].clone())
        del part

    if first_hold > 0:
        tail = tail_parts[0] if len(tail_parts) == 1 else torch.cat(tail_parts, dim=0)
        tail = tail.contiguous()
        if int(tail.shape[0]) != first_hold:
            raise RuntimeError(
                f"h3_streaming_av: retained {int(tail.shape[0])} source frames, expected {first_hold}"
            )
    else:
        tail = None
    del tail_parts
    _release_decode_memory()

    for ext_i, video_latent in enumerate(extension_videos, start=1):
        decoded = _decode_h3_video_cpu(video_vae, video_latent)
        expected = int(raw_frames[ext_i - 1])
        if int(decoded.shape[0]) != expected:
            raise RuntimeError(
                f"h3_streaming_av: Extension {ext_i} video decode produced "
                f"{int(decoded.shape[0])} frames; expected {expected}"
            )
        ctx = int(contexts[ext_i - 1])
        ov = int(seam_ovs[ext_i])
        if ov > 0:
            if tail is None or int(tail.shape[0]) != ov:
                raise RuntimeError(
                    f"h3_streaming_av: Extension {ext_i} retained tail mismatch"
                )
            dst = decoded[ctx - ov : ctx]
            alpha = torch.linspace(0.0, 1.0, ov + 2, dtype=torch.float32)[1:-1].view(
                -1, 1, 1, 1
            )
            tail.mul_(1.0 - alpha).add_(dst * alpha)
            del dst, alpha
            segments = [tail, decoded[ctx:]]
        else:
            tail = None
            segments = [decoded[ctx:]]

        next_hold = int(seam_ovs[ext_i + 1]) if ext_i + 1 < len(all_frames) else 0
        new_tail = yield from _yield_segments_and_hold(segments, next_hold)
        tail = new_tail
        del decoded, segments, new_tail
        _release_decode_memory()

    if tail is not None:
        raise RuntimeError("h3_streaming_av: unflushed seam tail after final extension")


def _assemble_av_audio(
    audio_vae,
    mode,
    base_frames,
    base_audio_latent,
    source_audio,
    ext_streams,
    raw_frames,
    contexts,
):
    """Assemble AV audio with the continuation owning each protected overlap.

    Every extension already contains the previous clip's protected audio context.
    Keep that full decoded extension audio on the final timeline and replace the
    corresponding tail of the preceding clip.  This preserves the generation-side
    audio feather inside the protected prefix instead of discarding it at assembly.
    """
    audio_sr = int(
        getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 44100),
        )
    )
    final_frames = int(base_frames) + sum(
        int(raw_frames[i]) - int(contexts[i]) for i in range(len(raw_frames))
    )
    total_samples = sample_boundary_from_frames(final_frames, audio_sr, FPS)
    audio_out = torch.empty((1, 2, total_samples), dtype=torch.float32, device="cpu")

    if mode == "existing_video":
        canonical = _canonical_audio(source_audio, audio_sr, int(base_frames))
        wave = canonical["waveform"].detach().to("cpu", torch.float32)
        want = sample_boundary_from_frames(int(base_frames), audio_sr, FPS)
    else:
        wave, got_sr = _decode_h3_audio_cpu(audio_vae, base_audio_latent)
        wave = _stereo_first_batch(wave, "starter audio")
        wave = _resample_waveform(wave, got_sr, audio_sr, "starter audio")
        want = sample_boundary_from_frames(int(base_frames), audio_sr, FPS)
        wave = _conform_waveform_length(wave, want, "starter audio")

    # Seed the buffer with the complete base clip. Each extension below overwrites
    # the previous clip's final protected-context span with its own full prefix.
    audio_out[..., :want].copy_(wave[..., :want])
    del wave
    _release_decode_memory()

    cumulative_frames = int(base_frames)
    for i, (_video_latent, audio_latent) in enumerate(ext_streams):
        wave, got_sr = _decode_h3_audio_cpu(audio_vae, audio_latent)
        wave = _stereo_first_batch(wave, f"Extension {i + 1} audio")
        wave = _resample_waveform(wave, got_sr, audio_sr, f"Extension {i + 1} audio")

        extension_start_frame = cumulative_frames - int(contexts[i])
        extension_end_frame = extension_start_frame + int(raw_frames[i])
        extension_start_sample = sample_boundary_from_frames(
            extension_start_frame, audio_sr, FPS
        )
        extension_end_sample = sample_boundary_from_frames(
            extension_end_frame, audio_sr, FPS
        )
        expected_full = extension_end_sample - extension_start_sample
        wave = _conform_waveform_length(
            wave, expected_full, f"Extension {i + 1} full audio"
        ).detach().to("cpu", torch.float32)

        if extension_start_sample < 0 or extension_end_sample > total_samples:
            raise RuntimeError(
                f"h3_streaming_av: Extension {i + 1} audio maps outside final timeline"
            )

        audio_out[..., extension_start_sample:extension_end_sample].copy_(
            wave[..., :expected_full]
        )
        _LOG.info(
            "h3_streaming_av: Extension %d owns %d-frame protected audio overlap "
            "from absolute frame %d",
            i + 1,
            int(contexts[i]),
            extension_start_frame,
        )
        cumulative_frames = extension_end_frame
        del wave
        _release_decode_memory()

    if cumulative_frames != final_frames:
        raise RuntimeError(
            f"h3_streaming_av: audio timeline ended at frame {cumulative_frames}, "
            f"expected {final_frames}"
        )
    return {"waveform": audio_out, "sample_rate": audio_sr}


def _vhs_h264_inputs(filename_default, trim_default):
    return {
        "filename_prefix": ("STRING", {"default": filename_default}),
        "pix_fmt": (["yuv420p", "yuv420p10le", "lossless_ffv1"], {"default": "yuv420p"}),
        "crf": ("INT", {"default": 19, "min": 0, "max": 100, "step": 1}),
        "save_metadata": ("BOOLEAN", {"default": False}),
        "trim_to_audio": ("BOOLEAN", {"default": bool(trim_default)}),
        "save_output": ("BOOLEAN", {"default": True}),
    }


def _expand_output_prefix(prefix, now=None):
    """Resolve Comfy-style date folders for UI and API callers (server local time)."""
    now = now or datetime.now()
    values = {"yyyy": f"{now.year:04d}", "yy": f"{now.year % 100:02d}",
              "MM": f"{now.month:02d}", "dd": f"{now.day:02d}",
              "HH": f"{now.hour:02d}", "hh": f"{now.hour % 12 or 12:02d}",
              "mm": f"{now.minute:02d}", "ss": f"{now.second:02d}"}
    def expand(match):
        pattern = match.group(1)
        # Only documented date tokens and safe separators are supported.
        parts = re.findall(r"yyyy|yy|MM|dd|HH|hh|mm|ss|[^A-Za-z]", pattern)
        if "".join(parts) != pattern:
            raise ValueError("Unsupported date token; use yyyy, yy, MM, dd, HH, hh, mm, ss.")
        return "".join(values.get(part, part) for part in parts)
    value = re.sub(r"%date:([^%]+)%", expand, str(prefix)).replace("\\", "/")
    if (not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or re.search(r"[<>:\x00-\x1f]", value) or "%date:" in value):
        raise ValueError("Output prefix must be a relative filename/subfolder inside ComfyUI output.")
    return value


def _run_vhs_h264(
    frames,
    audio,
    filename_prefix,
    pix_fmt,
    crf,
    save_metadata,
    trim_to_audio,
    save_output,
    prompt,
    extra_pnginfo,
    unique_id,
):
    vhs_cls = _resolve_vhs_video_combine()
    vhs = vhs_cls()

    # IMPORTANT: final filename allocation belongs to VHS.  Do not construct a
    # fixed output path in this module: VHS scans the destination directory and
    # advances its numeric counter for repeated runs using the same prefix.
    # Keeping that responsibility here would risk reintroducing the historical
    # bug where each run overwrote the previous final video.
    # Keep the existing H.264 path as the default. ``lossless_ffv1`` is a
    # sentinel in the existing pix_fmt widget so old workflow widget ordering
    # and serialized defaults remain unchanged. VHS's built-in FFV1 MKV format
    # supplies its own codec defaults; rgb48le keeps our 3-channel H3 frames in
    # 16-bit RGB without introducing an unnecessary alpha plane. FLAC is used
    # for audio by that format.
    lossless = str(pix_fmt) == "lossless_ffv1"
    output_format = "video/ffv1-mkv" if lossless else "video/h264-mp4"
    output_pix_fmt = "rgb48le" if lossless else str(pix_fmt)

    result = vhs.combine_video(
        frame_rate=24,
        loop_count=0,
        images=frames,
        filename_prefix=_expand_output_prefix(filename_prefix),
        format=output_format,
        pingpong=False,
        save_output=bool(save_output),
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
        audio=audio,
        unique_id=unique_id,
        pix_fmt=output_pix_fmt,
        crf=int(crf),
        save_metadata=bool(save_metadata),
        trim_to_audio=bool(trim_to_audio),
    )

    # VHS's own browser-side preview is only installed for nodes whose class
    # name is literally VHS_VideoCombine.  These H3 streaming nodes call VHS
    # internally, so preserve the normal VHS ``gifs`` UI payload and also expose
    # the same saved MP4 through ComfyUI's native PreviewVideo representation
    # (``images`` + ``animated``).  This restores a visible final video preview
    # without materializing the final RGB movie as an IMAGE tensor.
    if isinstance(result, dict):
        ui_payload = result.setdefault("ui", {})
        previews = ui_payload.get("gifs") or []
        if previews:
            preview = previews[0]
            if all(key in preview for key in ("filename", "type")):
                ui_payload["images"] = [{
                    "filename": preview["filename"],
                    "subfolder": preview.get("subfolder", ""),
                    "type": preview["type"],
                }]
                ui_payload["animated"] = (True,)
    return result


class MiniMaxH3StreamLiveExtensionAVToVHS:
    """Stream a modular AV Extension timeline directly into VHS video output.

    The frontend exposes a user-selected number of ``extension_N`` sockets.
    Backend support is deliberately wider than the shipped six-extension
    example so the node can be reused in custom workflows.  Disconnected
    extension sockets are ignored instead of being treated as errors.
    """

    MAX_EXTENSIONS = 64
    DEFAULT_EXTENSION_INPUTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "start_mode": ("STRING", {"forceInput": True}),
            "input_count": (
                "INT",
                {
                    "default": cls.DEFAULT_EXTENSION_INPUTS,
                    "min": 1,
                    "max": cls.MAX_EXTENSIONS,
                    "step": 1,
                    "tooltip": (
                        "Number of extension latent sockets shown by the node. "
                        "Set this value, then click Update inputs."
                    ),
                },
            ),
            "context_frames": ("INT", {"default": 39, "min": 5, "max": 9999}),
            "video_overlap_frames": (
                "INT",
                {"default": 39, "min": 0, "max": 9999},
            ),
            "source_fps": (
                "FLOAT",
                {"default": 24.0, "min": 24.0, "max": 24.0, "step": 0.001},
            ),
            "crop": (["disabled", "center"], {"default": "disabled"}),
        }
        required.update(_vhs_h264_inputs("video/masked_av_extension", True))
        optional = {
            "source_frames": ("IMAGE", {"lazy": True}),
            "source_audio": ("AUDIO", {"lazy": True}),
            "starter_latent": ("LATENT", {"lazy": True}),
            # Optional execution-order gate. In the bundled AV Extension workflow
            # this is fed by the last enabled per-extension VHS preview. Requesting
            # it before any final-stream latent makes that preview finish first.
            # Custom workflows may leave it disconnected.
            "preview_gate": ("VHS_FILENAMES", {"lazy": True}),
            # Optional compatibility/controller cap.  The bundled AV Extension
            # workflow connects this to its controller parameter so bypassed
            # sampler groups above the active count remain lazy.  Custom users
            # can leave it disconnected and simply connect any of the visible
            # extension sockets they need.
            "active_extensions": (
                "INT",
                {
                    "forceInput": True,
                    "tooltip": (
                        "Optional cap on extension sockets considered. Leave "
                        "unconnected for modular/custom workflows."
                    ),
                },
            ),
        }
        # The browser extension hides sockets above input_count. Declaring the
        # full supported range here keeps prompt validation/backend execution
        # compatible with dynamically added sockets.
        for i in range(1, cls.MAX_EXTENSIONS + 1):
            optional[f"extension_{i}"] = ("LATENT", {"lazy": True})
        return {
            "required": required,
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "stream_to_vhs"
    # Keep this as an intermediate-output node so ordinary VHS clip previews
    # retain output scheduling priority. A tiny terminal sink makes the final
    # stream executable.
    OUTPUT_NODE = False
    HAS_INTERMEDIATE_OUTPUT = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Modular low-RAM AV Extension final output. Set Input Count and click "
        "Update inputs to expose as many extension latent sockets as needed. "
        "Disconnected sockets are skipped. Clips are decoded one at a time and "
        "streamed directly into VHS without materializing the full RGB movie."
    )

    @classmethod
    def _extension_limit(cls, input_count, active_extensions=None):
        configured = max(1, min(cls.MAX_EXTENSIONS, int(input_count)))
        if active_extensions is None:
            return configured
        return min(configured, max(0, int(active_extensions)))

    def check_lazy_status(
        self,
        video_vae,
        audio_vae,
        start_mode,
        input_count,
        context_frames,
        video_overlap_frames,
        source_fps,
        crop,
        filename_prefix,
        pix_fmt,
        crf,
        save_metadata,
        trim_to_audio,
        save_output,
        source_frames=None,
        source_audio=None,
        starter_latent=None,
        active_extensions=None,
        **kwargs,
    ):
        # Strict ordering: if a VHS preview gate is connected, resolve it before
        # asking ComfyUI for any of the expensive final-stream inputs. This keeps
        # the last enabled extension preview visible before final assembly starts.
        if "preview_gate" in kwargs and kwargs["preview_gate"] is None:
            return ["preview_gate"]

        needed = []
        if str(start_mode) == "existing_video":
            if source_frames is None:
                needed.append("source_frames")
            if source_audio is None:
                needed.append("source_audio")
        elif starter_latent is None:
            needed.append("starter_latent")

        # Optional inputs that are not connected are absent from kwargs.
        # Connected lazy inputs are present with value None until evaluated.
        # Request only those connected sockets; gaps are intentionally valid.
        limit = self._extension_limit(input_count, active_extensions)
        for i in range(1, limit + 1):
            name = f"extension_{i}"
            if name in kwargs and kwargs[name] is None:
                needed.append(name)
        return needed

    def stream_to_vhs(
        self,
        video_vae,
        audio_vae,
        start_mode="existing_video",
        input_count=DEFAULT_EXTENSION_INPUTS,
        context_frames=39,
        video_overlap_frames=39,
        source_fps=24.0,
        crop="disabled",
        filename_prefix="video/masked_av_extension",
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=False,
        trim_to_audio=True,
        save_output=True,
        source_frames=None,
        source_audio=None,
        starter_latent=None,
        preview_gate=None,
        active_extensions=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs,
    ):
        limit = self._extension_limit(input_count, active_extensions)
        extension_latents = []
        extension_slots = []
        for i in range(1, limit + 1):
            name = f"extension_{i}"
            value = kwargs.get(name)
            if value is None:
                continue
            extension_slots.append(i)
            extension_latents.append(value)

        if not extension_latents:
            cap_text = (
                f" (active_extensions cap={int(active_extensions)})"
                if active_extensions is not None else ""
            )
            raise ValueError(
                "h3_streaming_av: connect at least one extension latent within "
                f"the first {limit} configured input socket(s){cap_text}"
            )

        count = len(extension_latents)
        ext_streams = [_streams_from_latent(x) for x in extension_latents]
        ext_videos = [video for video, _audio in ext_streams]
        raw_frames = [_pixel_frames(int(video.shape[2])) for video in ext_videos]
        width = int(ext_videos[0].shape[4]) * 16
        height = int(ext_videos[0].shape[3]) * 16
        for video in ext_videos[1:]:
            if int(video.shape[4]) * 16 != width or int(video.shape[3]) * 16 != height:
                raise ValueError(
                    "h3_streaming_av: all H3 extension clips must use one resolution"
                )

        mode = str(start_mode)
        if mode == "existing_video":
            if source_frames is None:
                raise ValueError(
                    "h3_streaming_av: Existing Video start requires source frames"
                )
            source_idx = _cfr_index_map(
                int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS
            )
            base_frames = int(source_idx.numel())
            base_video_latent = None
            base_audio_latent = None
        else:
            if starter_latent is None:
                raise ValueError("h3_streaming_av: generated start requires starter_latent")
            base_video_latent, base_audio_latent = _streams_from_latent(starter_latent)
            base_frames = _pixel_frames(int(base_video_latent.shape[2]))
            source_idx = None
            if (
                int(base_video_latent.shape[4]) * 16 != width
                or int(base_video_latent.shape[3]) * 16 != height
            ):
                raise ValueError(
                    "h3_streaming_av: starter and extensions must use one resolution"
                )

        contexts = []
        available = base_frames
        for frames in raw_frames:
            ctx = _snap_av_context_length(int(context_frames), available, frames)
            contexts.append(ctx)
            available = frames

        final_frames = base_frames + sum(
            int(raw_frames[i]) - int(contexts[i]) for i in range(count)
        )
        audio = _assemble_av_audio(
            audio_vae,
            mode,
            base_frames,
            base_audio_latent,
            source_audio,
            ext_streams,
            raw_frames,
            contexts,
        )

        if mode == "existing_video":
            factory = lambda: _existing_base_and_generated_extensions_generator(
                video_vae,
                source_frames,
                source_idx,
                width,
                height,
                crop,
                ext_videos,
                raw_frames,
                contexts,
                video_overlap_frames,
            )
        else:
            videos = [base_video_latent] + ext_videos
            all_raw = [base_frames] + raw_frames
            all_contexts = [0] + contexts
            factory = lambda: _generated_frame_generator(
                video_vae,
                videos,
                all_raw,
                all_contexts,
                video_overlap_frames,
                "h3_streaming_av",
            )

        frames = _OneShotFrameSequence(final_frames, factory)
        max_hold = max(
            _seam_overlaps(
                [base_frames] + raw_frames,
                [0] + contexts,
                video_overlap_frames,
            )
        )
        _LOG.info(
            "h3_streaming_av: streaming %d frames from %s + %d connected extensions "
            "(slots %s) into VHS; old final RGB buffer %.2f GiB is not allocated; "
            "max retained seam %d frames (%.2f GiB)",
            final_frames,
            mode,
            count,
            ",".join(str(i) for i in extension_slots),
            _rgb_gib(final_frames, height, width),
            max_hold,
            _rgb_gib(max_hold, height, width),
        )
        return _run_vhs_h264(
            frames,
            audio,
            filename_prefix,
            pix_fmt,
            crf,
            save_metadata,
            trim_to_audio,
            save_output,
            prompt,
            extra_pnginfo,
            unique_id,
        )


class MiniMaxH3LastActiveVHSPreviewBarrier:
    """Resolve the highest enabled VHS preview before final output.

    Preview inputs are lazy and frontend-modular. Only the configured number of
    ``preview_N`` sockets is shown, while the backend still accepts a wider range
    for saved/custom workflows. The highest connected preview at or below both
    the configured count and optional controller cap is requested.
    """

    MAX_PREVIEWS = 64
    DEFAULT_PREVIEW_INPUTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "input_count": (
                "INT",
                {
                    "default": cls.DEFAULT_PREVIEW_INPUTS,
                    "min": 1,
                    "max": cls.MAX_PREVIEWS,
                    "step": 1,
                    "tooltip": (
                        "Number of preview sockets shown by the node. "
                        "Set this value, then click Update inputs."
                    ),
                },
            ),
        }
        optional = {
            "active_extensions": ("INT", {"forceInput": True}),
            "active_clips": ("INT", {"forceInput": True}),
        }
        # The browser extension hides sockets above input_count. Declaring the
        # full supported range keeps old/saved workflows backend-compatible.
        for i in range(1, cls.MAX_PREVIEWS + 1):
            optional[f"preview_{i}"] = ("VHS_FILENAMES", {"lazy": True})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("preview_gate",)
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Execution-order barrier for H3 workflows. Set Input Count and click "
        "Update inputs to show only the preview sockets you need. It waits for "
        "the highest enabled VHS preview before final assembly."
    )

    @classmethod
    def _limit(
        cls,
        input_count=DEFAULT_PREVIEW_INPUTS,
        active_extensions=None,
        active_clips=None,
    ):
        configured = max(1, min(cls.MAX_PREVIEWS, int(input_count)))
        active = active_clips if active_clips is not None else active_extensions
        if active is None:
            return configured
        return min(configured, max(0, int(active)))

    def check_lazy_status(
        self,
        input_count=DEFAULT_PREVIEW_INPUTS,
        active_extensions=None,
        active_clips=None,
        **kwargs,
    ):
        limit = self._limit(input_count, active_extensions, active_clips)
        for i in range(limit, 0, -1):
            name = f"preview_{i}"
            # Disconnected optional sockets are absent. A connected lazy preview
            # is present with value None until its VHS output has completed.
            if name in kwargs:
                return [name] if kwargs[name] is None else []
        return []

    def select(
        self,
        input_count=DEFAULT_PREVIEW_INPUTS,
        active_extensions=None,
        active_clips=None,
        **kwargs,
    ):
        limit = self._limit(input_count, active_extensions, active_clips)
        for i in range(limit, 0, -1):
            value = kwargs.get(f"preview_{i}")
            if value is not None:
                return (value,)
        # The downstream stream node treats this value only as an execution gate.
        # An empty VHS_FILENAMES payload is therefore a valid no-preview sentinel.
        return ([],)


class MiniMaxH3FinalizeVHSOutput:
    """Tiny terminal sink for direct-stream final outputs.

    ``filenames`` is optional deliberately: bypassing/removing the upstream final
    combine node leaves a valid no-op output node instead of a prompt-validation
    error. When connected, it simply keeps the streaming node in the executable
    graph.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"filenames": ("VHS_FILENAMES", {"lazy": True})}}

    RETURN_TYPES = ()
    FUNCTION = "finalize"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Optional execution sink for H3 direct-stream outputs. Safe when the "
        "upstream final stream node is bypassed or disconnected."
    )

    def check_lazy_status(self, **kwargs):
        # Disconnected optional sockets are absent; a connected lazy input is
        # present as None until the upstream final stream has completed.
        if "filenames" in kwargs and kwargs["filenames"] is None:
            return ["filenames"]
        return []

    def finalize(self, filenames=None):
        return ()


class MiniMaxH3StreamLiveMusicVideoToVHS:
    """Stream a modular Music Video timeline directly into VHS video output."""

    MAX_CLIPS = 64
    DEFAULT_CLIP_INPUTS = 20

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video_vae": ("VAE",),
            "master_audio": ("AUDIO",),
            "input_count": (
                "INT",
                {
                    "default": cls.DEFAULT_CLIP_INPUTS,
                    "min": 1,
                    "max": cls.MAX_CLIPS,
                    "step": 1,
                    "tooltip": (
                        "Number of clip latent sockets shown by the node. "
                        "Set this value, then click Update inputs."
                    ),
                },
            ),
            "context_frames": ("INT", {"default": 39, "min": 5, "max": 9999}),
            "video_overlap_frames": (
                "INT", {"default": 39, "min": 0, "max": 9999}
            ),
        }
        required.update(_vhs_h264_inputs("video/h3_music_video", False))
        optional = {
            # Optional execution-order gate. In the bundled Music Video workflow
            # this is fed by the highest enabled per-clip VHS preview.
            "preview_gate": ("VHS_FILENAMES", {"lazy": True}),
            # Optional controller cap. The bundled workflow mirrors Active Clips
            # into a cache-isolated PrimitiveInt and connects it here. Standalone
            # users can leave it disconnected and use the connected clip prefix.
            "active_clips": (
                "INT",
                {
                    "forceInput": True,
                    "tooltip": (
                        "Optional cap on clip sockets considered. Leave unconnected "
                        "for modular/custom workflows."
                    ),
                },
            ),
        }
        for i in range(1, cls.MAX_CLIPS + 1):
            optional[f"clip_{i}"] = ("LATENT", {"lazy": True})
        return {
            "required": required,
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "stream_to_vhs"
    OUTPUT_NODE = False
    HAS_INTERMEDIATE_OUTPUT = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Modular low-RAM final Music Video output. Set Input Count and click "
        "Update inputs to expose clip latent sockets. Standalone workflows may "
        "leave trailing sockets disconnected; connected clips must remain a "
        "contiguous prefix because each clip belongs to a fixed master-song window. "
        "Clips are decoded one at a time and the untouched master song is muxed."
    )

    @classmethod
    def _clip_limit(cls, input_count, active_clips=None):
        configured = max(1, min(cls.MAX_CLIPS, int(input_count)))
        if active_clips is None:
            return configured
        return min(configured, max(0, int(active_clips)))

    @classmethod
    def _connected_slots(cls, limit, kwargs):
        return [i for i in range(1, limit + 1) if f"clip_{i}" in kwargs]

    @classmethod
    def _validate_connected_prefix(cls, limit, active_clips, kwargs):
        slots = cls._connected_slots(limit, kwargs)
        if active_clips is not None:
            expected = list(range(1, limit + 1))
            missing = [i for i in expected if i not in slots]
            if missing:
                raise ValueError(
                    "h3_streaming_music: active clip inputs must be connected as a "
                    f"contiguous sequence; missing clip_{missing[0]} within active_clips={int(active_clips)}"
                )
            return expected
        if not slots:
            raise ValueError(
                "h3_streaming_music: connect at least one clip latent within the "
                f"first {limit} configured input socket(s)"
            )
        last = max(slots)
        expected = list(range(1, last + 1))
        missing = [i for i in expected if i not in slots]
        if missing:
            raise ValueError(
                "h3_streaming_music: connected clip inputs must form a contiguous "
                f"prefix from clip_1; missing clip_{missing[0]} before clip_{last}. "
                "Music Video clips cannot be compacted across timeline gaps."
            )
        return expected

    def check_lazy_status(
        self,
        video_vae,
        master_audio,
        input_count,
        context_frames,
        video_overlap_frames,
        filename_prefix,
        pix_fmt,
        crf,
        save_metadata,
        trim_to_audio,
        save_output,
        active_clips=None,
        **kwargs,
    ):
        # Resolve the final active clip preview before requesting any sampler
        # latents. This makes preview-before-final ordering deterministic.
        if "preview_gate" in kwargs and kwargs["preview_gate"] is None:
            return ["preview_gate"]

        limit = self._clip_limit(input_count, active_clips)
        # Only connected sockets appear in kwargs. When a controller cap is
        # present, all sockets inside that active prefix are expected to exist.
        if active_clips is not None:
            names = [f"clip_{i}" for i in range(1, limit + 1)]
        else:
            connected = self._connected_slots(limit, kwargs)
            names = [f"clip_{i}" for i in connected]
        return [name for name in names if name in kwargs and kwargs[name] is None]

    def stream_to_vhs(
        self,
        video_vae,
        master_audio,
        input_count=DEFAULT_CLIP_INPUTS,
        context_frames=39,
        video_overlap_frames=39,
        filename_prefix="video/h3_music_video",
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=False,
        trim_to_audio=False,
        save_output=True,
        preview_gate=None,
        active_clips=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs,
    ):
        limit = self._clip_limit(input_count, active_clips)
        slots = self._validate_connected_prefix(limit, active_clips, kwargs)
        latents = [kwargs[f"clip_{i}"] for i in slots]
        if any(value is None for value in latents):
            raise ValueError("h3_streaming_music: an active clip latent was not resolved")
        count = len(latents)

        streams = [_streams_from_latent(latent) for latent in latents]
        videos = [video for video, _audio in streams]
        raw_frames = [_pixel_frames(int(video.shape[2])) for video in videos]

        width = int(videos[0].shape[4]) * 16
        height = int(videos[0].shape[3]) * 16
        channels = int(videos[0].shape[1])
        for index, video in enumerate(videos[1:], start=2):
            if (
                int(video.shape[4]) * 16 != width
                or int(video.shape[3]) * 16 != height
                or int(video.shape[1]) != channels
            ):
                raise ValueError(
                    f"h3_streaming_music: Clip {index} resolution/latent channels do not match Clip 1"
                )

        contexts = [0]
        for i in range(1, count):
            contexts.append(
                _snap_music_context_length(
                    int(context_frames), raw_frames[i - 1], raw_frames[i]
                )
            )
        final_frames = raw_frames[0] + sum(
            raw_frames[i] - contexts[i] for i in range(1, count)
        )

        sample_rate = int(master_audio["sample_rate"])
        waveform = master_audio["waveform"]
        if getattr(waveform, "ndim", 0) != 3:
            raise ValueError(
                "h3_streaming_music: master_audio waveform must be [B,C,L], got %s"
                % (tuple(getattr(waveform, "shape", ())),)
            )
        audio = {
            "waveform": waveform[:1].detach().to(device="cpu").contiguous(),
            "sample_rate": sample_rate,
        }

        factory = lambda: _generated_frame_generator(
            video_vae,
            videos,
            raw_frames,
            contexts,
            video_overlap_frames,
            "h3_streaming_music",
        )
        frames = _OneShotFrameSequence(final_frames, factory)
        max_hold = max(_seam_overlaps(raw_frames, contexts, video_overlap_frames))
        _LOG.info(
            "h3_streaming_music: streaming %d clips / %d frames into VHS; untouched master song; "
            "old final RGB buffer %.2f GiB is not allocated; max retained seam %d frames (%.2f GiB)",
            count,
            final_frames,
            _rgb_gib(final_frames, height, width),
            max_hold,
            _rgb_gib(max_hold, height, width),
        )
        return _run_vhs_h264(
            frames,
            audio,
            filename_prefix,
            pix_fmt,
            crf,
            save_metadata,
            trim_to_audio,
            save_output,
            prompt,
            extra_pnginfo,
            unique_id,
        )
