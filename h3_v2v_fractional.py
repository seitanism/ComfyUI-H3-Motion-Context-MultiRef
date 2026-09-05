"""Native MiniMax H3 V2V source latent with fractional per-stream denoise masks.

The node writes the full source video/audio into an existing H3 AV target latent
and controls V2V strength through H3's native denoise-mask path.  It never
changes sampler sigmas; BasicScheduler denoise remains 1.0.
"""

from __future__ import annotations

import json
import logging
import math

import torch

from .h3_audio_grid import audio_grid_geometry, encode_exact_audio_grid

try:
    import torchaudio
except ImportError:
    torchaudio = None


_LOG = logging.getLogger("h3_motion_context.v2v_fractional")
FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_v2v_fractional: expected a MiniMax H3 AV latent, got %r" % type(samples)
        )
    if len(parts) < 2:
        raise ValueError("h3_v2v_fractional: expected joint H3 video+audio latent streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "h3_v2v_fractional: expected video [B,C,T,H,W] and audio [B,C,2,T], got %s / %s"
            % (tuple(video.shape), tuple(audio.shape))
        )
    return video, audio


def _cfr_index_map(frame_count, source_fps, device, target_fps=FPS):
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_v2v_fractional: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_v2v_fractional: source video has no frames")
    out_n = max(1, int(round(n * float(target_fps) / source_fps)))
    if out_n == n and abs(source_fps - target_fps) < 1e-6:
        return torch.arange(n, device=device, dtype=torch.long)
    i = torch.arange(out_n, device=device, dtype=torch.float64)
    t = (i + 0.5) / float(target_fps)
    src = torch.round(t * source_fps - 0.5).to(torch.long)
    return src.clamp_(0, n - 1)


def _fit_start(available, wanted, mode):
    available = int(available)
    wanted = int(wanted)
    if available < wanted:
        raise ValueError(
            "h3_v2v_fractional: source becomes %d frames at 24 fps but target needs %d"
            % (available, wanted)
        )
    extra = available - wanted
    if mode == "end":
        return extra
    if mode == "center":
        return extra // 2
    return 0


def _resize_images(images, width, height, crop):
    import comfy.utils
    x = images[..., :3].movedim(-1, 1)
    x = comfy.utils.common_upscale(x, int(width), int(height), "lanczos", crop)
    return x.movedim(1, -1)


def _stereo_first_batch(waveform):
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_v2v_fractional: source_audio waveform must be [B,C,L], got %s"
            % (tuple(getattr(waveform, "shape", ())),)
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_v2v_fractional: source_audio has %d channels; downmix to stereo first"
        % channels
    )


def _audio_grid_geometry(audio_vae, target_audio_steps):
    # Backward-compatible local wrapper for tests/diagnostics; the geometry is
    # shared by every repo node that encodes H3 audio.
    return audio_grid_geometry(audio_vae, target_audio_steps)


def _audio_slice_for_target_grid(
    source_audio, audio_vae, start_frame, frame_count, target_audio_steps
):
    """Slice source audio on the target H3 40 Hz grid, not picture duration.

    Video ends at a 24 Hz frame boundary while H3 audio ends at the nearest
    40 Hz latent boundary.  Those endpoints differ by up to half an audio
    token.  The target AV latent is authoritative, so use exactly
    ``target_audio_steps`` VAE cells.  If the rounded audio grid extends a few
    milliseconds beyond an otherwise exact source clip, zero-pad PCM *before*
    encoding; this is the same boundary condition the H3 AudioVAE documents
    for its own right-padding and keeps the final latent data-derived rather
    than appending an all-zero latent token.
    """
    waveform = _stereo_first_batch(source_audio["waveform"])
    source_sr = int(source_audio["sample_rate"])
    vae_sr, samples_per_latent, grid_samples = _audio_grid_geometry(
        audio_vae, target_audio_steps
    )
    if source_sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_v2v_fractional: audio is %d Hz but the audio VAE wants %d Hz and torchaudio is unavailable"
                % (source_sr, vae_sr)
            )
        waveform = torchaudio.functional.resample(waveform, source_sr, vae_sr)

    start_sample = int(round(float(start_frame) / FPS * vae_sr))
    picture_end_sample = int(round(float(start_frame + frame_count) / FPS * vae_sr))
    picture_samples = picture_end_sample - start_sample
    end_sample = start_sample + grid_samples
    if start_sample >= int(waveform.shape[-1]):
        raise ValueError("h3_v2v_fractional: source audio does not reach the selected source-video interval")

    sliced = waveform[..., start_sample:min(end_sample, int(waveform.shape[-1]))]
    pcm_pad_samples = grid_samples - int(sliced.shape[-1])
    if pcm_pad_samples > 0:
        # H3 rounds video duration to the nearest 40 Hz audio cell.  At an
        # otherwise exact source-video endpoint this can put the AV grid a
        # fraction of one 800-sample cell beyond the available soundtrack.
        # Permit only that mathematically expected endpoint overhang (plus a
        # 1 ms media/resampling tolerance).  Anything larger is genuine missing
        # source audio and should not be hidden by padding.
        expected_grid_overhang = max(0, grid_samples - picture_samples)
        media_tolerance = max(2, int(round(vae_sr * 0.001)))
        if pcm_pad_samples > expected_grid_overhang + media_tolerance:
            raise ValueError(
                "h3_v2v_fractional: source audio is %d samples short of the target H3 audio grid; "
                "only %d samples are explained by video-to-40-Hz endpoint rounding"
                % (pcm_pad_samples, expected_grid_overhang)
            )
        sliced = torch.nn.functional.pad(sliced, (0, pcm_pad_samples))

    if int(sliced.shape[-1]) != grid_samples:
        raise RuntimeError(
            "h3_v2v_fractional: internal audio-grid slice mismatch %d != %d samples"
            % (int(sliced.shape[-1]), grid_samples)
        )

    return sliced, {
        "vae_sample_rate": vae_sr,
        "samples_per_latent": samples_per_latent,
        "picture_samples": picture_samples,
        "grid_samples": grid_samples,
        "grid_minus_picture_samples": grid_samples - picture_samples,
        "pcm_tail_pad_samples": max(0, pcm_pad_samples),
    }


def _fit_mask_frames(source_mask, source_frame_count, canonical_indices, fit_start, target_frames):
    if source_mask is None:
        return None
    mask = source_mask
    if getattr(mask, "ndim", 0) == 4 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if getattr(mask, "ndim", 0) != 3:
        raise ValueError(
            "h3_v2v_fractional: source_mask must be MASK [N,H,W], got %s"
            % (tuple(getattr(mask, "shape", ())),)
        )
    n = int(mask.shape[0])
    if n == 1:
        return mask.expand(int(target_frames), -1, -1)
    if n == int(source_frame_count):
        selected = canonical_indices[int(fit_start):int(fit_start) + int(target_frames)]
        return mask.index_select(0, selected.to(mask.device))
    if n == int(canonical_indices.numel()):
        return mask[int(fit_start):int(fit_start) + int(target_frames)]
    if n == int(target_frames):
        return mask
    raise ValueError(
        "h3_v2v_fractional: source_mask has %d frames; expected 1, source-frame count %d, canonical count %d, or target count %d"
        % (n, int(source_frame_count), int(canonical_indices.numel()), int(target_frames))
    )


def _mask_to_video_latent(mask_frames, latent_t, latent_h, latent_w, reduce_mode, crop="disabled"):
    # Match common_upscale center-crop coordinates before mask interpolation.
    if crop == "center":
        old_h, old_w = mask_frames.shape[-2:]
        aspect = float(latent_w) / float(latent_h)
        x = round((old_w - old_h * aspect) / 2) if old_w / old_h > aspect else 0
        y = round((old_h - old_w / aspect) / 2) if old_w / old_h < aspect else 0
        mask_frames = mask_frames[..., y:old_h-y, x:old_w-x]
    # Resize each source-frame mask to the video-latent grid first, then collapse
    # each H3 VAE temporal run (1,4,4,4,4...) to one latent step.
    m = mask_frames.to(dtype=torch.float32).unsqueeze(1)
    m = torch.nn.functional.interpolate(
        m,
        size=(int(latent_h), int(latent_w)),
        mode="bilinear",
        align_corners=False,
    )[:, 0].clamp_(0.0, 1.0)
    rows = []
    off = 0
    for k in range(int(latent_t)):
        span = FRAME_PER_TOKEN[k % 5]
        chunk = m[off:off + span]
        if int(chunk.shape[0]) != span:
            raise RuntimeError("h3_v2v_fractional: mask/video temporal grid mismatch")
        if reduce_mode == "mean":
            row = chunk.mean(dim=0)
        elif reduce_mode == "min":
            row = chunk.amin(dim=0)
        else:
            row = chunk.amax(dim=0)
        rows.append(row)
        off += span
    return torch.stack(rows, dim=0).unsqueeze(0).unsqueeze(0)


def _base_h3_model(model):
    base = getattr(model, "model", None)
    if base is None:
        base = getattr(model, "inner_model", None)
    if base is None:
        raise ValueError("h3_v2v_fractional: MODEL does not expose a Comfy BaseModel")
    return base


class H3V2VGranularFractionalDenoise:
    """Encode source AV into H3 and apply granular fractional denoise-mask strengths."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target MiniMax H3 AV latent whose exact video/audio shape is reused."
                }),
                "model": ("MODEL", {
                    "tooltip": "MiniMax H3 model. The precision compatibility probe/patch is applied lazily only when this node executes."
                }),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "source_frames": ("IMAGE", {
                    "tooltip": "Full source video. It is converted deterministically to H3's 24 fps and encoded over the entire target latent."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "Source audio matching source_frames. audio_strength=0 preserves it."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 24.0, "max": 24.0, "step": 0.001,
                }),
                "mode": (["global", "spatial"], {"default": "global"}),
                "global_strength": ("FLOAT", {
                    "default": 0.9995, "min": 0.0, "max": 1.0, "step": 0.00001,
                    "tooltip": "Native H3 video denoise-mask strength. Keep BasicScheduler denoise at 1.0."
                }),
                "inside_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.00001,
                }),
                "outside_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.00001,
                }),
                "audio_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.00001,
                    "tooltip": "0 preserves source audio; 1 fully denoises/regenerates audio."
                }),
                "source_fit": (["start", "center", "end"], {"default": "start"}),
                "crop": (["disabled", "center"], {"default": "center"}),
                "mask_temporal_reduce": (["max", "mean", "min"], {"default": "max"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "source_mask": ("MASK", {
                    "tooltip": "Spatial V2V mask for mode=spatial. 1 uses inside_strength; 0 uses outside_strength."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MODEL", "STRING")
    RETURN_NAMES = ("latent", "model", "diagnostics")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Full-source MiniMax H3 V2V latent with granular fractional video/audio denoise masks. "
        "Mask values can express continuous preserve-to-generate strength instead of only binary masking. "
        "V2V strength lives in the H3 mask; BasicScheduler denoise must remain 1.0. "
        "Near-1 mask precision compatibility is installed lazily and self-retires when ComfyUI provides it natively."
    )

    def prepare(
        self,
        latent,
        model,
        vae,
        audio_vae,
        source_frames,
        source_audio,
        source_fps=24.0,
        mode="global",
        global_strength=0.9995,
        inside_strength=1.0,
        outside_strength=0.0,
        audio_strength=0.0,
        source_fit="start",
        crop="center",
        mask_temporal_reduce="max",
        invert_mask=False,
        source_mask=None,
    ):
        # Nothing in h3_mask_precision is imported until this node actually runs.
        from .h3_mask_precision import capability_status, ensure_h3_mask_precision

        precision_before = capability_status()
        precision_after = ensure_h3_mask_precision()

        import comfy.nested_tensor
        import comfy.model_base as model_base

        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError("h3_v2v_fractional: MiniMax H3 V2V currently supports batch size 1")

        base = _base_h3_model(model)
        if not isinstance(base, getattr(model_base, "MiniMaxH3")):
            raise ValueError(
                "h3_v2v_fractional: connected MODEL is not MiniMax H3 (%r)" % type(base)
            )

        if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
            raise ValueError("h3_v2v_fractional: source_frames must be IMAGE [N,H,W,C]")

        target_frames = _pixel_frames(int(target_video.shape[2]))
        expected_audio_steps = int(round(target_frames / FPS * AUDIO_HZ))
        if int(target_audio.shape[-1]) != expected_audio_steps:
            raise RuntimeError(
                "h3_v2v_fractional: target has %d audio steps for %d frames; expected %d"
                % (int(target_audio.shape[-1]), target_frames, expected_audio_steps)
            )

        canonical = _cfr_index_map(
            int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS
        )
        fit_start = _fit_start(int(canonical.numel()), target_frames, str(source_fit))
        selected_indices = canonical[fit_start:fit_start + target_frames]
        selected_frames = source_frames.index_select(0, selected_indices)

        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16
        resized = _resize_images(selected_frames, width, height, str(crop))
        source_video_latent = vae.encode(resized)
        if getattr(source_video_latent, "ndim", 0) != 5:
            raise ValueError(
                "h3_v2v_fractional: video VAE returned %s; expected [B,C,T,H,W]"
                % (tuple(getattr(source_video_latent, "shape", ())),)
            )
        if tuple(source_video_latent.shape[1:]) != tuple(target_video.shape[1:]):
            raise RuntimeError(
                "h3_v2v_fractional: source video encoded to %s but target video latent is %s"
                % (tuple(source_video_latent.shape), tuple(target_video.shape))
            )

        target_audio_steps = int(target_audio.shape[-1])
        audio_pcm, audio_grid = _audio_slice_for_target_grid(
            source_audio, audio_vae, fit_start, target_frames, target_audio_steps
        )
        source_audio_latent, _encoded_grid = encode_exact_audio_grid(
            audio_vae, audio_pcm, target_audio_steps, "h3_v2v_fractional: source audio"
        )
        if tuple(source_audio_latent.shape[1:3]) != tuple(target_audio.shape[1:3]):
            raise RuntimeError(
                "h3_v2v_fractional: source audio latent shape %s does not match target %s"
                % (tuple(source_audio_latent.shape), tuple(target_audio.shape))
            )

        out_video = source_video_latent[:1].to(device=target_video.device, dtype=target_video.dtype)
        out_audio = source_audio_latent[:1].to(device=target_audio.device, dtype=target_audio.dtype)

        if str(mode) == "spatial":
            fitted_mask = _fit_mask_frames(
                source_mask,
                int(source_frames.shape[0]),
                canonical,
                fit_start,
                target_frames,
            )
            if fitted_mask is None:
                raise ValueError("h3_v2v_fractional: mode=spatial requires source_mask")
            if bool(invert_mask):
                fitted_mask = 1.0 - fitted_mask
            spatial = _mask_to_video_latent(
                fitted_mask,
                int(target_video.shape[2]),
                int(target_video.shape[3]),
                int(target_video.shape[4]),
                str(mask_temporal_reduce),
                str(crop),
            ).to(device=out_video.device, dtype=torch.float32)
            inside = float(inside_strength)
            outside = float(outside_strength)
            video_mask = (outside + spatial * (inside - outside)).clamp_(0.0, 1.0)
        else:
            video_mask = torch.full(
                (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
                float(global_strength),
                device=out_video.device,
                dtype=torch.float32,
            )

        audio_mask = torch.full(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            float(audio_strength),
            device=out_audio.device,
            dtype=torch.float32,
        )

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        # Diagnostics use the *live* post-probe H3 token-grid implementation, so
        # a future native implementation with equivalent or better precision is
        # reported directly rather than being forced back onto our 1/4096 grid.
        import comfy.utils
        packed_mask, mask_shapes = comfy.utils.pack_latents((video_mask, audio_mask))
        q_video, q_audio = base._token_grid_masks(packed_mask, mask_shapes)
        diagnostics = {
            "node": "H3V2VGranularFractionalDenoise",
            "scheduler_requirement": "BasicScheduler denoise must remain 1.0",
            "source_fps": float(source_fps),
            "canonical_24fps_frames": int(canonical.numel()),
            "target_frames": int(target_frames),
            "source_fit": str(source_fit),
            "fit_start_frame_24fps": int(fit_start),
            "mode": str(mode),
            "requested_video_strength": float(global_strength) if str(mode) == "global" else None,
            "video_mask_min": float(video_mask.amin().item()),
            "video_mask_max": float(video_mask.amax().item()),
            "token_grid_video_min": float(q_video.amin().item()),
            "token_grid_video_max": float(q_video.amax().item()),
            "requested_audio_strength": float(audio_strength),
            "token_grid_audio_min": float(q_audio.amin().item()),
            "token_grid_audio_max": float(q_audio.amax().item()),
            "video_mask_dtype": str(video_mask.dtype),
            "audio_mask_dtype": str(audio_mask.dtype),
            "audio_grid_samples": int(audio_grid["grid_samples"]),
            "audio_samples_per_latent": int(audio_grid["samples_per_latent"]),
            "audio_grid_minus_picture_samples": int(audio_grid["grid_minus_picture_samples"]),
            "audio_pcm_tail_pad_samples": int(audio_grid["pcm_tail_pad_samples"]),
            "precision_before": precision_before,
            "precision_after": precision_after,
        }
        _LOG.info(
            "h3_v2v_fractional: requested video %.8f -> token-grid [%.8f, %.8f], audio %.8f; BasicScheduler denoise stays 1.0",
            float(global_strength) if str(mode) == "global" else float(video_mask.mean().item()),
            float(q_video.amin().item()),
            float(q_video.amax().item()),
            float(audio_strength),
        )
        return (out, model, json.dumps(diagnostics, indent=2, sort_keys=True))
