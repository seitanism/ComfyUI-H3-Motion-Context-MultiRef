"""Two-ended MiniMax H3 AV bridge using PR #15375-style latent noise masks.

This node is intentionally small: it prepares the target latent only. Sampling is
still performed by normal ComfyUI sampler nodes. The first and last source
windows are VAE-encoded directly into the target H3 AV latent and protected by a
nested denoise mask (0 = preserve, 1 = generate). On ComfyUI builds without
native PR #15375 support, this repo's lazy compatibility layer is enabled only
when the node executes.
"""

from __future__ import annotations

import logging

import torch

from .h3_audio_grid import audio_grid_geometry, encode_exact_audio_grid


try:
    import torchaudio
except ImportError:
    torchaudio = None


_LOG = logging.getLogger("h3_motion_context.masked_bridge")
FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def largest_h3_video_run(frames):
    n = int(frames)
    if n < 5:
        return 0
    return 5 + ((n - 5) // 17) * 17


def _require_h3_mask_support():
    from .h3_compat import ensure_existing_video_compat
    # Native PR #15375 support wins, including the Aug-15+ no-preprocess-hook
    # architecture. The compatibility orchestrator validates the full path.
    ensure_existing_video_compat()


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
            "h3_masked_bridge: expected a MiniMax H3 AV latent, got %r" % type(samples)
        )
    if len(parts) < 2:
        raise ValueError("h3_masked_bridge: expected joint H3 video+audio latent streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_masked_bridge: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_masked_bridge: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _resize_images(images, width, height, crop, chunk=32):
    import comfy.utils
    if int(images.shape[0]) <= chunk:
        x = images[..., :3].movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "lanczos", crop)
        return x.movedim(1, -1)
    out = []
    for start in range(0, int(images.shape[0]), chunk):
        part = images[start:start + chunk, ..., :3].movedim(-1, 1)
        part = comfy.utils.common_upscale(part, width, height, "lanczos", crop)
        out.append(part.movedim(1, -1))
    return torch.cat(out, dim=0)


def _cfr_index_map(frame_count, source_fps, device, target_fps=FPS):
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_masked_bridge: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_masked_bridge: source video has no frames")
    out_n = max(1, int(round(n * float(target_fps) / source_fps)))
    if out_n == n and abs(source_fps - target_fps) < 1e-6:
        return torch.arange(n, device=device, dtype=torch.long)
    i = torch.arange(out_n, device=device, dtype=torch.float64)
    t = (i + 0.5) / float(target_fps)
    src = torch.round(t * source_fps - 0.5).to(torch.long)
    return src.clamp_(0, n - 1)


def _stereo_first_batch(waveform, label):
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_masked_bridge: %s waveform must be [B,C,L], got %s"
            % (label, tuple(getattr(waveform, "shape", ())))
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_masked_bridge: %s has %d channels; downmix to stereo first"
        % (label, channels)
    )


def _resample_waveform(waveform, source_sr, target_sr, label):
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            "h3_masked_bridge: %s is %d Hz but %d Hz is required and torchaudio is unavailable"
            % (label, source_sr, target_sr)
        )
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _canonical_audio(audio, target_sr, frame_count, label):
    if audio is None:
        raise ValueError("h3_masked_bridge: %s is required" % label)
    waveform = _stereo_first_batch(audio["waveform"], label)
    waveform = _resample_waveform(
        waveform, int(audio["sample_rate"]), int(target_sr), label
    )
    want = int(round(int(frame_count) / FPS * int(target_sr)))
    have = int(waveform.shape[-1])
    if have > want:
        waveform = waveform[..., :want]
    elif have < want:
        waveform = torch.nn.functional.pad(waveform, (0, want - have))
    return {"waveform": waveform, "sample_rate": int(target_sr)}


def _validate_preserve_frames(requested, start_available, end_available, target_frames):
    n = int(requested)
    if n < 39 or (n - 39) % 51 != 0:
        raise ValueError("h3_masked_bridge: preserve_frames must be a shared AV boundary: 39, 90, 141, 192, ... (39 + 51*k)")
    if largest_h3_video_run(n) != n:
        raise ValueError(
            "h3_masked_bridge: preserve_frames must be an exact H3 video run "
            "(5, 22, 39, 56, ...); got %d" % n
        )
    if int(start_available) < n:
        raise ValueError(
            "h3_masked_bridge: start clip has only %d canonical 24fps frames; need %d"
            % (int(start_available), n)
        )
    if int(end_available) < n:
        raise ValueError(
            "h3_masked_bridge: end clip has only %d canonical 24fps frames; need %d"
            % (int(end_available), n)
        )
    if 2 * n >= int(target_frames):
        raise ValueError(
            "h3_masked_bridge: target must contain a non-empty generated middle; "
            "target=%d, preserved=%d+%d" % (int(target_frames), n, n)
        )
    return n


def _encode_video_window(vae, frames, width, height, crop, n, label):
    resized = _resize_images(frames, width, height, crop)
    encoded = vae.encode(resized)
    if getattr(encoded, "ndim", 0) != 5:
        raise ValueError(
            "h3_masked_bridge: %s video VAE returned %s; expected [B,C,T,H,W]"
            % (label, tuple(getattr(encoded, "shape", ())))
        )
    steps = int(encoded.shape[2])
    covered = _pixel_frames(steps)
    if covered != int(n):
        raise RuntimeError(
            "h3_masked_bridge: %s %d-frame window encoded to %d video latent "
            "steps covering %d frames; refusing a phase-shifted seam"
            % (label, int(n), steps, covered)
        )
    return encoded[:1], steps


def _encode_audio_window(audio_vae, canonical_audio, n, side, label):
    waveform = canonical_audio["waveform"]
    exact_steps = int(n) / FPS * AUDIO_HZ
    expected = int(round(exact_steps))
    if abs(exact_steps - expected) > 1e-9:
        _LOG.warning(
            "h3_masked_bridge: %d preserved frames end between H3 audio ticks "
            "(%.6f -> %d steps); using a seam-aligned exact audio-grid window",
            int(n), exact_steps, expected,
        )

    _sr, samples_per_latent, grid_samples = audio_grid_geometry(audio_vae, expected)
    if int(waveform.shape[-1]) < grid_samples:
        raise ValueError(
            "h3_masked_bridge: %s audio is shorter than the %d-step H3 audio-grid window"
            % (label, expected)
        )

    # The bridge seam is the authoritative endpoint: the start clip's tail must
    # end exactly at the generated middle, while the end clip's head must start
    # exactly there.  For non-shared 24/40-Hz boundaries this intentionally
    # shifts only the *outer* edge by <= half an audio tick instead of allowing
    # Comfy's generic VAE center-crop to shift the seam itself.
    if side == "tail":
        window = waveform[..., -grid_samples:]
    elif side == "head":
        window = waveform[..., :grid_samples]
    else:
        raise ValueError("h3_masked_bridge: internal invalid audio side %r" % side)

    encoded, _diag = encode_exact_audio_grid(
        audio_vae, window, expected, "h3_masked_bridge: %s" % label
    )
    _LOG.debug(
        "h3_masked_bridge: %s exact grid %d steps x %d samples (%s-aligned)",
        label, expected, samples_per_latent, side,
    )
    return encoded[:1], expected


class MiniMaxH3MaskedAVBridge:
    """Freeze AV windows at both ends of an H3 target; denoise only the middle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target MiniMax H3 AV latent. For example, use a 192-frame H3 latent."
                }),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "start_frames": ("IMAGE", {
                    "tooltip": "Full first source clip or a frame batch containing its ending. The final preserve_frames are frozen at the bridge start."
                }),
                "start_audio": ("AUDIO", {
                    "tooltip": "Audio matching start_frames. The final preserved interval is frozen at the bridge start."
                }),
                "end_frames": ("IMAGE", {
                    "tooltip": "Full second source clip or a frame batch containing its beginning. The first preserve_frames are frozen at the bridge end."
                }),
                "end_audio": ("AUDIO", {
                    "tooltip": "Audio matching end_frames. The first preserved interval is frozen at the bridge end."
                }),
                "start_fps": ("FLOAT", {
                    "default": 24.0, "min": 24.0, "max": 24.0, "step": 0.001,
                    "tooltip": "Frame rate represented by start_frames. CFR input is converted deterministically to H3 24 fps."
                }),
                "end_fps": ("FLOAT", {
                    "default": 24.0, "min": 24.0, "max": 24.0, "step": 0.001,
                    "tooltip": "Frame rate represented by end_frames. CFR input is converted deterministically to H3 24 fps."
                }),
                "preserve_frames": ("INT", {
                    "default": 39, "min": 39, "max": 9984, "step": 51,
                    "tooltip": "Shared AV boundaries only: 39, 90, 141, 192, ... (39 + 51*k). The target must leave a generated middle."
                }),
                "crop": (["disabled", "center"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "middle_frames", "preserve_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Create a true two-ended H3 AV masked bridge. The ending of the first "
        "clip and beginning of the second clip are encoded into the target latent "
        "and protected with PR #15375-style video+audio denoise masks; only the "
        "middle is generated. 0 = preserve, 1 = denoise."
    )

    def prepare(
        self,
        latent,
        vae,
        audio_vae,
        start_frames,
        start_audio,
        end_frames,
        end_audio,
        start_fps=24.0,
        end_fps=24.0,
        preserve_frames=39,
        crop="center",
    ):
        if float(start_fps) != FPS or float(end_fps) != FPS:
            raise ValueError("H3 AV Bridge requires 24 fps sources; convert both videos to 24 fps first.")
        _require_h3_mask_support()

        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError("h3_masked_bridge: MiniMax H3 bridge currently supports batch size 1")

        target_frames = _pixel_frames(int(target_video.shape[2]))
        if largest_h3_video_run(target_frames) != target_frames:
            raise RuntimeError(
                "h3_masked_bridge: target latent covers %d frames, which is not an exact "
                "H3 video run (5, 22, 39, 56, ...)" % target_frames
            )
        expected_target_audio = int(round(target_frames / FPS * AUDIO_HZ))
        if int(target_audio.shape[-1]) != expected_target_audio:
            raise RuntimeError(
                "h3_masked_bridge: target has %d audio steps for %d frames; expected %d"
                % (int(target_audio.shape[-1]), target_frames, expected_target_audio)
            )

        if getattr(start_frames, "ndim", 0) != 4 or int(start_frames.shape[0]) < 1:
            raise ValueError("h3_masked_bridge: start_frames must be IMAGE [N,H,W,C]")
        if getattr(end_frames, "ndim", 0) != 4 or int(end_frames.shape[0]) < 1:
            raise ValueError("h3_masked_bridge: end_frames must be IMAGE [N,H,W,C]")

        start_idx = _cfr_index_map(int(start_frames.shape[0]), float(start_fps), start_frames.device, FPS)
        end_idx = _cfr_index_map(int(end_frames.shape[0]), float(end_fps), end_frames.device, FPS)
        n = _validate_preserve_frames(
            preserve_frames, int(start_idx.numel()), int(end_idx.numel()), target_frames
        )

        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16
        start_tail = start_frames.index_select(0, start_idx[-n:])
        end_head = end_frames.index_select(0, end_idx[:n])

        start_video, start_vsteps = _encode_video_window(
            vae, start_tail, width, height, crop, n, "start tail"
        )
        end_video, end_vsteps = _encode_video_window(
            vae, end_head, width, height, crop, n, "end head"
        )
        if start_vsteps != end_vsteps:
            raise RuntimeError(
                "h3_masked_bridge: start/end video windows encoded to different lengths (%d vs %d)"
                % (start_vsteps, end_vsteps)
            )
        if start_vsteps * 2 >= int(target_video.shape[2]):
            raise ValueError("h3_masked_bridge: preserved video windows consume the whole target")

        # A valid H3 target and valid H3 preserved run should automatically have
        # the same temporal VAE phase at the suffix. Keep this assertion explicit:
        # it is what makes a standalone encoded 39-frame head safe at the end.
        suffix_video_step = int(target_video.shape[2]) - end_vsteps
        if suffix_video_step % 5 != 0:
            raise RuntimeError(
                "h3_masked_bridge: suffix begins at video latent step %d (phase %d); "
                "refusing to place a standalone encoded suffix out of H3 temporal phase"
                % (suffix_video_step, suffix_video_step % 5)
            )

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        start_canonical = _canonical_audio(start_audio, vae_sr, int(start_idx.numel()), "start_audio")
        end_canonical = _canonical_audio(end_audio, vae_sr, int(end_idx.numel()), "end_audio")
        start_audio_lat, start_asteps = _encode_audio_window(
            audio_vae, start_canonical, n, "tail", "start tail"
        )
        end_audio_lat, end_asteps = _encode_audio_window(
            audio_vae, end_canonical, n, "head", "end head"
        )
        if start_asteps != end_asteps:
            raise RuntimeError(
                "h3_masked_bridge: start/end audio windows encoded to different lengths (%d vs %d)"
                % (start_asteps, end_asteps)
            )
        if start_asteps * 2 >= int(target_audio.shape[-1]):
            raise ValueError("h3_masked_bridge: preserved audio windows consume the whole target")

        out_video = target_video.clone()
        out_audio = target_audio.clone()

        sv = start_video.to(device=out_video.device, dtype=out_video.dtype)
        ev = end_video.to(device=out_video.device, dtype=out_video.dtype)
        sa = start_audio_lat.to(device=out_audio.device, dtype=out_audio.dtype)
        ea = end_audio_lat.to(device=out_audio.device, dtype=out_audio.dtype)

        if tuple(sv.shape[1:2] + sv.shape[3:]) != tuple(out_video.shape[1:2] + out_video.shape[3:]):
            raise ValueError(
                "h3_masked_bridge: start video latent shape %s does not match target %s"
                % (tuple(sv.shape), tuple(out_video.shape))
            )
        if tuple(ev.shape[1:2] + ev.shape[3:]) != tuple(out_video.shape[1:2] + out_video.shape[3:]):
            raise ValueError(
                "h3_masked_bridge: end video latent shape %s does not match target %s"
                % (tuple(ev.shape), tuple(out_video.shape))
            )
        if tuple(sa.shape[1:3]) != tuple(out_audio.shape[1:3]):
            raise ValueError(
                "h3_masked_bridge: start audio latent shape %s does not match target %s"
                % (tuple(sa.shape), tuple(out_audio.shape))
            )
        if tuple(ea.shape[1:3]) != tuple(out_audio.shape[1:3]):
            raise ValueError(
                "h3_masked_bridge: end audio latent shape %s does not match target %s"
                % (tuple(ea.shape), tuple(out_audio.shape))
            )

        out_video[:, :, :start_vsteps] = sv
        out_video[:, :, -end_vsteps:] = ev
        out_audio[..., :start_asteps] = sa
        out_audio[..., -end_asteps:] = ea

        # PR #15375 semantics: 0 = preserve existing latent, 1 = denoise/generate.
        # The compatibility layer/native H3 implementation snaps video masks to
        # 2x2 latent patches and audio masks to audio latent frames.
        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device,
            dtype=torch.float32,
        )
        audio_mask = torch.ones(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device,
            dtype=torch.float32,
        )
        video_mask[:, :, :start_vsteps] = 0.0
        video_mask[:, :, -end_vsteps:] = 0.0
        audio_mask[..., :start_asteps] = 0.0
        audio_mask[..., -end_asteps:] = 0.0

        import comfy.nested_tensor
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        middle_frames = target_frames - 2 * n
        _LOG.info(
            "h3_masked_bridge: target %d frames @ %dx%d; preserve %d frames at each end "
            "(%d video steps / %d audio steps each); generate middle %d frames (%.3fs)",
            target_frames, width, height, n, start_vsteps, start_asteps,
            middle_frames, middle_frames / FPS,
        )
        return (out, middle_frames, n)
