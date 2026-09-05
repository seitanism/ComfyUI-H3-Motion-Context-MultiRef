"""MiniMax H3 music-video clip prep with exact master-song audio and live video context."""

import logging

import torch

from .h3_audio_grid import audio_grid_geometry, encode_exact_audio_grid

import comfy.nested_tensor
import comfy.utils

from .h3_timing import sample_boundary_from_seconds

try:
    from .h3_compat import ensure_existing_video_compat
except ImportError:
    ensure_existing_video_compat = lambda: True

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context.song_audio")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _require_h3_mask_support():
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
            "h3_song_audio: expected a MiniMax H3 AV latent, got %r" % type(samples)
        )
    if len(parts) < 2:
        raise ValueError("h3_song_audio: expected joint H3 video+audio latent streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_song_audio: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_song_audio: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _video_steps_for_frames(frame_count):
    """Return the exact H3 video-latent step count for a valid pixel-frame run."""
    frame_count = int(frame_count)
    total = 0
    for steps in range(1, 100000):
        total += FRAME_PER_TOKEN[(steps - 1) % len(FRAME_PER_TOKEN)]
        if total == frame_count:
            return steps
        if total > frame_count:
            break
    raise ValueError(
        "h3_song_audio: %d frames are not an exact H3 temporal run" % frame_count
    )


def _resize_images(images, width, height, crop, chunk=32):
    if int(images.shape[0]) <= chunk:
        x = images[..., :3].movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "lanczos", crop)
        return x.movedim(1, -1)
    out = []
    for start in range(0, int(images.shape[0]), chunk):
        part = images[start : start + chunk, ..., :3].movedim(-1, 1)
        part = comfy.utils.common_upscale(part, width, height, "lanczos", crop)
        out.append(part.movedim(1, -1))
    return torch.cat(out, dim=0)


def _cfr_index_map(frame_count, source_fps, device, target_fps=FPS):
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_song_audio: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_song_audio: source video has no frames")
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
            "h3_song_audio: %s waveform must be [B,C,L], got %s"
            % (label, tuple(getattr(waveform, "shape", ())))
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_song_audio: %s has %d channels. Downmix multichannel audio to stereo before this node."
        % (label, channels)
    )


def _resample_waveform(waveform, source_sr, target_sr, label):
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            "h3_song_audio: %s is %d Hz but %d Hz is required and torchaudio is unavailable"
            % (label, source_sr, target_sr)
        )
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _fit_waveform(waveform, want, label, pad=True):
    want = int(want)
    have = int(waveform.shape[-1])
    if have == want:
        return waveform
    if have > want:
        _LOG.info(
            "h3_song_audio: trimming %s by %d samples to exact timeline",
            label,
            have - want,
        )
        return waveform[..., :want]
    if not pad:
        raise ValueError(
            "h3_song_audio: %s is %d samples short of the video timeline" % (label, want - have)
        )
    _LOG.warning(
        "h3_song_audio: %s is %d samples short; padding silence at tail",
        label,
        want - have,
    )
    return torch.nn.functional.pad(waveform, (0, want - have))


def _largest_h3_video_run(n):
    n = int(n)
    if n < 5:
        return 0
    return ((n - 5) // 17) * 17 + 5


def _snap_context_length(requested, available, target_frames):
    cap = min(int(requested), int(available), int(target_frames) - 1)
    run = _largest_h3_video_run(cap)
    if run < 5:
        raise ValueError(
            "h3_song_audio: need at least 5 source frames and a target longer than the preserved prefix"
        )
    if run != int(requested):
        _LOG.warning(
            "h3_song_audio: context_length %d -> exact H3 prefix %d (valid runs are 5, 22, 39, 56, ...; 39/90/141/... align AV clocks)",
            int(requested), run,
        )
    return run


class MiniMaxH3SongMaskedAVContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target AV latent from MiniMax H3 Reference to Video."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE. Used to encode the master-song slice into the target audio latent."
                }),
                "master_audio": ("AUDIO", {
                    "tooltip": "Full master song or master vocal audio. The node writes the exact requested slice into the H3 latent and protects it from denoising."
                }),
                "clip_start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 99999.0, "step": 0.001,
                    "tooltip": "Start time of this raw clip within the master song timeline. For clip N, this usually advances by raw_clip_seconds - context_seconds from the previous clip."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Optional preserved visual prefix from the previous clip. Uses exact H3 runs 5/22/39/56/...; 0 disables visual prefixing entirely."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 24.0, "max": 24.0, "step": 0.001,
                    "tooltip": "FPS of source_frames when visual context is connected."
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE. Required only for the legacy source_frames continuation path."
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Legacy decoded-frame continuation path. Prefer source_latent for checkpoint-free live clip chaining."
                }),
                "source_latent": ("LATENT", {
                    "tooltip": "Preferred live continuation input. Copies the previous H3 video latent tail directly into the new target while the master-song audio slice remains authoritative."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "AUDIO")
    RETURN_NAMES = ("latent", "trim_frames", "clip_audio")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Write an exact master-song slice into the target H3 audio latent and protect the entire audio stream from denoising. For continuations, prefer a previous live H3 source_latent so the video tail is copied directly without decode/re-encode."
    )

    def prepare(
        self,
        latent,
        audio_vae,
        master_audio,
        clip_start_seconds=0.0,
        context_length=39,
        source_fps=24.0,
        crop="disabled",
        vae=None,
        source_frames=None,
        source_latent=None,
    ):
        _require_h3_mask_support()

        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError("h3_song_audio: batch size 1 only")

        target_frames = _pixel_frames(int(target_video.shape[2]))
        # The target latent is authoritative.  Native ComfyUI allocates the H3
        # audio stream by rounding the video duration onto the 40 Hz audio grid.
        # That grid can extend a fraction of one audio token past the final pixel
        # frame (e.g. 124f @ 24 fps = 206.666... ticks -> 207 target ticks).
        expected_audio_steps = int(target_audio.shape[-1])
        calculated_audio_steps = int(round(target_frames / FPS * AUDIO_HZ))
        if expected_audio_steps != calculated_audio_steps:
            _LOG.warning(
                "h3_song_audio: target latent has %d audio steps for %d video frames; "
                "the nominal 40 Hz calculation gives %d. Using the target latent length.",
                expected_audio_steps,
                target_frames,
                calculated_audio_steps,
            )

        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        waveform = _stereo_first_batch(master_audio["waveform"], "master_audio")
        waveform = _resample_waveform(waveform, int(master_audio["sample_rate"]), vae_sr, "master_audio")
        clip_start_seconds = float(clip_start_seconds)
        start_sample = sample_boundary_from_seconds(clip_start_seconds, vae_sr)
        if start_sample < 0:
            raise ValueError("h3_song_audio: clip_start_seconds must be >= 0")
        # Derive BOTH endpoints from the absolute master-song timeline.
        # Do not round the clip duration separately and add it to start_sample:
        # at rates such as 32 kHz / 24 fps that can move an endpoint by one
        # sample on some clip indices.  Absolute endpoints guarantee that the
        # conditioning slice occupies exactly the same timeline interval as
        # this raw video's pixel frames.
        picture_end_seconds = clip_start_seconds + target_frames / FPS
        picture_end_sample = sample_boundary_from_seconds(picture_end_seconds, vae_sr)
        picture_samples = picture_end_sample - start_sample
        if picture_samples < 1:
            raise RuntimeError("h3_song_audio: computed an empty master-song slice")
        audio_slice = waveform[..., start_sample:picture_end_sample]
        audio_slice = _fit_waveform(audio_slice, picture_samples, "master audio slice")

        # Encode on the target H3 audio grid, not the picture-duration PCM
        # span.  This keeps clip_start_seconds exact and makes ComfyUI's generic
        # VAE center-crop a no-op.  The rounded 40-Hz target can end slightly
        # before or after the last pixel-frame boundary.
        _grid_sr, samples_per_latent, grid_samples = audio_grid_geometry(
            audio_vae, expected_audio_steps
        )
        if _grid_sr != vae_sr:
            raise RuntimeError(
                "h3_song_audio: audio VAE grid reports %d Hz but audio_sample_rate is %d"
                % (_grid_sr, vae_sr)
            )
        encode_end_sample = start_sample + grid_samples
        encode_slice = waveform[..., start_sample:min(encode_end_sample, int(waveform.shape[-1]))]
        pcm_pad = grid_samples - int(encode_slice.shape[-1])
        if pcm_pad > 0:
            expected_overhang = max(0, grid_samples - picture_samples)
            tolerance = max(2, int(round(vae_sr * 0.001)))
            if pcm_pad > expected_overhang + tolerance:
                raise ValueError(
                    "h3_song_audio: master audio is %d samples short of the target H3 audio grid; "
                    "only %d samples are explained by picture-to-40-Hz endpoint rounding"
                    % (pcm_pad, expected_overhang)
                )
            encode_slice = torch.nn.functional.pad(encode_slice, (0, pcm_pad))

        audio_full, _grid_diag = encode_exact_audio_grid(
            audio_vae, encode_slice, expected_audio_steps, "h3_song_audio: master song"
        )
        _LOG.debug(
            "h3_song_audio: exact audio grid %d steps x %d samples; picture delta %+d samples; tail pad %d",
            expected_audio_steps,
            samples_per_latent,
            grid_samples - picture_samples,
            max(0, pcm_pad),
        )

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_audio.copy_(audio_full[:1].to(device=out_audio.device, dtype=out_audio.dtype))

        video_steps = 0
        n = 0
        if source_latent is not None and source_frames is not None:
            raise ValueError(
                "h3_song_audio: connect either source_latent or source_frames, not both"
            )

        if source_latent is not None:
            source_video, _source_audio = _streams_from_latent(source_latent)
            if int(source_video.shape[0]) != 1:
                raise ValueError("h3_song_audio: source_latent batch size 1 only")
            if int(context_length) <= 0:
                raise ValueError(
                    "h3_song_audio: context_length must be > 0 when source_latent is connected"
                )
            if (
                int(source_video.shape[4]) != int(target_video.shape[4])
                or int(source_video.shape[3]) != int(target_video.shape[3])
                or int(source_video.shape[1]) != int(target_video.shape[1])
            ):
                raise ValueError(
                    "h3_song_audio: source_latent and target latent video shapes/resolution must match"
                )

            available = _pixel_frames(int(source_video.shape[2]))
            n = _snap_context_length(context_length, available, target_frames)
            video_steps = _video_steps_for_frames(n)
            if video_steps >= int(target_video.shape[2]):
                raise ValueError("h3_song_audio: video context consumes the whole target latent")
            if video_steps > int(source_video.shape[2]):
                raise RuntimeError("h3_song_audio: source_latent is shorter than requested context")

            tail_start = int(source_video.shape[2]) - video_steps
            if tail_start % len(FRAME_PER_TOKEN) != 0:
                raise ValueError(
                    "h3_song_audio: source_latent tail begins at H3 temporal phase %d; "
                    "use an H3-safe source length or the source_frames fallback"
                    % (tail_start % len(FRAME_PER_TOKEN))
                )
            vp = source_video[:, :, tail_start:, :, :].to(
                device=out_video.device, dtype=out_video.dtype
            )
            out_video[:, :, :video_steps] = vp

        elif source_frames is not None:
            if vae is None:
                raise ValueError("h3_song_audio: vae is required when source_frames is connected")
            if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
                raise ValueError("h3_song_audio: source_frames must be IMAGE [N,H,W,C]")
            if int(context_length) <= 0:
                raise ValueError("h3_song_audio: context_length must be > 0 when source_frames is connected")
            idx = _cfr_index_map(int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS)
            available = int(idx.numel())
            n = _snap_context_length(context_length, available, target_frames)
            tail_idx = idx[-n:]
            video_tail = source_frames.index_select(0, tail_idx)
            video_tail = _resize_images(video_tail, width, height, crop)
            video_prefix = vae.encode(video_tail)
            if getattr(video_prefix, "ndim", 0) != 5:
                raise ValueError(
                    "h3_song_audio: video VAE returned %s, expected [B,C,T,H,W]"
                    % (tuple(getattr(video_prefix, "shape", ())),)
                )
            video_steps = int(video_prefix.shape[2])
            covered = _pixel_frames(video_steps)
            if covered != n:
                raise RuntimeError(
                    "h3_song_audio: %d context frames encoded to %d video latent steps covering %d frames; refusing a phase-shifted seam"
                    % (n, video_steps, covered)
                )
            if video_steps >= int(target_video.shape[2]):
                raise ValueError("h3_song_audio: video context consumes the whole target latent")
            vp = video_prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
            if tuple(vp.shape[1:2] + vp.shape[3:]) != tuple(out_video.shape[1:2] + out_video.shape[3:]):
                raise ValueError(
                    "h3_song_audio: encoded video prefix shape %s does not match target %s"
                    % (tuple(vp.shape), tuple(out_video.shape))
                )
            out_video[:, :, :video_steps] = vp

        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device,
            dtype=torch.float32,
        )
        if video_steps > 0:
            video_mask[:, :, :video_steps] = 0.0

        audio_mask = torch.zeros(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device,
            dtype=torch.float32,
        )

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        clip_audio = {"waveform": audio_slice, "sample_rate": vae_sr}
        _LOG.info(
            "h3_song_audio: target %d frames (%.3fs); exact song slice %.3fs -> %.3fs encoded to %d audio latent steps and fully preserved; visual prefix %d frames -> %d video steps",
            target_frames,
            target_frames / FPS,
            float(clip_start_seconds),
            float(clip_start_seconds) + target_frames / FPS,
            expected_audio_steps,
            n,
            video_steps,
        )
        return (out, n, clip_audio)
