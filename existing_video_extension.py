"""MiniMax H3 existing-video continuation via preserved AV target-prefix masks.

The node uses native ComfyUI H3 AV-mask support when available. On older
ComfyUI builds, Update 2 installs only the missing runtime compatibility pieces
needed from ComfyUI PR #15375. No ComfyUI source files are modified on disk.
"""

import gc
import logging
from fractions import Fraction

import torch

import comfy.nested_tensor
import comfy.utils

from .h3_compat import ensure_existing_video_compat
from .h3_timing import largest_h3_video_run, is_exact_av_boundary

try:
    import torchaudio
except ImportError:
    torchaudio = None


_LOG = logging.getLogger("h3_motion_context.masked_extension")


def _require_h3_mask_support():
    # Lazy, capability-aware compatibility. Native PR #15375 support wins,
    # including the Aug-15+ design that no longer defines process_denoise_mask.
    # ensure_existing_video_compat() validates both the model engine and payload
    # path and raises a capability report if either side is incomplete.
    ensure_existing_video_compat()


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
            "h3_masked_extension: expected a MiniMax H3 AV latent, got %r"
            % type(samples)
        )
    if len(parts) < 2:
        raise ValueError(
            "h3_masked_extension: expected joint H3 video+audio latent streams"
        )
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_masked_extension: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_masked_extension: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _resize_images(images, width, height, crop, chunk=32):
    """Resize a video batch without asking common_upscale to hold it all at once."""
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
    """Map a CFR source onto target-fps frame centers without touching pixels."""
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_masked_extension: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_masked_extension: source video has no frames")
    out_n = max(1, int(round(n * float(target_fps) / source_fps)))
    if out_n == n and abs(source_fps - target_fps) < 1e-6:
        return torch.arange(n, device=device, dtype=torch.long)
    i = torch.arange(out_n, device=device, dtype=torch.float64)
    t = (i + 0.5) / float(target_fps)
    src = torch.round(t * source_fps - 0.5).to(torch.long)
    return src.clamp_(0, n - 1)


def _resample_frames_cfr(images, source_fps, target_fps=FPS):
    """Deterministic nearest-frame CFR conversion using frame-center timestamps."""
    if getattr(images, "ndim", 0) != 4 or int(images.shape[0]) < 1:
        raise ValueError(
            "h3_masked_extension: source_frames must be IMAGE [N,H,W,C]"
        )
    idx = _cfr_index_map(int(images.shape[0]), source_fps, images.device, target_fps)
    return images.index_select(0, idx)


def _stereo_first_batch(waveform, label):
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_masked_extension: %s waveform must be [B,C,L], got %s"
            % (label, tuple(getattr(waveform, "shape", ())))
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_masked_extension: %s has %d channels. Downmix multichannel audio "
        "to stereo before this node; silently taking two channels can destroy "
        "the source mix." % (label, channels)
    )


def _resample_waveform(waveform, source_sr, target_sr, label):
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            "h3_masked_extension: %s is %d Hz but %d Hz is required and "
            "torchaudio is unavailable" % (label, source_sr, target_sr)
        )
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _fit_waveform(waveform, want, label, pad=True):
    want = int(want)
    have = int(waveform.shape[-1])
    if have == want:
        return waveform
    if have > want:
        _LOG.info(
            "h3_masked_extension: trimming %s by %d samples to exact timeline",
            label,
            have - want,
        )
        return waveform[..., :want]
    if not pad:
        raise ValueError(
            "h3_masked_extension: %s is %d samples short of the video timeline"
            % (label, want - have)
        )
    _LOG.warning(
        "h3_masked_extension: %s is %d samples short; padding silence at tail",
        label,
        want - have,
    )
    return torch.nn.functional.pad(waveform, (0, want - have))


def _conform_waveform_length(waveform, want, label, max_fractional_change=0.005):
    """Conform a small decoder/container duration mismatch to the exact AV timeline.

    H3 video frames live on a 24 fps grid while H3 audio latents live on a 40 Hz
    grid.  A valid H3 video run is therefore not always an exact joint AV run.
    For example, 362 video frames map to 603.333 audio ticks, so the generated
    audio latent must use an integer number of ticks and the decoded waveform can
    be a few milliseconds shorter than the exact video duration.

    For these *small* mismatches, appending silence creates an artificial gap at
    every extension seam.  Instead, resample the waveform by the tiny rational
    ratio needed to make its duration match the exact frame-derived sample span.
    Larger mismatches are not treated as timing-grid drift and are rejected
    rather than silently repaired.
    """
    want = int(want)
    have = int(waveform.shape[-1])
    if have == want:
        return waveform
    if want <= 0 or have <= 0:
        raise ValueError(
            "h3_masked_extension: %s has invalid sample length %d -> %d"
            % (label, have, want)
        )

    fractional_change = abs(want - have) / float(want)
    if fractional_change > float(max_fractional_change):
        raise ValueError(
            "h3_masked_extension: %s differs from the exact timeline by %.3f%% "
            "(%d -> %d samples), too large for AV timebase conformance"
            % (label, fractional_change * 100.0, have, want)
        )

    ratio = Fraction(want, have).limit_denominator(10000)
    if torchaudio is not None:
        conformed = torchaudio.functional.resample(
            waveform, int(ratio.denominator), int(ratio.numerator)
        )
    else:
        # Comfy's audio stack normally provides torchaudio.  Keep a dependency-
        # free exact-length fallback for lightweight/mock environments.
        conformed = torch.nn.functional.interpolate(
            waveform,
            size=want,
            mode="linear",
            align_corners=False,
        )

    # Rational approximation / backend rounding can theoretically miss by a
    # sample.  Correct that residual without ever inserting a silence tail.
    got = int(conformed.shape[-1])
    if got != want:
        conformed = torch.nn.functional.interpolate(
            conformed,
            size=want,
            mode="linear",
            align_corners=False,
        )

    _LOG.info(
        "h3_masked_extension: time-conformed %s %d -> %d samples (%.4f%%) "
        "to match exact video timeline",
        label,
        have,
        want,
        (want - have) / float(have) * 100.0,
    )
    return conformed


def _canonical_audio(audio, target_sr, frame_count):
    if audio is None:
        raise ValueError(
            "h3_masked_extension: existing-video AV extension requires source_audio"
        )
    waveform = _stereo_first_batch(audio["waveform"], "source_audio")
    waveform = _resample_waveform(
        waveform, int(audio["sample_rate"]), int(target_sr), "source_audio"
    )
    want = int(round(int(frame_count) / FPS * int(target_sr)))
    waveform = _conform_waveform_length(waveform, want, "source audio")
    return {"waveform": waveform, "sample_rate": int(target_sr)}


def _snap_context_length(requested, available, target_frames):
    # A joint H3 AV prefix must satisfy TWO timing constraints:
    #
    #   1) it must be an exact video-VAE run: 5, 22, 39, 56, 73, 90, ...
    #   2) its endpoint must also land exactly on H3's 40-Hz audio grid.
    #
    # At 24 fps / 40 Hz the shared boundaries are:
    #
    #       39, 90, 141, 192, ... frames
    #
    # Allowing video-only runs such as 73 frames creates a fractional audio
    # endpoint (73/24*40 = 121.666... ticks). That cannot be represented as
    # one exact protected AV seam and can cause either an encoder-length
    # mismatch or an audio/video phase shift.
    #
    # Context length is allowed to cover the full target (a fully preserved
    # latent with no generate rows); prepare() logs a warning for that case
    # since sampling it directly is a no-op unless the noise_mask is replaced
    # downstream (e.g. by a graduated fade mask).
    cap = min(int(requested), int(available), int(target_frames))
    run = largest_h3_video_run(cap)
    while run >= 5 and not is_exact_av_boundary(run):
        run = largest_h3_video_run(run - 1)

    if run < 39:
        raise ValueError(
            "h3_masked_extension: need at least 39 usable source frames for an "
            "exact H3 video+audio context boundary at 24 fps / 40 Hz"
        )

    if run != int(requested):
        _LOG.warning(
            "h3_masked_extension: context_length %d -> exact H3 AV prefix %d "
            "(shared AV runs are 39, 90, 141, 192, ...)",
            int(requested), run,
        )
    return run


def _apply_audio_context_feather(audio_mask, audio_steps, feather_ticks, start=0):
    """Protect an audio prefix with an optional half-cosine 0->1 release.

    H3 mask semantics are 0=preserve and 1=generate. The total protected
    context duration is unchanged; only the final ``feather_ticks`` latent
    positions transition fractionally toward generation. ``start`` offsets the
    protected window for interior inserts (default 0 = prefix at the origin).
    """
    audio_steps = int(audio_steps)
    start = int(start)
    feather = max(0, min(int(feather_ticks), audio_steps))
    hard = audio_steps - feather
    if hard > 0:
        audio_mask[..., start:start + hard] = 0.0
    if feather > 0:
        i = torch.arange(
            1, feather + 1, device=audio_mask.device, dtype=audio_mask.dtype
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * i / float(feather))
        shape = [1] * audio_mask.ndim
        shape[-1] = feather
        audio_mask[..., start + hard:start + audio_steps] = ramp.view(*shape)
    return audio_mask


def _decode_h3_video_cpu(vae, latent):
    images = vae.decode(latent)
    if getattr(images, "ndim", 0) == 5 and int(images.shape[0]) == 1:
        images = images[0]
    if getattr(images, "ndim", 0) != 4:
        raise ValueError(
            "h3_video_decode: video VAE decode returned %s; expected 4-D frames"
            % (tuple(getattr(images, "shape", ())),)
        )
    if int(images.shape[-1]) in (3, 4):
        images = images[..., :3]
    elif int(images.shape[1]) in (3, 4):
        images = images.movedim(1, -1)[..., :3]
    else:
        raise ValueError(
            "h3_video_decode: cannot infer RGB channel axis from %s"
            % (tuple(images.shape),)
        )
    return images.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _decode_h3_audio_cpu(audio_vae, latent):
    waveform = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform = waveform / std
    sr = int(
        getattr(
            audio_vae, "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 44100),
        )
    )
    return waveform.detach().to(device="cpu", dtype=torch.float32), sr


def _release_decode_memory():
    gc.collect()
    try:
        import comfy.model_management as model_management
        model_management.soft_empty_cache()
    except Exception:
        pass


class MiniMaxH3ExistingVideoMaskedContext:
    """Prepare an arbitrary decoded video as a clean H3 AV target prefix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target H3 AV latent (for example from MiniMax H3 Reference to Video)."
                }),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE."
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded source video frames. The node converts CFR timing to H3's 24 fps."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "Audio from the same source video. Mono is duplicated to stereo; multichannel must be downmixed first."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Source-frame FPS. Use CFR or a decoder-normalized frame stream; the example forces 24 fps."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "AV prefixes snap to shared H3 video+audio boundaries: 39/90/141/192/... frames."
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
                "audio_feather_ticks": ("INT", {
                    "default": 8, "min": 0, "max": 256,
                    "tooltip": "Half-cosine release across the final audio-latent context ticks. 0 = hard audio mask; 8 = 0.2 s at H3's 40 Hz audio latent rate."
                }),
            },
            "optional": {
                "insert_frame": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": (
                        "Pixel frame where the preserved segment begins in the target "
                        "latent. Snaps down to the nearest multiple of 17 (latent phase grid). "
                        "Multiples of 51 also align the audio clock exactly. "
                        "0 is the original prefix behavior."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "trim_frames", "insert_frame", "preserved_frames")
    OUTPUT_TOOLTIPS = (
        "Target AV latent with the preserved source segment written in and a denoise mask applied. Connect to the H3 sampler.",
        "Frames to trim from the start of the generated output. Equals preserved_frames for prefix inserts (insert_frame=0); 0 for interior inserts. Wire to H3 Assemble Existing Video Extension.",
        "Actual insert position used after snapping to the nearest multiple of 17. Wire to H3 Assemble Interior Insert.",
        "Number of source frames preserved in the latent. Wire to H3 Assemble Interior Insert.",
    )
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Prepare an existing decoded video as an exact preserved H3 AV segment. "
        "The source tail is normalized to H3 timing, VAE-encoded into the target "
        "latent at insert_frame, and protected with a per-stream denoise mask so "
        "H3 generates only the surrounding portions. Use H3 Assemble Interior "
        "Insert as the output path for interior inserts."
    )

    def prepare(
        self,
        latent,
        vae,
        audio_vae,
        source_frames,
        source_audio,
        source_fps,
        context_length=39,
        crop="disabled",
        audio_feather_ticks=8,
        insert_frame=0,
    ):
        _require_h3_mask_support()

        insert_frame = int(insert_frame)
        if insert_frame % 17 != 0:
            snapped = (insert_frame // 17) * 17
            _LOG.warning(
                "h3_masked_extension: insert_frame %d is not a multiple of 17; "
                "snapping down to %d", insert_frame, snapped
            )
            insert_frame = snapped

        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError(
                "h3_masked_extension: existing-video extension currently supports H3 batch size 1"
            )

        target_frames = _pixel_frames(int(target_video.shape[2]))
        expected_target_audio_steps = int(round(target_frames / FPS * AUDIO_HZ))
        if int(target_audio.shape[-1]) != expected_target_audio_steps:
            raise RuntimeError(
                "h3_masked_extension: target latent has %d audio steps for %d "
                "video frames; expected %d on H3's 40 Hz audio grid"
                % (int(target_audio.shape[-1]), target_frames, expected_target_audio_steps)
            )
        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        # Build the canonical 24-fps index map, but only materialize the tail
        # here.  The full source is normalized later by the assembly node, which
        # uses this exact same mapping; this avoids keeping a second full-size
        # resized source video alive during H3 sampling.
        if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
            raise ValueError(
                "h3_masked_extension: source_frames must be IMAGE [N,H,W,C]"
            )
        idx = _cfr_index_map(
            int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS
        )
        available = int(idx.numel())
        n = _snap_context_length(context_length, available, target_frames)
        tail_idx = idx[-n:]
        video_tail = source_frames.index_select(0, tail_idx)
        video_tail = _resize_images(video_tail, width, height, crop)

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        canonical_audio = _canonical_audio(source_audio, vae_sr, available)

        # Video prefix: encode the exact last n canonical frames.
        video_prefix = vae.encode(video_tail)
        if getattr(video_prefix, "ndim", 0) != 5:
            raise ValueError(
                "h3_masked_extension: video VAE returned %s, expected [B,C,T,H,W]"
                % (tuple(getattr(video_prefix, "shape", ())),)
            )
        video_steps = int(video_prefix.shape[2])
        covered = _pixel_frames(video_steps)
        if covered != n:
            raise RuntimeError(
                "h3_masked_extension: %d context frames encoded to %d video "
                "latent steps covering %d frames; refusing a phase-shifted seam"
                % (n, video_steps, covered)
            )

        s = insert_frame // 17 * 5
        if s + video_steps > int(target_video.shape[2]):
            raise ValueError(
                "h3_masked_extension: insert at frame %d (%d video steps) + "
                "%d context steps = %d exceeds target %d video steps"
                % (insert_frame, s, video_steps, s + video_steps, int(target_video.shape[2]))
            )

        # Audio prefix: exact same physical interval, end-aligned to source end.
        context_samples = int(round(n / FPS * vae_sr))
        waveform = canonical_audio["waveform"]
        if int(waveform.shape[-1]) < context_samples:
            raise ValueError(
                "h3_masked_extension: source audio is shorter than the selected context"
            )
        audio_tail = waveform[..., -context_samples:]
        audio_prefix = audio_vae.encode(audio_tail.movedim(1, -1))
        if getattr(audio_prefix, "ndim", 0) != 4:
            raise ValueError(
                "h3_masked_extension: audio VAE returned %s, expected [B,C,2,T]"
                % (tuple(getattr(audio_prefix, "shape", ())),)
            )
        exact_audio_steps = n / FPS * AUDIO_HZ
        expected_audio_steps = int(round(exact_audio_steps))
        if abs(exact_audio_steps - expected_audio_steps) > 1e-9:
            _LOG.warning(
                "h3_masked_extension: %d-frame context ends between H3 audio "
                "latent ticks (%.6f steps -> %d). 39 frames is recommended "
                "for an exact AV boundary.",
                n,
                exact_audio_steps,
                expected_audio_steps,
            )
        got_audio_steps = int(audio_prefix.shape[-1])
        if got_audio_steps < expected_audio_steps:
            raise RuntimeError(
                "h3_masked_extension: %d-frame context needs %d audio latent "
                "steps but the audio VAE produced only %d"
                % (n, expected_audio_steps, got_audio_steps)
            )
        if got_audio_steps > expected_audio_steps:
            _LOG.warning(
                "h3_masked_extension: audio VAE produced %d steps for a %d-step "
                "context; end-aligning and keeping the final %d",
                got_audio_steps,
                expected_audio_steps,
                expected_audio_steps,
            )
            audio_prefix = audio_prefix[..., -expected_audio_steps:]
        audio_steps = expected_audio_steps

        exact_a_start = insert_frame / FPS * AUDIO_HZ
        a_start = int(round(exact_a_start))
        if insert_frame % 51 != 0:
            error_ms = abs(exact_a_start - a_start) / AUDIO_HZ * 1000.0
            _LOG.warning(
                "h3_masked_extension: insert_frame %d is not a multiple of 51 "
                "(joint AV boundary). Audio insert rounded from %.6f to %d steps "
                "(error %.3f ms). Use a multiple of 51 for exact AV alignment.",
                insert_frame,
                exact_a_start,
                a_start,
                error_ms,
            )

        if a_start + audio_steps > int(target_audio.shape[-1]):
            raise ValueError(
                "h3_masked_extension: insert at frame %d (audio step %d) + "
                "%d audio steps = %d exceeds target %d audio steps"
                % (insert_frame, a_start, audio_steps, a_start + audio_steps, int(target_audio.shape[-1]))
            )

        # Fill the clean prefix of the actual target streams.
        out_video = target_video.clone()
        out_audio = target_audio.clone()
        vp = video_prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
        ap = audio_prefix[:1].to(device=out_audio.device, dtype=out_audio.dtype)
        if tuple(vp.shape[1:2] + vp.shape[3:]) != tuple(
            out_video.shape[1:2] + out_video.shape[3:]
        ):
            raise ValueError(
                "h3_masked_extension: encoded video prefix shape %s does not match target %s"
                % (tuple(vp.shape), tuple(out_video.shape))
            )
        if tuple(ap.shape[1:3]) != tuple(out_audio.shape[1:3]):
            raise ValueError(
                "h3_masked_extension: encoded audio prefix shape %s does not match target %s"
                % (tuple(ap.shape), tuple(out_audio.shape))
            )
        out_video[:, :, s : s + video_steps] = vp
        out_audio[..., a_start : a_start + audio_steps] = ap

        # ComfyUI PR #15375 unbinds this nested mask, expands each stream to the
        # corresponding latent shape, then H3 turns the two masks into per-row
        # video/audio timesteps.  0 = preserve, 1 = generate.
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
        video_mask[:, :, s : s + video_steps] = 0.0
        _apply_audio_context_feather(
            audio_mask, audio_steps, audio_feather_ticks, start=a_start
        )

        if not bool((video_mask > 0).any()):
            _LOG.warning(
                "h3_masked_extension: the target timeline is fully covered by the "
                "preserved context (no generate rows). Sampling this directly is a "
                "no-op; set a new latent noise_mask (e.g. a graduated fade) before "
                "generating."
            )

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        trim = n if insert_frame == 0 else 0
        _LOG.info(
            "h3_masked_extension: source %.6g fps -> %d canonical 24fps frames at "
            "%dx%d; preserved segment %d frames at insert frame %d "
            "(video steps [%d:%d] / audio steps [%d:%d], %.3fs); "
            "audio feather %d ticks; target %d frames",
            float(source_fps),
            available,
            width,
            height,
            n,
            insert_frame,
            s,
            s + video_steps,
            a_start,
            a_start + audio_steps,
            n / FPS,
            max(0, min(int(audio_feather_ticks), audio_steps)),
            target_frames,
        )
        return (out, trim, insert_frame, n)


def _normalize_frame_mask(mask):
    """Coerce a ComfyUI MASK to [F, H, W] float32."""
    m = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    m = m.to(dtype=torch.float32)
    if m.ndim == 2:            # [H, W]
        m = m.unsqueeze(0)
    elif m.ndim == 4:          # [F, C, H, W] -> average channels away
        m = m.mean(dim=1)
    elif m.ndim != 3:          # want [F, H, W]
        raise ValueError(
            "h3_av_noise_mask: MASK must be [H,W], [F,H,W] or [F,C,H,W], got %s"
            % (tuple(m.shape),)
        )
    return m


def _mask_to_video_stream(mask, t, h, w):
    """Frame-space MASK -> [1,1,T,H,W] video noise-mask stream at latent res."""
    m = _normalize_frame_mask(mask)
    m = m.reshape(1, 1, m.shape[0], m.shape[1], m.shape[2])
    m = torch.nn.functional.interpolate(
        m, size=(int(t), int(h), int(w)), mode="trilinear", align_corners=False
    )
    return m.clamp_(0.0, 1.0)


def _mask_to_audio_stream(mask, t):
    """Frame-space MASK -> [1,1,2,T] audio noise-mask stream (spatial reduced).

    Audio latent time is uniform 40 Hz, so a plain linear resample of the
    spatially-averaged frame mask suffices (no phase handling), broadcast to the
    two audio channels.
    """
    m = _normalize_frame_mask(mask)
    m = m.mean(dim=(1, 2)).reshape(1, 1, -1)          # [1, 1, F]
    m = torch.nn.functional.interpolate(
        m, size=int(t), mode="linear", align_corners=False
    )                                                  # [1, 1, T]
    m = m.reshape(1, 1, 1, int(t)).expand(1, 1, 2, int(t)).contiguous()
    return m.clamp_(0.0, 1.0)


def _existing_mask_streams(latent):
    """Return (video_mask, audio_mask) from a latent's nested noise_mask, else (None, None)."""
    existing = latent.get("noise_mask")
    if existing is None:
        return None, None
    if hasattr(existing, "unbind"):
        parts = list(existing.unbind())
    elif isinstance(existing, (tuple, list)):
        parts = list(existing)
    else:
        return None, None
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


class MiniMaxH3SetAVNoiseMask:
    """Set a nested H3 AV noise mask (video + audio) on an AV latent.

    ComfyUI's stock ``Set Latent Noise Mask`` overwrites ``noise_mask`` with a
    single plain tensor, which H3 unpacks to only the video stream — the audio
    denoise-mask becomes ``None`` and preserved insert audio is silently
    regenerated.  This node writes a proper two-stream ``NestedTensor`` so both
    streams keep their intended per-step protection.  Provide masks in frame
    space; the node resizes each to the latent's stream resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",)},
            "optional": {
                "video_mask": ("MASK",),
                "audio_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "set_mask"
    CATEGORY = "conditioning/minimax"

    def set_mask(self, latent, video_mask=None, audio_mask=None):
        if video_mask is None and audio_mask is None:
            raise ValueError(
                "h3_av_noise_mask: both video_mask and audio_mask are None. To "
                "remove the noise mask entirely use MiniMaxH3ClearAVNoiseMask; to "
                "discard just one stream, zero that mask and set it."
            )

        video, audio = _streams_from_latent(latent)
        ex_video, ex_audio = _existing_mask_streams(latent)

        # For a missing stream: keep the latent's existing one if present,
        # otherwise default to all-generate (all-ones). H3 treats an all-ones
        # stream identically to an absent mask -- a stream is only installed as a
        # denoise condition when its min < 1 (h3_mask_payload_compat) -- so this
        # is exactly "that stream fully generates", the same state a Clear leaves.
        if video_mask is not None:
            out_video = _mask_to_video_stream(
                video_mask, video.shape[2], video.shape[3], video.shape[4]
            ).to(device=video.device)
        elif ex_video is not None:
            out_video = ex_video
        else:
            out_video = torch.ones(
                (1, 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])),
                device=video.device,
                dtype=torch.float32,
            )

        if audio_mask is not None:
            out_audio = _mask_to_audio_stream(audio_mask, audio.shape[-1]).to(
                device=audio.device
            )
        elif ex_audio is not None:
            out_audio = ex_audio
        else:
            out_audio = torch.ones(
                (1, 1, 2, int(audio.shape[-1])),
                device=audio.device,
                dtype=torch.float32,
            )

        out = latent.copy()
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        _LOG.info(
            "h3_av_noise_mask: set nested AV noise mask (video %s / audio %s)",
            tuple(out_video.shape),
            tuple(out_audio.shape),
        )
        return (out,)


class MiniMaxH3ClearAVNoiseMask:
    """Remove any noise mask from an H3 AV latent (nested-aware).

    The essentials 'remove mask' node is an external dependency; this ships the
    same capability and correctly drops a nested AV noise mask so the sampler
    treats the whole latent as fully generated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "clear_mask"
    CATEGORY = "conditioning/minimax"

    def clear_mask(self, latent):
        out = latent.copy()
        out.pop("noise_mask", None)
        return (out,)


class MiniMaxH3AssembleInterior:
    """Frame/sample-exact assembly for an interior existing-video insert.

    The model was conditioned to preserve the insert region via a noise mask, so
    the seams here are the same nature as the prefix seam that MiniMaxH3AssembleExtension
    ships hard-cut. No crossfade is applied; use the soft keyframe node for soft anchoring.

    This node is the required output path for interior inserts: the VAE-decoded
    output around an interior preserved region does not pixel-match the source
    because of the causal VAE, so without this node interior inserts have no
    correct output path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "continuation_images": ("IMAGE", {
                    "tooltip": "Decoded H3 output frames (full timeline, all frames)."
                }),
                "continuation_audio": ("AUDIO", {
                    "tooltip": "Decoded H3 output audio (full timeline)."
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded source video frames. Normalized using the same 24-fps mapping the context node used."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "Audio from the source video."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "insert_frame": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Wire from the insert_frame output of H3 Existing Video Masked Context."
                }),
                "preserved_frames": ("INT", {
                    "default": 39, "min": 1,
                    "tooltip": "Wire from the preserved_frames output of H3 Existing Video Masked Context."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Use 24 for MiniMax H3."
                }),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "Must match the crop setting used by H3 Existing Video Masked Context."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "assemble"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Splice the canonical source frames and audio back over the interior "
        "preserved interval in the decoded H3 continuation. Hard-cut splice with "
        "exact AV accounting. Required output path for interior inserts because the "
        "causal VAE cannot exactly round-trip interior latent regions."
    )

    def assemble(
        self,
        continuation_images,
        continuation_audio,
        source_frames,
        source_audio,
        source_fps,
        insert_frame,
        preserved_frames,
        fps=24.0,
        crop="disabled",
    ):
        fps = float(fps)
        if fps <= 0:
            raise ValueError("h3_masked_extension: fps must be > 0")

        insert_frame = int(insert_frame)
        preserved_frames = int(preserved_frames)
        n = preserved_frames

        if getattr(continuation_images, "ndim", 0) != 4 or int(continuation_images.shape[0]) < 1:
            raise ValueError("h3_masked_extension: continuation_images is empty")

        cont_frames = int(continuation_images.shape[0])
        if insert_frame + n > cont_frames:
            raise ValueError(
                "h3_masked_extension: insert_frame %d + preserved_frames %d = %d "
                "exceeds continuation length %d"
                % (insert_frame, n, insert_frame + n, cont_frames)
            )

        height = int(continuation_images.shape[1])
        width = int(continuation_images.shape[2])

        # Normalize source to 24 fps using the same mapping the context node used,
        # then take the last n canonical frames.
        idx = _cfr_index_map(
            int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS
        )
        available = int(idx.numel())
        if available < n:
            raise ValueError(
                "h3_masked_extension: canonical source has %d frames, need %d"
                % (available, n)
            )
        tail_idx = idx[-n:]
        src_images = source_frames.index_select(0, tail_idx)
        src_images = _resize_images(src_images, width, height, crop)

        # Splice source images into the continuation.
        out_images = continuation_images.clone()
        out_images[insert_frame : insert_frame + n] = src_images

        # Audio splice: split continuation audio, replace insert interval with source.
        cont_sr = int(continuation_audio["sample_rate"])
        cont_wave = _stereo_first_batch(continuation_audio["waveform"], "continuation_audio")

        src_wave = _stereo_first_batch(source_audio["waveform"], "source_audio")
        src_wave = _resample_waveform(
            src_wave, int(source_audio["sample_rate"]), cont_sr, "source_audio"
        )

        # Compute exact sample boundaries for each region.
        total_want = int(round(cont_frames / fps * cont_sr))
        before_want = int(round(insert_frame / fps * cont_sr))
        insert_end = insert_frame + n
        insert_end_want = int(round(insert_end / fps * cont_sr))
        src_want = insert_end_want - before_want  # samples for the n source frames

        before_wave = _fit_waveform(cont_wave[..., :before_want], before_want, "continuation before-insert")
        after_want = total_want - insert_end_want
        after_wave = _fit_waveform(cont_wave[..., insert_end_want:], after_want, "continuation after-insert")

        # Mirror _canonical_audio: fit the resampled source to the full canonical sample
        # count before taking the tail. This matches what the context node does — short
        # source audio is padded with silence so the splice agrees with what H3 was
        # conditioned to preserve.
        src_canonical_samples = int(round(available / FPS * cont_sr))
        canonical_src_wave = _fit_waveform(src_wave, src_canonical_samples, "source audio (canonical)")
        needed_from_src = int(round(n / FPS * cont_sr))
        src_segment = canonical_src_wave[..., -needed_from_src:]
        src_segment = _fit_waveform(src_segment, src_want, "source insert audio")

        waveform = torch.cat((before_wave, src_segment, after_wave), dim=-1)
        waveform = _fit_waveform(waveform, total_want, "assembled audio")

        audio = {"waveform": waveform, "sample_rate": cont_sr}

        expected = int(round(cont_frames / fps * cont_sr))
        if int(waveform.shape[-1]) != expected:
            raise RuntimeError(
                "h3_masked_extension: internal AV accounting failed: got %d audio "
                "samples, expected %d for %d frames"
                % (int(waveform.shape[-1]), expected, cont_frames)
            )

        _LOG.info(
            "h3_masked_extension: interior splice at frames [%d:%d] of %d-frame "
            "continuation; source %d canonical frames; %d audio samples at %d Hz "
            "(drift 0 samples by construction)",
            insert_frame,
            insert_frame + n,
            cont_frames,
            n,
            int(waveform.shape[-1]),
            cont_sr,
        )
        return (out_images, audio)


class MiniMaxH3GeneratedAVMaskedContext:
    """Continue from a previous generated H3 clip directly in latent space.

    Unlike ExistingVideoMaskedContext, this node does not decode and re-encode
    the previous clip. It copies the previous clip's final valid H3 video/audio
    latent run into the new target's prefix and protects both streams with
    per-token mask=0. This is the preferred way to chain generated H3 clips.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Fresh target AV latent from MiniMax H3 Image/Reference to Video."
                }),
                "source_latent": ("LATENT", {
                    "tooltip": "Previous generated H3 AV latent. Its final AV run is copied directly into the new target prefix."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Protected AV prefix. Snaps to shared 24-fps video / 40-Hz audio boundaries: 39/90/141/192/... frames."
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 8, "min": 0, "max": 256,
                    "tooltip": "Half-cosine release across the final audio-latent context ticks. 0 = hard audio mask; 8 = 0.2 s at H3's 40 Hz audio latent rate."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Chain generated H3 clips without a decode/re-encode round trip. The "
        "previous clip's final joint video/audio latent run is copied into the "
        "new target prefix and protected with AV noise-mask 0; only the future "
        "region denoises."
    )

    def prepare(self, latent, source_latent, context_length=39, audio_feather_ticks=8):
        _require_h3_mask_support()
        target_video, target_audio = _streams_from_latent(latent)
        source_video, source_audio = _streams_from_latent(source_latent)

        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError(
                "h3_masked_extension: generated-latent continuation currently supports target batch size 1"
            )
        if int(source_video.shape[0]) != 1 or int(source_audio.shape[0]) != 1:
            raise ValueError(
                "h3_masked_extension: generated-latent continuation currently supports source batch size 1"
            )

        target_frames = _pixel_frames(int(target_video.shape[2]))
        source_frames = _pixel_frames(int(source_video.shape[2]))
        n = _snap_context_length(context_length, source_frames, target_frames)

        # H3-valid pixel runs 5/22/39/... map to temporal latent runs
        # 2/7/12/... . Both full H3 clips and valid context runs are 2 mod 5
        # latent steps, so the source tail starts at the same temporal phase as
        # the target prefix. Direct slicing is therefore phase-aligned.
        video_steps = 2 + 5 * ((n - 5) // 17)
        if _pixel_frames(video_steps) != n:
            raise RuntimeError(
                "h3_masked_extension: internal H3 context mapping failed for %d frames" % n
            )
        audio_steps = int(round(n / FPS * AUDIO_HZ))

        if video_steps >= int(target_video.shape[2]):
            raise ValueError(
                "h3_masked_extension: video context consumes the whole target latent"
            )
        if audio_steps >= int(target_audio.shape[-1]):
            raise ValueError(
                "h3_masked_extension: audio context consumes the whole target latent"
            )
        if int(source_video.shape[2]) < video_steps:
            raise ValueError(
                "h3_masked_extension: source latent has too few video steps for the requested context"
            )
        if int(source_audio.shape[-1]) < audio_steps:
            raise ValueError(
                "h3_masked_extension: source latent has too few audio steps for the requested context"
            )

        if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
            target_video.shape[1:2] + target_video.shape[3:]
        ):
            raise ValueError(
                "h3_masked_extension: source/target video latent geometry differs: %s vs %s. "
                "Keep chained clips at the same H3 resolution."
                % (tuple(source_video.shape), tuple(target_video.shape))
            )
        if tuple(source_audio.shape[1:3]) != tuple(target_audio.shape[1:3]):
            raise ValueError(
                "h3_masked_extension: source/target audio latent geometry differs: %s vs %s"
                % (tuple(source_audio.shape), tuple(target_audio.shape))
            )

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_video[:, :, :video_steps] = source_video[:, :, -video_steps:].to(
            device=out_video.device, dtype=out_video.dtype
        )
        out_audio[..., :audio_steps] = source_audio[..., -audio_steps:].to(
            device=out_audio.device, dtype=out_audio.dtype
        )

        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device, dtype=torch.float32,
        )
        audio_mask = torch.ones(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device, dtype=torch.float32,
        )
        video_mask[:, :, :video_steps] = 0.0
        _apply_audio_context_feather(audio_mask, audio_steps, audio_feather_ticks)

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        _LOG.info(
            "h3_masked_extension: latent->latent AV continuation protected %d frames = %d video steps / %d audio steps; audio feather %d ticks; target %d frames",
            n, video_steps, audio_steps, max(0, min(int(audio_feather_ticks), audio_steps)), target_frames,
        )
        return (out, n)


class MiniMaxH3StartMaskedContext:
    """Prepare Extension 1 from an existing video or a live generated starter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Fresh target AV latent for Extension 1."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE; used for existing-video starts."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE; used for existing-video starts."}),
                "start_mode": ("STRING", {"forceInput": True}),
                "context_length": ("INT", {"default": 39, "min": 5, "max": 9999}),
                "audio_feather_ticks": ("INT", {"default": 8, "min": 0, "max": 256}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": {
                "source_frames": ("IMAGE", {"lazy": True}),
                "source_audio": ("AUDIO", {"lazy": True}),
                "live_starter_latent": ("LATENT", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Checkpoint-free Extension-1 start selector. Existing Video requests decoded source AV; "
        "T2V/I2V request the live generated starter latent."
    )

    @classmethod
    def IS_CHANGED(cls, start_mode="existing_video", **kwargs):
        return "live-start:%s" % str(start_mode)

    def check_lazy_status(
        self, latent, vae, audio_vae, start_mode, context_length,
        audio_feather_ticks, source_fps, crop,
        source_frames=None, source_audio=None, live_starter_latent=None,
    ):
        if str(start_mode) == "existing_video":
            needed = []
            if source_frames is None:
                needed.append("source_frames")
            if source_audio is None:
                needed.append("source_audio")
            return needed
        if live_starter_latent is None:
            return ["live_starter_latent"]
        return []

    def prepare(
        self, latent, vae, audio_vae, start_mode="existing_video", context_length=39,
        audio_feather_ticks=8, source_fps=24.0, crop="disabled",
        source_frames=None, source_audio=None, live_starter_latent=None,
    ):
        if str(start_mode) == "existing_video":
            if source_frames is None or source_audio is None:
                raise ValueError("h3_masked_extension: Existing Video start requires source frames and audio")
            return MiniMaxH3ExistingVideoMaskedContext().prepare(
                latent, vae, audio_vae, source_frames, source_audio, source_fps,
                context_length=context_length, crop=crop, audio_feather_ticks=audio_feather_ticks,
            )
        if live_starter_latent is None:
            raise ValueError("h3_masked_extension: T2V/I2V start requires live_starter_latent")
        return MiniMaxH3GeneratedAVMaskedContext().prepare(
            latent, live_starter_latent, context_length, audio_feather_ticks
        )



class MiniMaxH3AssembleExtension:
    """Frame/sample-exact assembly for an existing-video continuation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded source video frames. They are normalized to the same 24-fps mapping used by the context node."
                }),
                "source_audio": ("AUDIO",),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001
                }),
                "continuation_images": ("IMAGE",),
                "continuation_audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Use 24 for MiniMax H3. Audio is forced to the exact integer-sample duration implied by each image batch before concatenation."
                }),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "Must match the crop setting used by H3 Existing Video Masked Context so the prepended source and conditioning tail use the identical spatial transform."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "assemble"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Concatenate the canonical source and the trimmed H3 continuation. "
        "No crossfade: both audio pieces are resampled/matched to exact frame "
        "durations first so the join cannot accumulate AV drift."
    )

    def assemble(
        self,
        source_frames,
        source_audio,
        source_fps,
        continuation_images,
        continuation_audio,
        fps=24.0,
        crop="disabled",
    ):
        fps = float(fps)
        if fps <= 0:
            raise ValueError("h3_masked_extension: fps must be > 0")
        if getattr(continuation_images, "ndim", 0) != 4 or int(continuation_images.shape[0]) < 1:
            raise ValueError("h3_masked_extension: continuation_images is empty")
        height = int(continuation_images.shape[1])
        width = int(continuation_images.shape[2])
        source_images = _resample_frames_cfr(source_frames, float(source_fps), fps)
        source_images = _resize_images(source_images, width, height, crop)

        cont_sr = int(continuation_audio["sample_rate"])
        src_wave = _stereo_first_batch(source_audio["waveform"], "source_audio")
        src_wave = _resample_waveform(
            src_wave, int(source_audio["sample_rate"]), cont_sr, "source_audio"
        )
        cont_wave = _stereo_first_batch(
            continuation_audio["waveform"], "continuation_audio"
        )
        cont_wave = _resample_waveform(
            cont_wave,
            int(continuation_audio["sample_rate"]),
            cont_sr,
            "continuation_audio",
        )

        src_frames = int(source_images.shape[0])
        cont_frames = int(continuation_images.shape[0])

        src_want = int(round(src_frames / fps * cont_sr))
        total_want = int(round((src_frames + cont_frames) / fps * cont_sr))
        cont_want = total_want - src_want

        src_wave = _fit_waveform(src_wave, src_want, "source audio")
        cont_wave = _fit_waveform(cont_wave, cont_want, "continuation audio")

        images = torch.cat((source_images, continuation_images), dim=0)
        waveform = torch.cat((src_wave, cont_wave), dim=-1)
        audio = {"waveform": waveform, "sample_rate": cont_sr}

        expected = int(round(int(images.shape[0]) / fps * cont_sr))
        if int(waveform.shape[-1]) != expected:
            raise RuntimeError(
                "h3_masked_extension: internal AV accounting failed: got %d audio "
                "samples, expected %d for %d frames"
                % (int(waveform.shape[-1]), expected, int(images.shape[0]))
            )
        _LOG.info(
            "h3_masked_extension: assembled %d + %d = %d frames; %d audio samples "
            "at %d Hz (drift 0 samples by construction)",
            int(source_images.shape[0]),
            int(continuation_images.shape[0]),
            int(images.shape[0]),
            int(waveform.shape[-1]),
            cont_sr,
        )
        return (images, audio)
