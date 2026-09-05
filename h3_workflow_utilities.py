"""Update 9 workflow contracts: 24 fps input validation and exact AV bridging."""
import math


def _bridge_timing(target_frames, preserve_frames):
    target, preserve = int(target_frames), int(preserve_frames)
    if target < 5 or (target - 5) % 17:
        raise ValueError("H3 target length must be 5 + 17*k frames: 5, 22, 39, ..., 192, ...")
    if preserve < 39 or (preserve - 39) % 51:
        raise ValueError("H3 AV context must be 39 + 51*k frames: 39, 90, 141, 192, ...")
    if 2 * preserve >= target:
        raise ValueError("H3 AV Bridge target must be longer than both preserved contexts combined.")
    return target, preserve


class MiniMaxH3Validate24FPSVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "video_info": ("VHS_VIDEOINFO",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "validate"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Require a 24 fps source and loaded frame batch for current H3 workflows. "
        "Convert the source to constant 24 fps before loading it. VHS metadata cannot certify VFR timing."
    )

    def validate(self, images, video_info):
        if not isinstance(video_info, dict):
            raise ValueError("H3 requires VHS video_info to verify 24 fps input.")
        for key in ("source_fps", "loaded_fps"):
            value = float(video_info.get(key, 0))
            if not math.isfinite(value) or abs(value - 24.0) > 1e-3:
                raise ValueError(f"H3 requires 24 fps input ({key}={value:g}). Convert to constant 24 fps first; use select_every_nth=1.")
        return (images,)


class MiniMaxH3AVBridgeTiming:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "target_frames": ("INT", {"default": 192, "min": 5, "max": 9999, "step": 17,
                                      "tooltip": "5 + 17*k; must exceed twice the context."}),
            "preserve_frames": ("INT", {"default": 39, "min": 39, "max": 9984, "step": 51,
                                        "tooltip": "39, 90, 141, 192, ... shared video/audio boundaries."}),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("target_frames", "preserve_frames")
    FUNCTION = "resolve"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Validate 24 fps H3 bridge timing before generation. Contexts use 39 + 51*k; targets use 5 + 17*k."

    def resolve(self, target_frames=192, preserve_frames=39):
        return _bridge_timing(target_frames, preserve_frames)


class MiniMaxH3AssembleBridgeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generated_audio": ("AUDIO",), "start_audio": ("AUDIO",), "end_audio": ("AUDIO",),
            "start_frames": ("IMAGE",), "end_frames": ("IMAGE",),
            "target_frames": ("INT", {"forceInput": True}),
            "preserve_frames": ("INT", {"forceInput": True}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "assemble"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Trim the two protected H3 audio contexts, conform the generated middle to its 24 fps picture span, "
        "and concatenate source A + middle + source B on absolute sample boundaries."
    )

    def assemble(self, generated_audio, start_audio, end_audio, start_frames, end_frames,
                 target_frames, preserve_frames):
        import torch
        from . import existing_video_extension as ext
        from .h3_timing import sample_boundary_from_frames as boundary
        target, preserve = _bridge_timing(target_frames, preserve_frames)
        counts = (int(start_frames.shape[0]), int(end_frames.shape[0]))
        if min(counts) < preserve:
            raise ValueError("Each bridge source must contain at least preserve_frames at 24 fps.")
        sr = int(generated_audio["sample_rate"])
        wave = ext._stereo_first_batch(generated_audio["waveform"], "bridge generated audio").to("cpu")
        target_ticks, context_ticks = round(target / 24 * 40), preserve * 5 // 3
        tick_boundary = lambda t: int(round(t / 40 * sr))
        expected_decoded = tick_boundary(target_ticks)
        if abs(int(wave.shape[-1]) - expected_decoded) > 2:
            raise ValueError("Bridge decoded audio length differs from its H3 40 Hz latent grid; check target_frames and decoder.")
        middle = wave[..., tick_boundary(context_ticks):tick_boundary(target_ticks - context_ticks)]
        start_end = boundary(counts[0], sr)
        middle_end = boundary(counts[0] + target - 2 * preserve, sr)
        final_end = boundary(counts[0] + target - 2 * preserve + counts[1], sr)
        want = middle_end - start_end
        if abs(int(middle.shape[-1]) - want) > math.ceil(sr / 40) + 2:
            raise ValueError("Bridge middle mismatch exceeds one H3 audio cell; refusing to hide a timing error.")
        middle = ext._conform_waveform_length(middle, want, "bridge middle", max_fractional_change=0.05)
        sources = []
        for name, audio, count, length in (("start", start_audio, counts[0], start_end),
                                           ("end", end_audio, counts[1], final_end - middle_end)):
            canonical = ext._canonical_audio(audio, sr, count)["waveform"].to("cpu")
            # Absolute boundaries may differ from relative rounding by one sample.
            sources.append(ext._fit_waveform(canonical, length, "bridge " + name))
        result = torch.cat((sources[0], middle, sources[1]), dim=-1)
        if result.shape[-1] != final_end:
            raise RuntimeError("Bridge audio sample accounting failed.")
        return ({"waveform": result, "sample_rate": sr},)
