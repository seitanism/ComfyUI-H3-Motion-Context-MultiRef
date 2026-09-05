"""Lazy PR #15972 equivalent, scoped to a connected H3 audio VAE instance."""
import logging

_LOG = logging.getLogger("h3_motion_context")
_MARKER = "_h3_audio_crop_compat_15972"


def ensure_h3_audio_vae_crop(vae):
    """Disable generic PCM center cropping only if this H3 AudioVAE needs it.

    Native crop_input=False wins. No constructor/class is monkeypatched and no
    network/version check is used: installed behavior, including backports, wins.
    Lightweight test VAEs without a real H3 first-stage model are left untouched.
    """
    first = getattr(vae, "first_stage_model", None)
    try:
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    except ImportError:
        return {"is_h3_audio": False, "patched": False}
    if not isinstance(first, MiniMaxH3AudioVAE):
        return {"is_h3_audio": False, "patched": False}
    if not hasattr(vae, "crop_input"):
        raise RuntimeError("H3 audio VAE wrapper has no crop_input control; update ComfyUI.")
    if vae.crop_input:
        vae.crop_input = False
        setattr(vae, _MARKER, True)
        _LOG.info("H3 audio VAE: disabled generic input crop (PR #15972 compatibility)")
    return {"is_h3_audio": True, "patched": bool(getattr(vae, _MARKER, False)),
            "crop_input": bool(vae.crop_input)}


class MiniMaxH3AudioVAECompatibility:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Route an H3 audio VAE through this node before stock H3 audio-reference nodes. "
        "Disables generic PCM center cropping only on an unfixed H3 AudioVAE instance. "
        "Already-fixed native VAEs are unchanged. Repo-owned audio encodes also do this lazily."
    )

    def prepare(self, audio_vae):
        status = ensure_h3_audio_vae_crop(audio_vae)
        if not status["is_h3_audio"]:
            raise ValueError("H3 Audio VAE Compatibility requires the MiniMax H3 audio VAE.")
        return (audio_vae,)
