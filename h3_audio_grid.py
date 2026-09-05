"""Shared MiniMax H3 audio-grid helpers.

H3 audio latents run at 40 Hz (normally 800 PCM samples per latent at 32 kHz).
ComfyUI's generic VAE wrapper may crop non-grid-aligned inputs before the H3
AudioVAE sees them, so repo nodes that need timeline-precise H3 audio should
prepare an exact latent-grid PCM span *before* calling ``audio_vae.encode``.
"""

AUDIO_HZ = 40.0


def audio_grid_geometry(audio_vae, target_audio_steps, expected_hz=AUDIO_HZ):
    """Return ``(sample_rate, samples_per_latent, target_pcm_samples)``.

    Prefer geometry advertised by the underlying AudioVAE and fall back to the
    H3 40-Hz contract only when the model does not expose it.  This deliberately
    avoids depending on ComfyUI's generic wrapper crop implementation.
    """
    sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    first_stage = getattr(audio_vae, "first_stage_model", None)
    samples_per_latent = getattr(first_stage, "samples_per_latent", None)
    if samples_per_latent is None:
        samples_per_latent = getattr(first_stage, "hop_length", None)
    if samples_per_latent is None:
        nominal = sample_rate / float(expected_hz)
        samples_per_latent = int(round(nominal))
        if abs(samples_per_latent - nominal) > 1e-9:
            raise RuntimeError(
                "H3 audio VAE sample rate %d is not integral on the %.0f Hz latent grid"
                % (sample_rate, float(expected_hz))
            )

    samples_per_latent = int(samples_per_latent)
    if samples_per_latent <= 0:
        raise RuntimeError("invalid H3 audio VAE samples-per-latent %d" % samples_per_latent)

    actual_hz = sample_rate / float(samples_per_latent)
    if abs(actual_hz - float(expected_hz)) > 1e-6:
        raise RuntimeError(
            "H3 audio VAE geometry is %d Hz / %d samples = %.6f latent Hz; expected %.0f"
            % (sample_rate, samples_per_latent, actual_hz, float(expected_hz))
        )

    steps = int(target_audio_steps)
    if steps < 1:
        raise ValueError("target H3 audio grid must contain at least one latent step")
    return sample_rate, samples_per_latent, steps * samples_per_latent


def encode_exact_audio_grid(audio_vae, waveform, target_audio_steps, label="H3 audio"):
    """Encode PCM that is already exactly aligned to the requested H3 grid.

    The strict output-length check is intentional: once the input is an exact
    multiple of the live AudioVAE hop, any different latent length is a real
    wrapper/encoder contract problem and should not be hidden by latent padding
    or trimming.
    """
    from .h3_audio_vae_compat import ensure_h3_audio_vae_crop
    ensure_h3_audio_vae_crop(audio_vae)
    sample_rate, samples_per_latent, expected_samples = audio_grid_geometry(
        audio_vae, target_audio_steps
    )
    have = int(waveform.shape[-1])
    if have != expected_samples:
        raise RuntimeError(
            "%s PCM is %d samples; exact H3 grid requires %d (%d x %d)"
            % (label, have, expected_samples, int(target_audio_steps), samples_per_latent)
        )

    encoded = audio_vae.encode(waveform.movedim(1, -1))
    if getattr(encoded, "ndim", 0) != 4:
        raise ValueError(
            "%s audio VAE returned %s; expected [B,C,2,T]"
            % (label, tuple(getattr(encoded, "shape", ())))
        )
    got = int(encoded.shape[-1])
    if got != int(target_audio_steps):
        raise RuntimeError(
            "%s exact-grid PCM (%d samples = %d x %d) encoded to %d latent steps; "
            "expected %d. This is an audio-VAE wrapper/encoder contract mismatch."
            % (
                label,
                expected_samples,
                int(target_audio_steps),
                samples_per_latent,
                got,
                int(target_audio_steps),
            )
        )
    return encoded, {
        "vae_sample_rate": sample_rate,
        "samples_per_latent": samples_per_latent,
        "grid_samples": expected_samples,
    }
