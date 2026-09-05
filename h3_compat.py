"""Capability orchestration for H3 Motion Context on current ComfyUI.

Classic Motion Context / MultiRef uses native ComfyUI guide support introduced
by PR #15439 and never monkey-patches H3 core.  The existing-video masked
extension is a separate feature: it still needs PR #15375-equivalent AV mask
support when the live ComfyUI build does not provide that support natively.
"""

from __future__ import annotations

import logging

import torch

_LOG = logging.getLogger("h3_motion_context")
_LOGGED = set()


def _log_once(key, message, *args):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    _LOG.info(message, *args)


def native_guide_status():
    """Probe the live H3 core for the #15439 capabilities we rely on.

    This is intentionally behavioral for PackedLayout: a guide is placed at a
    nonzero target frame while an ordinary Ref2VA image ref is present.  The
    guide's video and audio condition segments must both land relative to the
    *post-ref target origin*.
    """
    import comfy.ldm.minimax.model as mm
    import comfy.model_base as model_base

    out = {
        "packed_layout_available": False,
        "ref_aware_arbitrary_guides": False,
        "guide_audio_segment": False,
        "native_keyframe_ref_merge": False,
        "native_keyframe_ref_audio_merge": False,
    }

    layout_cls = getattr(mm, "PackedLayout", None)
    if layout_cls is not None:
        out["packed_layout_available"] = True
        try:
            kf = {
                "resolved_frame_index": 3,
                "latent": torch.zeros((1, 24, 1, 2, 2)),
                "audio_latent": torch.zeros((1, 32, 2, 2)),
            }
            refs = [{"kind": "image", "latent_h": 2, "latent_w": 2}]
            layout = layout_cls(7, 7, 2, 2, 16, keyframes=[kf], refs=refs)
            cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
            aud_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond_audio"]
            # One image ref advances target origin by exactly 1.0.
            expected = 8.0 + float(mm.FRAME_RESCALE) * 3.0
            if cond_spans:
                a, b = cond_spans[0]
                vals = layout.position_ids[a:b, 0]
                out["ref_aware_arbitrary_guides"] = bool(torch.allclose(
                    vals, torch.full_like(vals, expected), atol=1e-9, rtol=0.0))
            if aud_spans:
                a, b = aud_spans[0]
                vals = layout.position_ids[a:b, 0]
                # Stereo audio rows begin at the guide's cond_t.
                rt = 2
                out["guide_audio_segment"] = bool(
                    torch.allclose(vals[:rt], torch.tensor([expected, expected + 1.0], dtype=vals.dtype), atol=1e-9, rtol=0.0)
                )
        except Exception as exc:
            out["packed_layout_error"] = repr(exc)

    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    if fn is not None:
        try:
            # Probe the live method behavior instead of grepping its source.
            # Packaged Comfy builds, decorators, wrappers, monkey-patches and
            # harmless upstream refactors can all make inspect.getsource()
            # unavailable or textually different while preserving the exact
            # required semantics.
            probe = cls.__new__(cls)
            # BaseModel.extra_conds() only needs these attributes when called
            # without cross-attention/noise inputs.  Avoid running the very
            # heavyweight H3 model constructor just for a capability check.
            probe.concat_keys = ()
            probe.latent_shapes = None

            kf_v = torch.zeros((1, 24, 1, 2, 2))
            ref_v = torch.ones((1, 24, 1, 2, 2))
            kf_a = torch.zeros((1, 32, 2, 2))
            ref_a = torch.ones((1, 32, 2, 2))
            keyframes = [{"resolved_frame_index": 0,
                          "latent": kf_v, "audio_latent": kf_a}]
            refs = [
                {"kind": "image", "latent_h": 2, "latent_w": 2,
                 "latent": ref_v},
                {"kind": "audio", "ref_audio_t": 2,
                 "audio_latent": ref_a},
            ]
            result = fn(probe, minimax_keyframes=keyframes, minimax_refs=refs)
            wrapped = result.get("minimax_payload") if isinstance(result, dict) else None
            payload = getattr(wrapped, "cond", wrapped)
            if isinstance(payload, dict):
                videos = payload.get("cond_video_latents", [])
                audios = payload.get("cond_audio_latents", [])
                out["native_keyframe_ref_merge"] = (
                    len(videos) == 2 and videos[0] is kf_v and videos[1] is ref_v
                )
                out["native_keyframe_ref_audio_merge"] = (
                    len(audios) == 2 and audios[0] is kf_a and audios[1] is ref_a
                )
        except Exception as exc:
            out["extra_conds_probe_error"] = repr(exc)
    return out


def _conditioning_ref_requirements(conditioning):
    """Return which #15439 ref/keyframe merge paths this conditioning uses.

    Classic Motion Context without Ref2VA references does not need either
    merge capability.  Image/video references need the video-latent merge;
    standalone/reference-video audio needs the audio-latent merge as well.
    """
    need_video = False
    need_audio = False
    for item in conditioning or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        meta = item[1]
        if not isinstance(meta, dict):
            continue
        for ref in meta.get("minimax_refs", []) or []:
            if not isinstance(ref, dict):
                continue
            if ref.get("latent") is not None:
                need_video = True
            if ref.get("audio_latent") is not None:
                need_audio = True
    return need_video, need_audio


def ensure_motion_context_compat(conditioning=None):
    """Require only the native #15439 features the current graph actually uses.

    The guide/layout capabilities are always required. Ref2VA merge support is
    conditional: Simple Motion Context has no refs and should not be rejected
    because an unrelated ref-merge probe fails, while Advanced/MultiRef paths
    still fail loudly if the merge they need is genuinely unavailable.
    """
    status = native_guide_status()
    need_video_ref_merge, need_audio_ref_merge = _conditioning_ref_requirements(conditioning)

    missing = []
    if not status.get("ref_aware_arbitrary_guides"):
        missing.append("ref_aware_arbitrary_guides")
    if not status.get("guide_audio_segment"):
        missing.append("guide_audio_segment")
    if need_video_ref_merge and not status.get("native_keyframe_ref_merge"):
        missing.append("native_keyframe_ref_merge")
    if need_audio_ref_merge and not status.get("native_keyframe_ref_audio_merge"):
        missing.append("native_keyframe_ref_audio_merge")

    if missing:
        raise RuntimeError(
            "h3_motion_context: this graph requires native MiniMax H3 guide/"
            "MultiRef support from ComfyUI PR #15439, but the live runtime is "
            "missing %s. Ref merge requirements for this conditioning: "
            "video=%s, audio=%s. Capability report: %r"
            % (", ".join(missing), need_video_ref_merge,
               need_audio_ref_merge, status)
        )
    _log_once(
        "native_15439",
        "h3_motion_context: native ComfyUI H3 guide core detected; "
        "required Ref2VA merge paths verified for this conditioning; "
        "no Motion Context core patch installed",
    )
    return True


def ensure_existing_video_compat():
    """Enable #15375-equivalent support only for the masked extension node."""
    from .h3_mask_compat import ensure_h3_mask_compat
    from .h3_mask_payload_compat import ensure_av_mask_payload_compat

    ensure_h3_mask_compat()
    ensure_av_mask_payload_compat()
    from .h3_mask_velocity import ensure_h3_mask_velocity
    ensure_h3_mask_velocity()
    _log_once(
        "existing_video_ready",
        "h3_motion_context: existing-video H3 AV-mask support ready "
        "(native #15375 capabilities are preferred automatically)",
    )
    return True


def ensure_hybrid_compat():
    """Research helper: native guide support plus optional masked extension."""
    ensure_motion_context_compat()
    ensure_existing_video_compat()
    return True


def capability_report():
    from .h3_mask_compat import capability_status as mask_capability_status
    from .h3_mask_payload_compat import capability_status as mask_payload_status
    return {
        "native_guide": native_guide_status(),
        "mask": mask_capability_status(),
        "mask_payload": mask_payload_status(),
    }
