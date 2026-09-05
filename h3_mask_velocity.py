"""Lazy PR #15988 velocity conversion compatibility for H3 masked sampling.

Derived from poorpaper's ComfyUI PR #15988 (GPL-3.0). Preserve native wrapper
execution and apply mask scaling before audio carry conversion, never after it.
"""
import ast
import functools
import inspect
import logging
import textwrap

import torch

_LOG = logging.getLogger("h3_motion_context")
_MARKER = "_h3_mask_velocity_15988"


def _probe(h3m):
    """Run the live forward with a tiny fake network, including audio carry."""
    cls = h3m.MiniMaxH3Model
    probe = cls.__new__(cls)
    if isinstance(probe, torch.nn.Module):
        torch.nn.Module.__init__(probe)
    probe.sigma_shift_video = 12.0
    probe.sigma_shift_audio = 3.0
    v = torch.full((1, 2, 1, 2, 2), 2.0)
    a = torch.full((1, 2, 2, 3), 3.0)
    vm = torch.tensor([[[[[1.0, 0.75], [0.5, 0.0]]]]])
    am = torch.tensor([[[[1.0, 0.5, 0.25], [0.75, 0.5, 0.0]]]])
    probe._forward = lambda *args, **kwargs: [v.clone(), a.clone()]
    sigma = torch.tensor([0.5])
    for scale in (1.0, 4.0):
        x_audio = torch.full_like(a, 2.0)
        result = cls.forward(probe, [torch.zeros_like(v), x_audio], sigma * 1000,
                             torch.empty(1, 1, 1), transformer_options={},
                             minimax_payload={"audio_scale": scale},
                             denoise_mask=vm, audio_denoise_mask=am)
        expected_a = a * am
        if scale != 1.0:
            sigma_a = h3m.time_shift_sigma(sigma[0], 12.0, 3.0)
            expected_a = ((1.0 - scale) * x_audio * (sigma_a / sigma[0])
                          + (1.0 + (scale - 1.0) * sigma_a) * expected_a)
        if not torch.allclose(result[0], v * vm) or not torch.allclose(result[1], expected_a):
            return False
    return True


def capability_status():
    import comfy.ldm.minimax.model as h3m
    try:
        ready = _probe(h3m)
        error = None
    except Exception as exc:
        ready, error = False, repr(exc)
    return {"velocity_ready": ready, "probe_error": error,
            "patch_active": bool(getattr(h3m.MiniMaxH3Model.forward, _MARKER, False))}


def ensure_h3_mask_velocity():
    """Install only on a recognized unfixed forward; verify or roll back."""
    import comfy.ldm.minimax.model as h3m
    before = capability_status()
    if before["velocity_ready"]:
        return before
    if before["probe_error"]:
        raise RuntimeError("Cannot verify H3 mask velocity compatibility: " + before["probe_error"])
    cls = h3m.MiniMaxH3Model
    current = cls.forward
    if getattr(current, _MARKER, False):
        raise RuntimeError("H3 velocity patch no longer passes its behavioral check; restart/update ComfyUI.")
    if current.__code__.co_freevars:
        raise RuntimeError("Refusing to patch a closure-wrapped H3 forward; update ComfyUI for PR #15988.")
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(current)))
    except (OSError, TypeError) as exc:
        raise RuntimeError("H3 forward source is unavailable; update ComfyUI for PR #15988.") from exc
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.name != "forward" or fn.decorator_list:
        raise RuntimeError("Unrecognized H3 forward; refusing velocity compatibility edit.")
    executions = [i for i, stmt in enumerate(fn.body)
                  if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                  and isinstance(stmt.value.func, ast.Attribute)
                  and stmt.value.func.attr == "execute"]
    conversions = [i for i, stmt in enumerate(fn.body)
                   if isinstance(stmt, ast.If) and "id='scale'" in ast.dump(stmt.test)
                   and all(token in ast.dump(stmt) for token in
                           ("id='out'", "id='audio_src'", "id='carry'"))]
    if len(executions) != 1 or len(conversions) != 1 or conversions[0] <= executions[0]:
        raise RuntimeError("Unrecognized H3 forward/carry boundary; update ComfyUI for PR #15988.")
    i = conversions[0]
    # Supports both the original direct output and newer malloc-graph copying.
    # Scale after graph outputs are copied, but before audio carry conversion.
    suffix = ast.dump(ast.Module(body=fn.body[executions[0]+1:], type_ignores=[]))
    if "id='denoise_mask'" in suffix or "id='audio_denoise_mask'" in suffix:
        raise RuntimeError("Unrecognized existing H3 velocity mask conversion; refusing double scaling.")
    addition = ast.parse('''
if denoise_mask is not None:
    out[0] = out[0] * denoise_mask
if audio_denoise_mask is not None:
    out[1] = out[1] * audio_denoise_mask
''').body
    fn.body[i:i] = addition
    ast.fix_missing_locations(tree)
    namespace = dict(current.__globals__)
    exec(compile(tree, current.__code__.co_filename, "exec"), namespace)
    replacement = namespace["forward"]
    functools.update_wrapper(replacement, current, updated=())
    setattr(replacement, _MARKER, True)
    cls.forward = replacement
    after = capability_status()
    if not after["velocity_ready"]:
        cls.forward = current
        raise RuntimeError("H3 velocity compatibility failed verification and was rolled back: %r" % after)
    _LOG.info("H3 masked velocity compatibility enabled lazily (PR #15988)")
    return after
