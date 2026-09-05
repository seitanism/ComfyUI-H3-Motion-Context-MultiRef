"""Lazy, self-retiring high-precision MiniMax H3 denoise-mask compatibility.

This module is intentionally *not* imported at custom-node startup.  The V2V
fractional-mask node imports and calls :func:`ensure_h3_mask_precision` only
when that node actually executes.

The live ComfyUI implementation is probed before every first install.  Native
behavior wins independently for each capability:

* near-1 token masks keep at least 1/4096 precision,
* values below exactly 1.0 are not treated as the full-generation shortcut,
* denoise_mask/audio_denoise_mask arrive at the H3 diffusion model as FP32.

When upstream ComfyUI implements equivalent or better behavior, no patch is
installed.  All runtime changes are H3-specific and disappear on restart.
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import textwrap
import types

import torch

_LOG = logging.getLogger("h3_motion_context")
_MARKER = "_h3_motion_context_fractional_mask_precision_v3"
_LEVELS = 4096.0
_PROBE_VALUE = 0.9995
_PROBE_VALUE_FINE = 0.9997
_EXPECTED_PROBE = 4094.0 / 4096.0
_EXPECTED_PROBE_FINE = 4095.0 / 4096.0


def quantize_4096(mask):
    """H3's desired monotonic mask quantizer: ceil to a 1/4096 grid."""
    return torch.ceil(mask * _LEVELS) / _LEVELS


def _mark(fn):
    try:
        setattr(fn, _MARKER, True)
    except Exception:
        pass
    return fn


def _is_ours(fn):
    return bool(getattr(fn, _MARKER, False))


def _signature_has(fn, *names):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in names)


def _probe_model_base_precision(cls, model_base):
    """Behaviorally probe token quantization and condition omission semantics."""
    out = {
        "token_grid_precision": False,
        "near_one_video_condition": False,
        "near_one_audio_condition": False,
        "probe_token_value": None,
        "probe_token_value_09997": None,
        "probe_error": None,
    }
    required = ("_pool_masks_to_token_grid", "_token_grid_masks", "_denoise_mask_values")
    if cls is None or not all(callable(getattr(cls, name, None)) for name in required):
        return out

    try:
        probe = cls.__new__(cls)
        probe.diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))
        # Shapes deliberately use one channel: pack/unpack only needs matching
        # stream shapes for the mask probe, not actual H3 latent channel counts.
        probe_results = []
        condition_results = []
        for requested in (_PROBE_VALUE, _PROBE_VALUE_FINE):
            video = torch.full((1, 1, 1, 2, 2), requested, dtype=torch.float32)
            audio = torch.full((1, 1, 2, 1), requested, dtype=torch.float32)
            packed, shapes = model_base.utils.pack_latents((video, audio))
            grids = cls._token_grid_masks(probe, packed, shapes)
            value = float(grids[0].reshape(-1)[0].item())
            probe_results.append((requested, value))
            vals = cls._denoise_mask_values(probe, packed, shapes)
            condition_results.append((
                isinstance(vals, dict) and "denoise_mask" in vals,
                isinstance(vals, dict) and "audio_denoise_mask" in vals,
            ))

        out["probe_token_value"] = probe_results[0][1]
        out["probe_token_value_09997"] = probe_results[1][1]
        # Probe two values so a 1/2048 grid cannot accidentally pass just
        # because 0.9995 lands on the same ceiling as the 1/4096 target.
        out["token_grid_precision"] = all(
            value < 1.0
            and abs(value - requested) <= (1.0 / _LEVELS) + 1e-7
            for requested, value in probe_results
        )
        out["near_one_video_condition"] = all(v for v, _ in condition_results)
        out["near_one_audio_condition"] = all(a for _, a in condition_results)
    except Exception as exc:
        out["probe_error"] = repr(exc)
    return out


def _probe_video_row_shortcut(h3m):
    fn = getattr(h3m, "mask_row_values", None)
    if not callable(fn):
        return False, "mask_row_values missing"
    try:
        mask = torch.full((1, 2, 2), _EXPECTED_PROBE, dtype=torch.float32)
        rows = fn(mask, 1, 2, 2)
        return bool(rows is not None), None
    except Exception as exc:
        return False, repr(exc)


def _audio_shortcut_status(h3m):
    """Execute the live audio-mask branch with near-one masks, without the network."""
    fn = getattr(getattr(h3m, "MiniMaxH3Model", None), "_forward", None)
    if fn is None:
        return False, "MiniMaxH3Model._forward missing"
    if _is_ours(fn):
        return True, None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "audio_denoise_mask"]
        if len(branches) != 1:
            return False, "unrecognized audio-mask branch"
        probe_tree = ast.parse("def probe(audio_denoise_mask):\n    t_a = 0.25\n    t_pin_a = 0.999\n    audio_rows_t = None\n    seg_t = {'audio': t_a}\n")
        probe_tree.body[0].body.append(branches[0])
        probe_tree.body[0].body.extend(ast.parse(
            "return audio_rows_t if audio_rows_t is not None else seg_t['audio']"
        ).body)
        ast.fix_missing_locations(probe_tree)
        ns = dict(fn.__globals__)
        exec(compile(probe_tree, "<H3 audio shortcut probe>", "exec"), ns)
        for value in (_EXPECTED_PROBE, _EXPECTED_PROBE_FINE):
            result = ns["probe"](torch.full((1, 1, 2, 2), value))
            if result is None:
                return False, None
            if torch.is_tensor(result):
                if result.numel() == 0:
                    return False, None
            elif abs(float(result) - (1.0 - value * 0.75)) > 1e-6:
                return False, None
        return True, None
    except Exception as exc:
        return False, repr(exc)


class _SamplingProbe:
    def calculate_input(self, sigma, x):
        return x

    def timestep(self, t):
        return t

    def calculate_denoised(self, sigma, model_output, x):
        return model_output


class _DiffusionCapture:
    dtype = torch.bfloat16

    def __init__(self):
        self.kwargs = None

    def __call__(self, x, t, **kwargs):
        self.kwargs = kwargs
        # BaseModel accepts a tensor output for this lightweight probe.
        return x


def _probe_fp32_transport(cls):
    """Verify H3 masks arrive at diffusion as unchanged FP32 values."""
    if cls is None or not callable(getattr(cls, "_apply_model", None)):
        return False, "MiniMaxH3._apply_model missing", None
    try:
        probe = cls.__new__(cls)
        capture = _DiffusionCapture()
        probe.diffusion_model = capture
        probe.model_sampling = _SamplingProbe()
        probe.current_patcher = None
        probe.manual_cast_dtype = None
        # Avoid unrelated H3-specific timestep behavior in older builds; this
        # probe is solely about the generic extra-condition transport step.
        probe.process_timestep = types.MethodType(
            lambda self, timestep, **kwargs: timestep, probe
        )
        probe.get_dtype_inference = types.MethodType(
            lambda self: torch.bfloat16, probe
        )
        x = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        t = torch.ones((1,), dtype=torch.float32)
        mask = torch.full((1, 1, 1, 1), _EXPECTED_PROBE, dtype=torch.float32)
        cls._apply_model(
            probe,
            x,
            t,
            denoise_mask=mask,
            audio_denoise_mask=mask.clone(),
        )
        kwargs = capture.kwargs or {}
        video_mask = kwargs.get("denoise_mask")
        audio_mask = kwargs.get("audio_denoise_mask")
        value = (
            float(video_mask.reshape(-1)[0].item())
            if hasattr(video_mask, "reshape") and video_mask.numel()
            else None
        )
        ready = bool(
            getattr(video_mask, "dtype", None) == torch.float32
            and getattr(audio_mask, "dtype", None) == torch.float32
            and value is not None
            and value < 1.0
            and abs(value - _EXPECTED_PROBE) < 1e-7
            and abs(float(audio_mask.reshape(-1)[0].item()) - _EXPECTED_PROBE) < 1e-7
        )
        return ready, None, value
    except Exception as exc:
        return False, repr(exc), None


def capability_status():
    """Return a live capability report; does not modify ComfyUI."""
    import comfy.model_base as model_base
    import comfy.ldm.minimax.model as h3m

    cls = getattr(model_base, "MiniMaxH3", None)
    model = _probe_model_base_precision(cls, model_base)
    video_rows, video_err = _probe_video_row_shortcut(h3m)
    audio_rows, audio_err = _audio_shortcut_status(h3m)
    fp32, fp32_err, fp32_value = _probe_fp32_transport(cls)

    status = {
        **model,
        "token_grid_patch_active": bool(cls and _is_ours(getattr(cls, "_token_grid_masks", None))),
        "mask_values_patch_active": bool(cls and _is_ours(getattr(cls, "_denoise_mask_values", None))),
        "video_row_patch_active": _is_ours(getattr(h3m, "mask_row_values", None)),
        "audio_row_patch_active": _is_ours(getattr(getattr(h3m, "MiniMaxH3Model", None), "_forward", None)),
        "fp32_transport_patch_active": bool(cls and _is_ours(getattr(cls, "_apply_model", None))),
        "video_exact_one_shortcut": video_rows,
        "audio_exact_one_shortcut": audio_rows,
        "fp32_condition_transport": fp32,
        "fp32_probe_value": fp32_value,
        "video_row_probe_error": video_err,
        "audio_row_probe_error": audio_err,
        "fp32_probe_error": fp32_err,
    }
    status["precision_ready"] = bool(
        status["token_grid_precision"]
        and status["near_one_video_condition"]
        and status["near_one_audio_condition"]
        and status["video_exact_one_shortcut"]
        and status["audio_exact_one_shortcut"]
        and status["fp32_condition_transport"]
    )
    return status


def _install_model_base_precision(cls, model_base):
    token_current = getattr(cls, "_token_grid_masks", None)
    values_current = getattr(cls, "_denoise_mask_values", None)
    if not callable(token_current) or not callable(values_current):
        raise RuntimeError(
            "h3_motion_context: current MiniMaxH3 does not expose the native "
            "token-grid mask helpers required for high-precision compatibility"
        )

    if not _is_ours(token_current):
        @functools.wraps(token_current, updated=())
        def token_grid_masks(self, denoise_mask, latent_shapes):
            masks = model_base.utils.unpack_latents(denoise_mask, latent_shapes)
            pooled = self._pool_masks_to_token_grid(masks)
            return [quantize_4096(mask) for mask in pooled]
        _mark(token_grid_masks)
        cls._token_grid_masks = token_grid_masks

    if not _is_ours(values_current):
        @functools.wraps(values_current, updated=())
        def denoise_mask_values(self, denoise_mask, latent_shapes):
            if latent_shapes is None or len(latent_shapes) < 2:
                return {}
            masks = self._token_grid_masks(denoise_mask, latent_shapes)
            out = {}
            if torch.amin(masks[0]).item() < 1.0:
                out["denoise_mask"] = masks[0][:1, :1].clone()
            if torch.amin(masks[1]).item() < 1.0:
                out["audio_denoise_mask"] = masks[1][:1].amax(dim=1, keepdim=True)
            return out
        _mark(denoise_mask_values)
        cls._denoise_mask_values = denoise_mask_values


def _install_video_row_precision(h3m):
    current = getattr(h3m, "mask_row_values", None)
    if not callable(current):
        raise RuntimeError("h3_motion_context: H3 mask_row_values helper not found")
    if _is_ours(current):
        return

    @functools.wraps(current, updated=())
    def mask_row_values(mask, latent_t, lat_h, lat_w):
        m = torch.nn.functional.pad(
            mask,
            (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]),
            mode="replicate",
        )
        m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
        values = m.reshape(-1)
        if bool((values >= 1.0).all()):
            return None
        return values

    _mark(mask_row_values)
    h3m.mask_row_values = mask_row_values


class _AudioShortcutTransformer(ast.NodeTransformer):
    """Replace only the H3 audio near-1 cutoff expression with exact 1.0."""
    def __init__(self):
        self.replacements = 0

    @staticmethod
    def _is_one_minus_1e3(node):
        return (isinstance(node, ast.Constant) and node.value == 0.999) or bool(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and isinstance(node.left, ast.Constant)
            and float(node.left.value) == 1.0
            and isinstance(node.right, ast.Constant)
            and abs(float(node.right.value) - 1e-3) < 1e-12
        )

    def visit_If(self, node):
        self.generic_visit(node)
        text = ast.dump(node.test, include_attributes=False)
        if "audio_denoise_mask" in text:
            return node
        # The actual threshold test is inside the audio branch and references m,
        # not audio_denoise_mask. Limit the rewrite to the exact Compare form.
        test = node.test
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Call)
            and isinstance(test.operand.func, ast.Name)
            and test.operand.func.id == "bool"
            and test.operand.args
        ):
            expr = test.operand.args[0]
            # (m >= 1.0 - 1e-3).all()
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                base = expr.func.value
                if isinstance(base, ast.Compare) and len(base.comparators) == 1:
                    if isinstance(base.left, ast.Name) and base.left.id == "m" and self._is_one_minus_1e3(base.comparators[0]):
                        base.comparators[0] = ast.Constant(value=1.0)
                        self.replacements += 1
        return node


def _install_audio_row_precision(h3m):
    cls = getattr(h3m, "MiniMaxH3Model", None)
    current = getattr(cls, "_forward", None) if cls is not None else None
    if not callable(current):
        raise RuntimeError("h3_motion_context: MiniMaxH3Model._forward not found")
    if _is_ours(current):
        return
    try:
        source = textwrap.dedent(inspect.getsource(current))
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "h3_motion_context: H3 audio near-1 precision still needs a runtime "
            "patch, but MiniMaxH3Model._forward source is unavailable: %r" % (exc,)
        ) from exc

    tree = ast.parse(source)
    tx = _AudioShortcutTransformer()
    tree = tx.visit(tree)
    ast.fix_missing_locations(tree)
    if tx.replacements != 1:
        raise RuntimeError(
            "h3_motion_context: refusing to patch unknown H3 _forward layout; "
            "expected one audio 1e-3 shortcut, found %d" % tx.replacements
        )
    ns = dict(h3m.__dict__)
    exec(compile(tree, getattr(current, "__code__", None).co_filename, "exec"), ns)
    replacement = ns.get("_forward")
    if not callable(replacement):
        raise RuntimeError("h3_motion_context: failed to rebuild H3 _forward precision patch")
    functools.update_wrapper(replacement, current, updated=())
    _mark(replacement)
    cls._forward = replacement


class _ApplyModelFP32Transformer(ast.NodeTransformer):
    """Make only H3 denoise-mask kwargs FP32 inside the live _apply_model body."""
    def __init__(self):
        self.replacements = 0

    def visit_For(self, node):
        self.generic_visit(node)
        if not (isinstance(node.target, ast.Name) and node.target.id == "o"):
            return node
        if not (isinstance(node.iter, ast.Name) and node.iter.id == "kwargs"):
            return node
        body = node.body
        for i, stmt in enumerate(body):
            if not isinstance(stmt, ast.If):
                continue
            dump = ast.dump(stmt.test, include_attributes=False)
            if "hasattr" not in dump or "dtype" not in dump:
                continue

            key_test = ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Name(id="o", ctx=ast.Load()),
                        ops=[ast.In()],
                        comparators=[ast.Tuple(
                            elts=[ast.Constant("denoise_mask"), ast.Constant("audio_denoise_mask")],
                            ctx=ast.Load(),
                        )],
                    ),
                    ast.Call(
                        func=ast.Name(id="hasattr", ctx=ast.Load()),
                        args=[ast.Name(id="extra", ctx=ast.Load()), ast.Constant("dtype")],
                        keywords=[],
                    ),
                ],
            )
            fp32_assign = ast.Assign(
                targets=[ast.Name(id="extra", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="comfy", ctx=ast.Load()),
                            attr="model_management",
                            ctx=ast.Load(),
                        ),
                        attr="cast_to_device",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id="extra", ctx=ast.Load()),
                        ast.Name(id="device", ctx=ast.Load()),
                        ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="float32", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
            replacement = ast.If(test=key_test, body=[fp32_assign], orelse=[stmt])
            body[i] = replacement
            self.replacements += 1
            break
        return node


def _install_fp32_transport(cls, model_base):
    current = getattr(cls, "_apply_model", None)
    if not callable(current):
        raise RuntimeError("h3_motion_context: MiniMaxH3._apply_model not found")
    if _is_ours(current):
        return

    # The method is normally inherited from BaseModel. Transform the live
    # function body instead of shipping a frozen copy, so unrelated upstream
    # changes in _apply_model are preserved.
    try:
        source = textwrap.dedent(inspect.getsource(current))
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "h3_motion_context: FP32 H3 mask transport is required, but the live "
            "_apply_model source is unavailable: %r" % (exc,)
        ) from exc

    tree = ast.parse(source)
    tx = _ApplyModelFP32Transformer()
    tree = tx.visit(tree)
    ast.fix_missing_locations(tree)
    if tx.replacements != 1:
        raise RuntimeError(
            "h3_motion_context: refusing to patch unknown _apply_model layout; "
            "expected one tensor-cast loop, found %d" % tx.replacements
        )

    ns = dict(model_base.__dict__)
    exec(compile(tree, getattr(current, "__code__", None).co_filename, "exec"), ns)
    replacement = ns.get("_apply_model")
    if not callable(replacement):
        raise RuntimeError("h3_motion_context: failed to rebuild H3 FP32 _apply_model")
    functools.update_wrapper(replacement, current, updated=())
    _mark(replacement)
    cls._apply_model = replacement


def ensure_h3_mask_precision():
    """Install only the high-precision H3 capabilities missing at runtime."""
    # First make sure the underlying native/fallback AV-mask engine exists.
    # This call is itself lazy and capability-aware.
    from .h3_compat import ensure_existing_video_compat
    ensure_existing_video_compat()

    import comfy.model_base as model_base
    import comfy.ldm.minimax.model as h3m

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None:
        raise RuntimeError("h3_motion_context: MiniMaxH3 model class not found")

    before = capability_status()
    if before["precision_ready"]:
        _LOG.info(
            "h3_motion_context: native H3 high-precision denoise-mask support detected; no precision patch installed"
        )
        return before

    if not (
        before["token_grid_precision"]
        and before["near_one_video_condition"]
        and before["near_one_audio_condition"]
    ):
        _install_model_base_precision(cls, model_base)

    if not before["video_exact_one_shortcut"]:
        _install_video_row_precision(h3m)

    if not before["audio_exact_one_shortcut"]:
        _install_audio_row_precision(h3m)

    if not before["fp32_condition_transport"]:
        _install_fp32_transport(cls, model_base)

    after = capability_status()
    if not after["precision_ready"]:
        raise RuntimeError(
            "h3_motion_context: H3 high-precision mask compatibility is incomplete after patching. "
            "Before=%r After=%r" % (before, after)
        )

    _LOG.info(
        "h3_motion_context: H3 fractional-mask precision compatibility enabled lazily "
        "(1/4096 grid, exact-1 shortcut, FP32 mask transport)"
    )
    return after
