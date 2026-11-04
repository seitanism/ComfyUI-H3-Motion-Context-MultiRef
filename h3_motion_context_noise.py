"""H3 Motion Context Noise.

Degradation protection for chained H3 clips, built to sit right after
MiniMaxH3GeneratedAVMaskedContext (or any node that copies the previous clip's
latent into the head of the target and locks it with noise_mask 0).

Wire it:

    Generated AV Masked Context -> H3 Motion Context Noise -> sampler
    model ----------------------> H3 Motion Context Noise -> guider
    previous clip's latent -----> source_latent (the same wire the masked
                                  context node gets)

It does three things, each with its own switch, all under one
degradation_protection switch:

1. Catch. The first latent of every clip that comes in on source_latent is
   saved to catch_folder, numbered in order (1, 2, 3...). A clip that comes in
   again unchanged (a re-roll) is recognised by its fingerprint and keeps its
   number. Only the reset button deletes caught latents.

2. Latent 0 protection. The held latent 0 is swapped for the caught latent 0
   of source_shot (shot 1 by default: the clean start of the chain), still
   locked. Every clip then renders against the original clean frame instead
   of a frame that has been copied clip after clip.

3. Context noise. The held latents get part of the sampler's noise instead of
   being handed over fully clean, fading to none at the join. H3's partial
   masks do this: a mask value m puts a held latent at m x the current noise
   level. The ethanfel fade (default) is ethanfel's drift_control recipe: full
   weight, a 4-latent taper to clean at the join, and at every step the values
   are scaled by next sigma / current sigma.

After sampling, the real held latents are put back into the output, so the join
and anything that reads the head stay exactly as before.
"""

import hashlib
import json
import logging
import os
import re

import torch

import comfy.nested_tensor
import comfy.patcher_extension

try:
    import folder_paths
except ImportError:  # standalone tests
    folder_paths = None

_LOG = logging.getLogger("h3_motion_context_noise")
WHO = "H3 Motion Context Noise"
FADES = ["ethanfel", "linear", "ease"]
_FOLDER_RE = re.compile(r"^[A-Za-z0-9_\-]+(/[A-Za-z0-9_\-]+)*$")


# ------------------------------------------------------------------ latents
def _streams(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        return list(samples.unbind()), True
    if isinstance(samples, (list, tuple)):
        return list(samples), True
    return [samples], False


def _masks(latent, parts):
    old = latent.get("noise_mask")
    if old is None:
        return None
    if hasattr(old, "unbind"):
        masks = list(old.unbind())
    elif isinstance(old, (list, tuple)):
        masks = list(old)
    else:
        masks = [old]
    return masks if len(masks) == len(parts) else None


def _held_steps(video, vmask):
    """Video latent steps the mask locks (max < 0.5), counting from 0 while they stay locked."""
    if vmask is None or vmask.ndim != 5 or vmask.shape[2] != video.shape[2]:
        return []
    held = []
    for t in range(int(video.shape[2])):
        if float(vmask[:, :, t:t + 1].max()) < 0.5:
            held.append(t)
        else:
            break
    return held


def _fingerprint(video):
    v = video.detach().float().cpu().contiguous()
    h = hashlib.sha1(v.numpy().tobytes())
    h.update(str(tuple(v.shape)).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ catch folder
def _catch_dir(folder):
    folder = (folder or "").strip().replace("\\", "/").strip("/")
    if not _FOLDER_RE.match(folder) or ".." in folder.split("/"):
        raise ValueError(f"{WHO}: catch_folder must be a plain folder name like h3_latent0_catch, got {folder!r}")
    base = folder_paths.get_output_directory() if folder_paths is not None else os.getcwd()
    return os.path.join(base, folder)


def _index(path):
    try:
        with open(os.path.join(path, "index.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and isinstance(data.get("shots"), list) else {"shots": []}
    except (OSError, ValueError):
        return {"shots": []}


def _save_tensor(t, p):
    try:
        from safetensors.torch import save_file
        save_file({"latent0": t.detach().cpu().contiguous()}, p)
    except ImportError:
        torch.save(t.detach().cpu(), p)


def _load_tensor(p):
    try:
        from safetensors.torch import load_file
        return load_file(p)["latent0"]
    except ImportError:
        return torch.load(p)


def catch(folder, source_video):
    """Save source_video's latent 0 under the next shot number, unless this exact clip was caught before.
    Returns the clip's shot number."""
    path = _catch_dir(folder)
    os.makedirs(path, exist_ok=True)
    idx = _index(path)
    fp = _fingerprint(source_video)
    for s in idx["shots"]:
        if s.get("fingerprint") == fp:
            return int(s["shot"])
    n = max([int(s["shot"]) for s in idx["shots"]] or [0]) + 1
    _save_tensor(source_video[:, :, 0:1].clone(), os.path.join(path, f"shot_{n:05d}.safetensors"))
    idx["shots"].append({"shot": n, "fingerprint": fp})
    with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
    return n


def load_caught(folder, shot):
    """Caught latent 0 of `shot`, or (latent 0 of the latest caught shot, its number) when `shot` isn't there."""
    path = _catch_dir(folder)
    idx = _index(path)
    shots = sorted(int(s["shot"]) for s in idx["shots"])
    if not shots:
        return None, None
    use = int(shot) if int(shot) in shots else shots[-1]
    p = os.path.join(path, f"shot_{use:05d}.safetensors")
    if not os.path.exists(p):
        return None, None
    return _load_tensor(p), use


def reset(folder):
    """Delete everything this node caught in `folder` (only its own files)."""
    path = _catch_dir(folder)
    if not os.path.isdir(path):
        return 0
    n = 0
    for name in os.listdir(path):
        if re.match(r"^shot_\d{5}\.safetensors$", name) or name == "index.json":
            os.remove(os.path.join(path, name))
            n += 1
    try:
        os.rmdir(path)
    except OSError:
        pass
    return n


# ------------------------------------------------------------------ noise ramp
def fade_weights(n, fade):
    """Weight per held latent, first (oldest) to last (the join)."""
    out = []
    for i in range(n):
        if fade == "ethanfel":
            k = n - 1 - i                      # 0 = the latent at the join
            w = 1.0 if k >= 4 else k / 4.0     # full, then .75 / .5 / .25 / 0
        else:
            pos = i / max(1, n - 1)
            w = 1.0 - pos if fade == "linear" else (1.0 - pos) ** 2
        out.append(w)
    return out


# ------------------------------------------------------------------ model patches
def _restore_wrapper(originals):
    """OUTER_SAMPLE: put the real held latents back into the output after sampling."""
    def wrapper(executor, *args, **kwargs):
        out = executor(*args, **kwargs)
        try:
            import comfy.utils
            shapes = kwargs.get("latent_shapes")
            multi = shapes is not None and len(shapes) > 1
            parts = [p.clone() for p in comfy.utils.unpack_latents(out, shapes)] if multi else [out.clone()]
            v = parts[0]
            for t, orig in originals.items():
                if v.ndim == 5 and t < v.shape[2] and orig.shape[1] == v.shape[1] and orig.shape[3:] == v.shape[3:]:
                    v[:, :, t:t + 1] = orig.to(device=v.device, dtype=v.dtype)
            parts[0] = v
            out = comfy.utils.pack_latents(parts)[0] if multi else parts[0]
        except Exception as e:  # never lose a finished render over this
            _LOG.warning("%s: couldn't put the held latents back (%s); output left as sampled", WHO, e)
        return out
    return wrapper


def _sigma_ratio_fn(values, video_shape, prev):
    """denoise_mask_function: held mask values x next_sigma / current_sigma (ethanfel's matched_noise_ratio)."""
    _, C, T, H, W = (int(x) for x in video_shape)
    n_video = C * T * H * W

    def fn(sigma, mask, extra_options):
        if prev is not None:
            mask = prev(sigma, mask, extra_options)
        sigmas = extra_options.get("sigmas")
        if sigmas is None:
            return mask
        sg = sigmas.float().cpu()
        s = float(torch.as_tensor(sigma).flatten()[0])
        i = int(torch.argmin((sg - s).abs()))
        nxt = float(sg[i + 1]) if i + 1 < len(sg) else 0.0
        r = max(0.0, min(1.0, nxt / s)) if s > 0 else 0.0
        mask = mask.contiguous().clone()
        if mask.ndim == 3 and mask.shape[-1] >= n_video:          # packed AV mask [B, 1, N]
            vm = mask[:, :, :n_video].reshape(mask.shape[0], C, T, H, W).clone()
            for t, v in values.items():
                vm[:, :, t] = v * r
            mask = torch.cat([vm.reshape(mask.shape[0], 1, n_video), mask[:, :, n_video:]], dim=-1)
        elif mask.ndim == 5:                                       # video-only mask
            for t, v in values.items():
                mask[:, :, t] = v * r
        return mask
    return fn


# ------------------------------------------------------------------ node
class MiniMaxH3MotionContextNoise:
    CATEGORY = "conditioning/minimax"
    RETURN_TYPES = ("LATENT", "MODEL")
    RETURN_NAMES = ("latent", "model")
    FUNCTION = "apply"
    DESCRIPTION = ("Degradation protection for chained H3 clips. Put it right after Generated AV Masked Context, wire "
                   "the previous clip's latent into source_latent and the model through to the guider. Catches the "
                   "first latent of every clip, locks source_shot's clean latent 0 into the held head (latent 0 "
                   "protection), and gives the held latents part of the noise, fading to clean at the join (context "
                   "noise). The real held latents are put back after sampling.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "From Generated AV Masked Context (held head locked with noise_mask 0)."}),
                "model": ("MODEL", {"tooltip": "Plug the model through to the guider: the real held latents are put "
                                               "back after sampling, and the ethanfel fade works per step."}),
                "degradation_protection": ("BOOLEAN", {"default": True, "tooltip": "Off = everything passes through "
                                                       "(clips are still caught, so the numbers stay right)."}),
                "latent_0_protection": ("BOOLEAN", {"default": True, "tooltip": "Lock source_shot's clean latent 0 "
                                                    "into the held head."}),
                "context_noise": ("BOOLEAN", {"default": True, "tooltip": "Give the held latents part of the noise, "
                                              "fading to clean at the join."}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Noise on the first held latent (1 = the same noise level as "
                                                        "the new frames). Fades to 0 at the join."}),
                "fade": (FADES, {"default": "ethanfel", "tooltip": "ethanfel = his drift_control recipe: full weight, "
                                 "a 4-latent taper to clean at the join, scaled by next/current sigma every step. "
                                 "linear / ease = a straight / eased ramp over the held latents."}),
                "source_shot": ("INT", {"default": 1, "min": 1, "max": 100000, "tooltip": "Which caught shot's "
                                        "latent 0 to lock in. 1 = the clean start of the chain. Set it to the first "
                                        "shot of a new scene after a scene change."}),
                "catch_folder": ("STRING", {"default": "h3_latent0_catch", "tooltip": "Folder in output/ where the "
                                            "caught latents are kept. Use the reset button (or a new folder) for a "
                                            "new chain."}),
            },
            "optional": {
                "source_latent": ("LATENT", {"tooltip": "The previous clip's latent: the same wire Generated AV "
                                                        "Masked Context gets. Its first latent is caught."}),
            },
        }

    def apply(self, latent, model, degradation_protection=True, latent_0_protection=True, context_noise=True,
              noise_strength=1.0, fade="ethanfel", source_shot=1, catch_folder="h3_latent0_catch",
              source_latent=None):
        parts, nested = _streams(latent)
        video = parts[0]

        # 1. catch, always (so the numbers stay right even with protection off)
        if source_latent is not None:
            src = _streams(source_latent)[0][0]
            if src.ndim == 4:
                src = src.unsqueeze(0)
            n = catch(catch_folder, src)
            print(f"[{WHO}] previous clip caught as shot {n} in {catch_folder}")

        if not degradation_protection or (not latent_0_protection and not context_noise):
            return (latent, model)
        if video.ndim != 5:
            print(f"[{WHO}] expected a 5-D video latent, got {tuple(video.shape)} - passed through")
            return (latent, model)
        masks = _masks(latent, parts)
        held = _held_steps(video, masks[0] if masks else None)
        if not held:
            print(f"[{WHO}] nothing held (first clip?) - passed through")
            return (latent, model)

        video = video.clone()
        vmask = masks[0].clone()
        originals = {}

        # 2. latent 0 protection
        if latent_0_protection:
            lat0, used = load_caught(catch_folder, source_shot)
            if lat0 is None:
                print(f"[{WHO}] nothing caught yet in {catch_folder} - latent 0 protection skipped")
            elif lat0.shape[1] != video.shape[1] or tuple(lat0.shape[3:]) != tuple(video.shape[3:]):
                print(f"[{WHO}] caught latent {tuple(lat0.shape)} doesn't match this clip {tuple(video.shape)} "
                      "(resolution change?) - latent 0 protection skipped")
            else:
                if used != int(source_shot):
                    print(f"[{WHO}] shot {source_shot} isn't caught yet - using shot {used} instead")
                originals[0] = video[:, :, 0:1].clone()
                video[:, :, 0:1] = lat0.to(device=video.device, dtype=video.dtype)
                print(f"[{WHO}] latent 0 locked to shot {used}'s clean latent 0")

        # 3. context noise
        values = {}
        if context_noise and float(noise_strength) > 0:
            ramp = held[1:] if latent_0_protection else held        # protected latent 0 stays clean
            for t, w in zip(ramp, fade_weights(len(ramp), fade)):
                m = float(noise_strength) * w
                if m <= 0:
                    continue
                vmask[:, :, t:t + 1] = m
                values[t] = round(m, 4)
                originals.setdefault(t, video[:, :, t:t + 1].clone())
            if values:
                print(f"[{WHO}] context noise ({fade}) on held latents: {values}")

        if not originals:
            return (latent, model)
        new_parts = [video] + parts[1:]
        new_masks = [vmask] + masks[1:]
        out = dict(latent)
        if nested:
            out["samples"] = comfy.nested_tensor.NestedTensor(tuple(new_parts))
            out["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(new_masks))
        else:
            out["samples"], out["noise_mask"] = video, vmask

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "h3_motion_context_noise_restore",
                               _restore_wrapper(originals))
        if values and fade == "ethanfel":
            m.model_options["denoise_mask_function"] = _sigma_ratio_fn(
                values, video.shape, m.model_options.get("denoise_mask_function"))
        return (out, m)


# ------------------------------------------------------------------ reset button route
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/h3_motion_context_noise/reset")
    async def _reset_route(request):
        origin, host = request.headers.get("Origin"), request.headers.get("Host")
        if origin and host and origin.split("://", 1)[-1] != host:
            return web.json_response({"error": "cross-origin request refused"}, status=403)
        try:
            data = await request.json()
            n = reset(str(data.get("catch_folder", "")))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
        print(f"[{WHO}] reset: deleted {n} caught file(s)")
        return web.json_response({"deleted": n})
except Exception:  # no server (tests) - the button just won't do anything
    pass


NODE_CLASS_MAPPINGS = {"MiniMaxH3MotionContextNoise": MiniMaxH3MotionContextNoise}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3MotionContextNoise": "H3 Motion Context Noise"}
