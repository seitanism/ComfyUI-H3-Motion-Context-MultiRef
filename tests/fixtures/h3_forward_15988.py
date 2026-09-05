"""Small upstream forward snapshots for PR #15988 contract tests.
Source: Comfy-Org/ComfyUI and poorpaper/ComfyUI PR #15988; GPL-3.0.
No network or model weights required.
"""
import torch
import comfy.patcher_extension

def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)

class LegacyH3(torch.nn.Module):
    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
        scale = float((minimax_payload or {}).get('audio_scale', 1.0))
        audio_src = x[1]
        if scale != 1.0:
            shift_v = float(transformer_options.get('minimax_h3_sigma_shift_video', self.sigma_shift_video))
            shift_a = float(transformer_options.get('minimax_h3_sigma_shift_audio', self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-06)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x = [x[0], audio_src * carry]
        compile_allocations = comfy.model_prefetch.malloc_graph_enabled(x[0].device)
        if compile_allocations:
            out = [torch.empty_like(x[0]), torch.empty_like(x[1])]
            comfy.model_prefetch.malloc_graph_begin(self, x[0].device)
        graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(self._forward, self, comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
        if compile_allocations:
            out[0].copy_(graph_out[0])
            out[1].copy_(graph_out[1])
            del graph_out
            comfy.model_prefetch.malloc_graph_end()
        else:
            out = graph_out
        if scale != 1.0:
            out[1] = (1.0 - scale) * (audio_src * carry) + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1]
        return out

class FixedH3(torch.nn.Module):
    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
        scale = float((minimax_payload or {}).get('audio_scale', 1.0))
        audio_src = x[1]
        if scale != 1.0:
            shift_v = float(transformer_options.get('minimax_h3_sigma_shift_video', self.sigma_shift_video))
            shift_a = float(transformer_options.get('minimax_h3_sigma_shift_audio', self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-06)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x = [x[0], audio_src * carry]
        out = comfy.patcher_extension.WrapperExecutor.new_class_executor(self._forward, self, comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
        if denoise_mask is not None:
            out[0] = out[0] * denoise_mask
        if audio_denoise_mask is not None:
            out[1] = out[1] * audio_denoise_mask
        if scale != 1.0:
            out[1] = (1.0 - scale) * (audio_src * carry) + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1]
        return out
