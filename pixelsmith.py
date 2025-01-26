from time import sleep
from typing import Callable
import torch
from tqdm.auto import tqdm

from comfy.model_base import SDXL
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from nodes import MAX_RESOLUTION
from .utils import (
    ModelSamplingType,
    PixelsmithSampler,
    apply_chess_mask,
    combine_hires_guided,
    create_gradient_border,
    crop,
    crop_assign,
    get_value_coordinates,
    max_denoise,
)


# NOTE: Based on SamplerCustomAdvanced
#
# Pixelsmith is based on https://github.com/Thanos-DB/Pixelsmith/blob/f15d725d19a782ae340c7c1db6cd301550fcf6cf/pixelsmith_pipeline.py#L1715
# licensed under GPL v3
class Pixelsmith:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "slider": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}),
                "patch_size": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(
        self,
        noise: Noise_RandomNoise,
        guider: comfy.samplers.CFGGuider,
        sampler: comfy.samplers.KSAMPLER,
        sigmas: torch.Tensor,
        latent_image,
        slider: float,
        patch_size: int,
    ):
        model: SDXL = guider.model_patcher.model
        ms: ModelSamplingType = guider.model_patcher.model.model_sampling
        p_sampler = PixelsmithSampler(sampler)
        slider_step = int((len(sigmas) - 1) * slider)

        patch_size = patch_size // 8

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Pixelsmith
        guided_noise = noise.generate_noise(latent)
        guided_at = lambda step: ms.noise_scaling(sigmas[step], guided_noise, latent_image, max_denoise(ms, sigmas))
        hires = guided_at(0)
        hires_prev = hires.clone()

        _, _, h, w = hires.shape
        sigmas_map = torch.ones((1, 1, h, w)) * sigmas.max()
        step_fn: Callable[[], int] = lambda: torch.where(sigmas_map.max() == sigmas)[0].item()  # type: ignore
        step = step_fn()
        # guided image generation before slider
        with tqdm(total=slider_step, disable=disable_pbar) as pbar:
            while step < slider_step:
                pbar.update(step - pbar.n)
                # dbg = guided_at(step)
                # # dbg = model.process_latent_in(dbg)
                # callback(step, dbg, dbg, len(sigmas) - 1)
                # dbg = model.process_latent_out(dbg)
                # sleep(0.05)
                # step += 1
                # continue

                new_step = step_fn()
                if new_step != step:
                    step = new_step

                i, j = get_value_coordinates(sigmas_map, patch_size)
                crop_fn: Callable[[torch.Tensor], torch.Tensor] = lambda t: crop(t, i, j, patch_size)
                crop_assign_fn: Callable[[torch.Tensor, torch.Tensor]] = lambda t, v: crop_assign(
                    t, v, i, j, patch_size
                )

                # Random patch cropping
                guided_crop = crop_fn(guided_at(step))
                guided_pred_crop = crop_fn(guided_at(step + 1))
                hires_crop = crop_fn(hires)
                hires_prev_crop = crop_fn(hires_prev)
                sigmas_crop = crop_fn(sigmas_map)

                hires_next_crop = combine_hires_guided(hires_crop, guided_crop)

                uniques = torch.unique(sigmas_crop)
                vmax = uniques[-1]
                time_mask = torch.where(sigmas_crop == vmax, 1, 0)
                if len(uniques) > 1:
                    hires_next_crop = hires_next_crop * time_mask + hires_prev_crop * (time_mask == 0)

                hires_pred_crop = guider.sample(
                    noise.generate_noise({"samples": hires_next_crop}),
                    hires_next_crop,
                    p_sampler,
                    sigmas[step : step + 2],
                    denoise_mask=noise_mask,
                    callback=callback,
                    disable_pbar=True,
                    seed=noise.seed,
                ).to("cpu")

                smoothed_time_mask = create_gradient_border(time_mask, gradient_width=10)
                full_replace_mask = smoothed_time_mask == 1
                no_replace_mask = smoothed_time_mask == 0
                gradient_mask = (smoothed_time_mask > 0) & (smoothed_time_mask < 1)

                hires_pred_crop = apply_chess_mask(hires_pred_crop, guided_pred_crop, patch_size)

                crop_assign_fn(
                    hires,
                    (
                        full_replace_mask * hires_pred_crop
                        + no_replace_mask * hires_crop
                        + gradient_mask * (hires_pred_crop + hires_crop) / 2
                    ),
                )

                crop_assign_fn(
                    sigmas_map,
                    torch.where(
                        sigmas_crop == sigmas_crop.max(),
                        torch.ones_like(sigmas_crop) * sigmas[step + 1],
                        sigmas_crop,
                    ),
                )

                callback(step, hires_prev, hires_prev, len(sigmas))

                if torch.all(sigmas_map == sigmas_map.max()):
                    hires_prev = model.process_latent_out(hires.clone())

        # pure generation after slider
        # should probably be reverted to segmented approach from original Pixelsmith for memory efficiency
        samples = guider.sample(
            guided_noise,
            hires_prev,
            sampler,
            sigmas[slider_step:],
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        # samples = dbg
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)
