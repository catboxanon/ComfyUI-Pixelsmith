import math
import numpy as np
import torch
import torch.nn.functional as F

from comfy.model_sampling import EPS, ModelSamplingDiscrete
from comfy.samplers import KSAMPLER, KSamplerX0Inpaint


class ModelSamplingType(ModelSamplingDiscrete, EPS):
    pass


def crop(tensor: torch.Tensor, i: int, j: int, patch_size: int):
    p = patch_size // 2
    return tensor[..., i - p : i + p, j - p : j + p]


def crop_assign(tensor: torch.Tensor, value: torch.Tensor, i: int, j: int, patch_size: int):
    p = patch_size // 2
    tensor[..., i - p : i + p, j - p : j + p] = value


def get_value_coordinates(tensor: torch.Tensor, patch_size: int):
    _, _, h, w = tensor.shape
    tensor = tensor[0, 0]
    value_indices = torch.nonzero(tensor == tensor.max(), as_tuple=False)
    random_indices = value_indices[torch.randperm(value_indices.size(0))][0]

    p = patch_size // 2
    i = torch.clamp(random_indices, p, h - p).tolist()[0]
    j = torch.clamp(random_indices, p, w - p).tolist()[1]

    return (i, j)


@torch.no_grad()
def create_gradient_border(mask, gradient_width=5):
    mask = mask.float().to("cpu")
    inverted_mask = mask
    distances = F.conv2d(inverted_mask, torch.ones(1, 1, 1, 1, device="cpu"), padding=0)
    distance_mask = distances <= gradient_width
    kernel_size = gradient_width * 2 + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device="cpu") / (kernel_size**2)
    padded_mask = F.pad(inverted_mask, (gradient_width, gradient_width, gradient_width, gradient_width), mode="reflect")
    smoothed_distances = F.conv2d(padded_mask, kernel, padding=0).clamp(0, 1)
    smoothed_mask = (mask + (1 - mask) * smoothed_distances * distance_mask.float()).clamp(0, 1)
    return smoothed_mask


def max_denoise(model_sampling, sigmas):
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def combine_hires_guided(hires: torch.Tensor, guided: torch.Tensor):
    rfft2_fn = lambda t: torch.fft.rfft2(t, dim=(-2, -1), norm="ortho")
    irfft2_fn = lambda t: torch.fft.irfft2(t, dim=(-2, -1), norm="ortho")
    phi_fn = lambda f: torch.exp(1j * torch.angle(f))

    hires_fft = rfft2_fn(hires)
    hires_phi = phi_fn(hires_fft)

    guided_fft = rfft2_fn(guided)
    guided_phi = phi_fn(guided_fft)

    result_amp = torch.abs(hires_fft)
    result_phase = phi_fn(guided_phi + hires_phi)

    return irfft2_fn(result_amp * result_phase)


def apply_chess_mask(hires_pred: torch.Tensor, guided_pred: torch.Tensor, patch_size: int):
    shift_left = torch.rand(1).item() < 0.5
    shift_down = torch.rand(1).item() < 0.5

    mask_first_row = torch.zeros(1, patch_size)
    mask_first_row[:, ::2] = 1
    mask_second_row = torch.roll(mask_first_row, shifts=1, dims=1)

    stacked_rows = torch.cat((mask_first_row, mask_second_row), dim=-2)
    den_mask = torch.tile(stacked_rows, (patch_size // stacked_rows.shape[0], 1))
    den_mask = den_mask[np.newaxis, np.newaxis, ...].to(torch.float16)
    den_mask = torch.roll(den_mask, shifts=(-1 if shift_down else 0, -1 if shift_left else 0), dims=(2, 3))

    return hires_pred * (1 - den_mask) + guided_pred * den_mask


class PixelsmithSampler(KSAMPLER):
    def __init__(self, original_sampler: KSAMPLER):
        super().__init__(
            original_sampler.sampler_function,
            original_sampler.extra_options,
            original_sampler.inpaint_options,
        )

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False):  # TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
        )

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            **self.extra_options,
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples
