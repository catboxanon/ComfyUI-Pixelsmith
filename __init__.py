import torch
import torch.nn.functional as F
import numpy as np

import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

# NOTE: Based on SamplerCustomAdvanced
class Pixelsmith:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE", ),
                "guider": ("GUIDER", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_size": ("INT", {"default": 128, "min": 1, "max": 512, "step": 64}),
            }
        }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    # https://github.com/Thanos-DB/Pixelsmith/blob/f15d725d19a782ae340c7c1db6cd301550fcf6cf/pixelsmith_pipeline.py#L1180
    def random_crop(self, z, i, j, patch_size):
        p = patch_size//2
        return z[..., i-p:i+p, j-p:j+p]

    # https://github.com/Thanos-DB/Pixelsmith/blob/f15d725d19a782ae340c7c1db6cd301550fcf6cf/pixelsmith_pipeline.py#L1184
    def get_value_coordinates(self, tensor):
        value_indices = torch.nonzero(tensor == tensor.max(), as_tuple=False)
        random_indices = value_indices[torch.randperm(value_indices.size(0))]
        return random_indices

    # https://github.com/Thanos-DB/Pixelsmith/blob/f15d725d19a782ae340c7c1db6cd301550fcf6cf/pixelsmith_pipeline.py#L1729
    @torch.no_grad()
    def create_gradient_border(self, mask, gradient_width=5):
        mask = mask.float().to('cpu')
        inverted_mask = mask
        distances = F.conv2d(inverted_mask, torch.ones(1, 1, 1, 1, device='cpu'), padding=0)
        distance_mask = distances <= gradient_width
        kernel_size = gradient_width * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device='cpu') / (kernel_size ** 2)
        padded_mask = F.pad(inverted_mask, (gradient_width, gradient_width, gradient_width, gradient_width), mode='reflect')
        smoothed_distances = F.conv2d(padded_mask, kernel, padding=0).clamp(0, 1)
        smoothed_mask = (mask + (1 - mask) * smoothed_distances * distance_mask.float()).clamp(0, 1)
        return smoothed_mask

    def sample(self, noise: Noise_RandomNoise, guider: comfy.samplers.CFGGuider, sampler: comfy.samplers.Sampler, sigmas, latent_image, denoise, patch_size):
        # Almost all code in this function is from here, with adapations to work with ComfyUI:
        # https://github.com/Thanos-DB/Pixelsmith/blob/f15d725d19a782ae340c7c1db6cd301550fcf6cf/pixelsmith_pipeline.py#L1715

        # NOTE: The "Slider" parameter in the original paper is just a denoise parameter where the steps are fixed to 50 (diffusers default).
        #       Ideally we would not need `denoise` passed in at all, but sigmas that are passed in do not contain such information.
        slider = int(50 * denoise)

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

        # TODO: Verify this is correct
        #       I think this is just setting up the latents for the respective # of timesteps and not much else.
        guid_latents = [latent_image for _ in range(sigmas.size(0))]
        latents = guid_latents[0]

        b, c, latent_size_h, latent_size_w = latents.shape
        times = torch.ones((1, 1, latent_size_h, latent_size_w)).int().to('cpu') * sigmas.max().item()
        p = patch_size // 2
        prev_latents = latents.clone()

        while times.float().mean() >= 0:
            random_indices = self.get_value_coordinates(times[0, 0])[0]
            i = torch.clamp(random_indices, p, latent_size_h - p).tolist()[0]
            j = torch.clamp(random_indices, p, latent_size_w - p).tolist()[1]

            # Random patch cropping
            sub_latents = self.random_crop(latents, i, j, patch_size)
            sub_prev_latents = self.random_crop(prev_latents, i, j, patch_size)
            sub_time = self.random_crop(times, i, j, patch_size)

            t = times.max()
            ii = torch.where(t == sigmas)[0].item()

            if ii < slider:
                sub_guid_latents = self.random_crop(guid_latents[ii], i, j, patch_size) # type: ignore
            if ii < len(guid_latents) - 1 and ii < slider:
                sub_guid_latents_ahead = self.random_crop(guid_latents[ii + 1], i, j, patch_size) # type: ignore

            if sub_time.float().mean() >= 0:
                fft_sub_latents = torch.fft.rfft2(sub_latents, dim=(-2, -1), norm='ortho')
                fft_sub_guid_latents = torch.fft.rfft2(sub_guid_latents, dim=(-2, -1), norm='ortho')
                magnitude_latents = torch.abs(fft_sub_latents)
                complex_latents = torch.exp(1j * torch.angle(fft_sub_latents))
                complex_guid_latents = torch.exp(1j * torch.angle(fft_sub_guid_latents))
                if ii < slider:
                    mixed_phase = torch.angle(complex_latents + complex_guid_latents)
                else:
                    mixed_phase = torch.angle(fft_sub_latents)
                fft_sub_latents = magnitude_latents * torch.exp(1j * mixed_phase)
                sub_latents = torch.fft.irfft2(fft_sub_latents, dim=(-2, -1), norm='ortho')

                shift_left = torch.rand(1).item() < 0.5
                shift_down = torch.rand(1).item() < 0.5

                d_rate = 2
                mask_first_row = torch.zeros(1, patch_size)
                mask_first_row[:, ::d_rate] = 1
                mask_second_row = torch.roll(mask_first_row, shifts=1, dims=1)
                for d in range(1, d_rate):
                    stacked_rows = torch.concatenate((mask_first_row, mask_second_row), axis=-2) # type: ignore
                den_mask = torch.tile(stacked_rows, (patch_size // stacked_rows.shape[0], 1)).to('cpu')
                den_mask = den_mask[np.newaxis, np.newaxis, ...].to(torch.float16)
                den_mask = torch.roll(den_mask, shifts=(-1 if shift_down else 0, -1 if shift_left else 0), dims=(2, 3))

                uniques = torch.unique(sub_time)
                vmax = uniques[-1]
                time_mask = torch.where(sub_time == vmax, 1, 0).to('cpu')
                if len(uniques) > 1:
                    sub_latents = sub_latents * time_mask + sub_prev_latents * (time_mask == 0)

                # TODO: Verify this is correct
                #       This is supposed to be the denoise loop that performs a single step of guidance.
                sub_time_index = (sigmas == sub_time.max()).nonzero().item()
                sub_time_sigmas = sigmas[sub_time_index:sub_time_index+2]
                guider.sample(noise.generate_noise({'samples': sub_latents}), sub_latents, sampler, sub_time_sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)

                smoothed_time_mask = self.create_gradient_border(time_mask, gradient_width=10)
                full_replace_mask = smoothed_time_mask == 1
                no_replace_mask = smoothed_time_mask == 0
                gradient_mask = (smoothed_time_mask > 0) & (smoothed_time_mask < 1)

                if ii < len(guid_latents) - 1:
                    sub_latents = sub_latents * (1 - den_mask) + sub_guid_latents_ahead * den_mask if ii < slider else sub_latents

                latents[..., i-p:i+p, j-p:j+p] = sub_latents * full_replace_mask + \
                        latents[..., i-p:i+p, j-p:j+p] * no_replace_mask + \
                        (sub_latents + latents[..., i-p:i+p, j-p:j+p]) / 2 * gradient_mask

                if times.float().mean() > (sigmas.min().item()):
                    next_timestep_index = (sigmas == sub_time.max()).nonzero(as_tuple=True)[0][-1]
                    next_timestep = sigmas[next_timestep_index + 1].item()
                    times[..., i-p:i+p, j-p:j+p] = torch.where(sub_time == sub_time.max(), torch.ones_like(sub_time).to(sub_time.device) * next_timestep, sub_time)
                else:
                    times[..., i-p:i+p, j-p:j+p] = torch.where(sub_time == sub_time.max(),
                                                            torch.ones_like(sub_time).to(sub_time.device) * 0,
                                                            sub_time)

                if torch.all(times == times.max()):
                    prev_latents = latents.clone()

                if times.float().mean() == 0:
                    break

        samples = latents
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised,)

NODE_CLASS_MAPPINGS = {
    "Pixelsmith": Pixelsmith
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixelsmith": "Pixelsmith"
}
