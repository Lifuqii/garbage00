import os
from typing import List, Optional, Union, Any, Dict, Tuple, Callable # Added Tuple

import torch
import torch.nn as nn # <--- 新增导入 nn
import torch.nn.functional as F # <--- 新增导入 F
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.models.controlnet import ControlNetModel
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms
from .style_encoder import Style_Aware_Encoder # Assuming this structure
from .tools import pre_processing # Assuming this structure

from .utils import is_torch2_available

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor, # This will be the original one
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor, # Original
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor, # This should now be our modified IPAttnProcessor2_0
    )
else:
    from .attention_processor import AttnProcessor # Original
    from .attention_processor import CNAttnProcessor # Original
    from .attention_processor import IPAttnProcessor # This should now be our modified IPAttnProcessor

from .resampler import Resampler # Assuming this structure


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

# --- 新增修改: 增强型投影模型 ---
class EnhancedImageProjModel(torch.nn.Module):
    """Projection Model with enhanced MLP for StyleShot"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=2, hidden_dim_ratio=2):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        
        mlp_hidden_dim = int(clip_embeddings_dim * hidden_dim_ratio)
        output_dim = self.clip_extra_context_tokens * cross_attention_dim

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2), # Another layer
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim // 2, output_dim)
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        print(f"Initialized EnhancedImageProjModel with clip_embeddings_dim={clip_embeddings_dim}, mlp_hidden_dim={mlp_hidden_dim}, output_dim={output_dim}")


    def forward(self, image_embeds): # input: (batch, clip_embeddings_dim)
        projected_embeds = self.proj(image_embeds) # (batch, clip_extra_context_tokens * cross_attention_dim)
        
        # Reshape to (batch, num_tokens, cross_attention_dim)
        clip_extra_context_tokens = projected_embeds.reshape(
            image_embeds.shape[0], # Use batch size from input embeds
            self.clip_extra_context_tokens, 
            self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
# --- 结束修改 ---


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel( # Original IPAdapter uses this
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim_ = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim # Renamed to avoid conflict
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim_ is None: # Use the renamed variable
                # For self-attention, use the original AttnProcessor
                if is_torch2_available():
                    from .attention_processor import AttnProcessor2_0 as OriginalAttnProcessor
                    attn_procs[name] = OriginalAttnProcessor()
                else:
                    from .attention_processor import AttnProcessor as OriginalAttnProcessor
                    attn_procs[name] = OriginalAttnProcessor()

            else:
                # For cross-attention, use our modified IPAttnProcessor (which handles PyTorch 2.0 internally)
                attn_procs[name] = IPAttnProcessor( # This will be the correct one due to imports at top
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim_, # Use the renamed variable
                    scale=1.0, # This base scale can be modulated by fusion_alpha
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16) # Ensure IPAttnProcessor handles dtype correctly if params are float32

        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            # CNAttnProcessor is not modified in this round, use original based on torch version
            CurrentCNAttnProcessor = CNAttnProcessor 
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet_ in self.pipe.controlnet.nets: # Renamed
                    controlnet_.set_attn_processor(CurrentCNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CurrentCNAttnProcessor(num_tokens=self.num_tokens))


    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        
        # Filter out non-IPAttnProcessor modules if any were accidentally collected
        ip_adapter_weights = {}
        attn_processor_state_dict = self.pipe.unet.attn_processors
        
        # This part needs to be robust if new parameters (fusion_alpha, ip_temperature) are not in old checkpoints
        # Option 1: Load with strict=False for ip_layers if new params are added
        # Option 2: Ensure checkpoints are saved with these new params
        # For now, let's assume they might not be there and handle it gracefully if possible,
        # or rely on the user to retrain/save new checkpoints.
        # The `load_state_dict` for nn.ModuleList will try to match based on order.
        
        # Collect IPAttnProcessor parameters specifically
        ip_layers_params = []
        for name, proc in attn_processor_state_dict.items():
            if isinstance(proc, IPAttnProcessor): # Check for our modified IPAttnProcessor
                ip_layers_params.extend(proc.parameters())
            # For IPAttnProcessor2_0, it's the same class name due to conditional import
            elif is_torch2_available() and isinstance(proc, globals().get('IPAttnProcessor2_0', type(None))):
                 if proc.__class__.__name__ == 'IPAttnProcessor2_0': # Being more specific
                    ip_layers_params.extend(proc.parameters())


        # Reconstruct a temporary ModuleList of only IPAttnProcs to load weights
        # This is a bit tricky because state_dict["ip_adapter"] has keys like "0.to_k_ip.weight"
        # which map to the *order* of IPAttnProcessors in the original setup.
        
        # We assume state_dict["ip_adapter"] correctly corresponds to only the IPAttnProcessor layers
        # and their original parameters (to_k_ip, to_v_ip).
        # New parameters (fusion_alpha, ip_temperature) will be initialized and not loaded
        # from old checkpoints unless the checkpoint is saved *after* these params were added.

        ip_attn_modules = nn.ModuleList([
            proc for proc in self.pipe.unet.attn_processors.values() 
            if isinstance(proc, IPAttnProcessor) or \
               (is_torch2_available() and proc.__class__.__name__ == 'IPAttnProcessor2_0' and isinstance(proc, globals().get('IPAttnProcessor2_0')))
        ])
        
        if len(ip_attn_modules) > 0:
            # Load existing weights, ignoring missing ones (like new params if loading old ckpt)
            # And also ignoring unexpected ones (if ckpt has more than current model for some reason)
            # `strict=False` is important here if the checkpoint doesn't have fusion_alpha / ip_temperature
            ip_attn_modules.load_state_dict(state_dict["ip_adapter"], strict=False) 
            print(f"IP Adapter weights loaded for {len(ip_attn_modules)} IPAttnProcessor modules. Strict loading: False (to accommodate new params).")
        else:
            print("No IPAttnProcessor modules found to load IP Adapter weights.")


    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds_ = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds # Renamed
        else:
            clip_image_embeds_ = clip_image_embeds.to(self.device, dtype=torch.float16) # Renamed
        image_prompt_embeds = self.image_proj_model(clip_image_embeds_)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds_))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            # Check for both IPAttnProcessor and our modified IPAttnProcessor2_0
            is_ip_attn_proc = isinstance(attn_processor, IPAttnProcessor)
            if is_torch2_available() and not is_ip_attn_proc: # if not the base one, check the 2.0 one by name
                is_ip_attn_proc = attn_processor.__class__.__name__ == 'IPAttnProcessor2_0' and isinstance(attn_processor, globals().get('IPAttnProcessor2_0'))

            if is_ip_attn_proc:
                attn_processor.scale = scale # The original scale, fusion_alpha will modulate this

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""
    # ... (IPAdapterXL content remains largely the same, ensure init_proj and get_image_embeds are consistent if modified above)
    # Make sure set_ip_adapter is called if overridden, or that it inherits correctly.
    # For this example, I'm assuming IPAdapterXL would also benefit from the modified IPAttnProcessor if used.

    def generate(
        self,
        pil_image, # Note: IPAdapterXL in original code uses pil_image not pil_image=None
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image): # pil_image is now a required arg based on original
            num_prompts = 1
            pil_image_list = [pil_image]
        elif isinstance(pil_image, List):
            num_prompts = len(pil_image)
            pil_image_list = pil_image
        else:
            raise TypeError("`pil_image` must be a PIL Image or a list of PIL Images.")


        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        # Pass the list to get_image_embeds
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image=pil_image_list) # Use pil_image kwarg
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1) # num_prompts is already handled by get_image_embeds batching
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)


        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None): # Added clip_image_embeds for consistency
        if pil_image is not None: # Ensure handling for pil_image being None if clip_image_embeds is provided
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds_ = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2] # Renamed
        elif clip_image_embeds is not None:
             clip_image_embeds_ = clip_image_embeds.to(self.device, dtype=torch.float16)
        else:
            raise ValueError("Either pil_image or clip_image_embeds must be provided.")

        image_prompt_embeds = self.image_proj_model(clip_image_embeds_)
        
        # Create zeros for uncond based on the shape of clip_image_embeds_
        uncond_clip_image_embeds_ = torch.zeros_like(clip_image_embeds_)
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds_)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel( # Uses MLPProjModel not Resampler
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size, # Should be image_encoder.config.hidden_size for full features
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter): # Should inherit from IPAdapterPlus or IPAdapterXL for consistency
    """SDXL IPAdapter Plus"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280, # Typical for SDXL UNet
            depth=4,
            dim_head=64,
            heads=20, # num_attention_heads for SDXL cross-attention
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size, # CLIP vision encoder hidden size
            output_dim=self.pipe.unet.config.cross_attention_dim, # SDXL UNet cross attention dim
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image): # Original takes only pil_image
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        # For fine-grained, use hidden_states[-2]
        clip_image_embeds_ = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds_)
        
        uncond_clip_image_embeds_ = torch.zeros_like(clip_image_embeds_) # Correct way to get uncond
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds_)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image, # Required argument
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        # This generate method is identical to IPAdapterXL's generate.
        # It can be inherited if IPAdapterPlusXL inherits from IPAdapterXL.
        # For clarity, repeating it here but noting it's similar to IPAdapterXL.
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
            pil_image_list = [pil_image]
        elif isinstance(pil_image, List):
            num_prompts = len(pil_image)
            pil_image_list = pil_image
        else:
            raise TypeError("`pil_image` must be a PIL Image or a list of PIL Images.")

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image_list) # Pass list
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        return images


def StyleProcessor(style_image, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    # centercrop for style condition
    crop = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ]
    )
    style_image = crop(style_image)
    high_style_patch, middle_style_patch, low_style_patch = pre_processing(style_image.convert("RGB"), transform)
    # shuffling
    high_style_patch, middle_style_patch, low_style_patch = (high_style_patch[torch.randperm(high_style_patch.shape[0])],
                                                             middle_style_patch[torch.randperm(middle_style_patch.shape[0])],
                                                             low_style_patch[torch.randperm(low_style_patch.shape[0])])
    return (high_style_patch.to(device, dtype=torch.float32), middle_style_patch.to(device, dtype=torch.float32), low_style_patch.to(device, dtype=torch.float32))


class StyleShot(torch.nn.Module):
    """StyleShot generation"""
    def __init__(self, device, pipe, ip_ckpt, style_aware_encoder_ckpt, transformer_patch):
        super().__init__()
        self.num_tokens = 6 # Total tokens from 3 patches (2 per patch)
        self.device = device
        self.pipe = pipe

        # Ensure the correct IPAttnProcessor (modified one) is set up
        self.set_ip_adapter(device) # This will now use the modified IPAttnProcessor
        self.ip_ckpt = ip_ckpt

        self.style_aware_encoder = Style_Aware_Encoder(CLIPVisionModelWithProjection.from_pretrained(transformer_patch)).to(self.device, dtype=torch.float32)
        self.style_aware_encoder.load_state_dict(torch.load(style_aware_encoder_ckpt))

        self.style_image_proj_modules = self.init_proj() # This will use EnhancedImageProjModel

        self.load_ip_adapter()
        self.pipe = self.pipe.to(self.device, dtype=torch.float32) # Ensure pipe is on correct device and dtype

    def init_proj(self):
        # --- 修改: 使用 EnhancedImageProjModel ---
        style_image_proj_modules = torch.nn.ModuleList([
                            EnhancedImageProjModel( # <--- 使用新的模型
                                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                                clip_embeddings_dim=self.style_aware_encoder.projection_dim,
                                clip_extra_context_tokens=2, # StyleShot specific: 2 tokens per patch
                                hidden_dim_ratio=2 # Example ratio, can be tuned
                            ),
                            EnhancedImageProjModel( # <--- 使用新的模型
                                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                                clip_embeddings_dim=self.style_aware_encoder.projection_dim,
                                clip_extra_context_tokens=2,
                                hidden_dim_ratio=2
                            ),
                            EnhancedImageProjModel( # <--- 使用新的模型
                                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                                clip_embeddings_dim=self.style_aware_encoder.projection_dim,
                                clip_extra_context_tokens=2,
                                hidden_dim_ratio=2
                            )])
        # --- 结束修改 ---
        return style_image_proj_modules.to(self.device, dtype=torch.float32)


    def load_ip_adapter(self):
        sd = torch.load(self.ip_ckpt, map_location="cpu")
        style_image_proj_sd = {}
        ip_sd = {} # This is for the UNet's IPAttnProcessor parameters
        controlnet_sd = {} # For ControlNet parameters if used

        # Separate keys based on prefixes
        for k, v in sd.items(): # Iterate through items
            if k.startswith("style_image_proj_modules."):
                style_image_proj_sd[k.replace("style_image_proj_modules.", "")] = v
            elif k.startswith("adapter_modules."): # This seems to be the key for UNet IPAttnProcessor params in StyleShot
                ip_sd[k.replace("adapter_modules.", "")] = v
            elif k.startswith("controlnet."):
                controlnet_sd[k.replace("controlnet.", "")] = v
            # elif k.startswith("unet"): # Usually UNet weights are separate
            #     pass 

        # Load state dict for style_image_proj_modules (our EnhancedImageProjModel)
        # Use strict=False if checkpoints might be from the older ImageProjModel
        # or if EnhancedImageProjModel has a different structure than what's saved.
        # For newly trained models with EnhancedImageProjModel, strict=True is fine.
        print(f"Loading style_image_proj_modules. Found {len(style_image_proj_sd)} keys in checkpoint.")
        self.style_image_proj_modules.load_state_dict(style_image_proj_sd, strict=False)


        # Load state dict for IPAttnProcessor modules in UNet
        # These are parameters like to_k_ip, to_v_ip, and potentially new fusion_alpha, ip_temperature
        
        # Collect IPAttnProcessor modules from UNet
        ip_attn_unet_modules = nn.ModuleList([
            proc for proc in self.pipe.unet.attn_processors.values()
            if isinstance(proc, IPAttnProcessor) or \
               (is_torch2_available() and proc.__class__.__name__ == 'IPAttnProcessor2_0' and isinstance(proc, globals().get('IPAttnProcessor2_0')))
        ])

        if len(ip_attn_unet_modules) > 0 and len(ip_sd) > 0:
            print(f"Loading adapter_modules (IPAttnProcessor weights for UNet). Found {len(ip_sd)} keys in checkpoint for {len(ip_attn_unet_modules)} UNet IPAttn modules.")
            # strict=False is crucial if loading a checkpoint saved *before* fusion_alpha/ip_temperature were added,
            # or if the number of IPAttnProcessor layers changed.
            ip_attn_unet_modules.load_state_dict(ip_sd, strict=False)
        elif len(ip_sd) == 0 :
             print("No 'adapter_modules.' keys found in checkpoint for UNet IPAttnProcessor weights.")
        else:
            print("No IPAttnProcessor modules found in UNet to load 'adapter_modules.' weights.")


        if hasattr(self.pipe, "controlnet") and isinstance(self.pipe, StyleContentStableDiffusionControlNetPipeline):
            if len(controlnet_sd) > 0:
                print(f"Loading controlnet weights. Found {len(controlnet_sd)} keys.")
                self.pipe.controlnet.load_state_dict(controlnet_sd, strict=True) # Or strict=False if structure might vary
            else:
                print("ControlNet pipeline detected, but no 'controlnet.' keys found in checkpoint.")


    def set_ip_adapter(self, device): # StyleShot's own set_ip_adapter
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim_ = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim_ is None:
                # Use original AttnProcessor for self-attention
                if is_torch2_available():
                    from .attention_processor import AttnProcessor2_0 as OriginalAttnProcessor
                    attn_procs[name] = OriginalAttnProcessor()
                else:
                    from .attention_processor import AttnProcessor as OriginalAttnProcessor
                    attn_procs[name] = OriginalAttnProcessor()
            else:
                # Use our modified IPAttnProcessor (which handles PyTorch 2.0 internally via import)
                attn_procs[name] = IPAttnProcessor( # This refers to the potentially modified one from attention_processor.py
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim_,
                    scale=1.0, # Base scale, will be modulated by fusion_alpha
                    num_tokens=self.num_tokens, # StyleShot uses 6 tokens (3 patches * 2 tokens/patch)
                ).to(device, dtype=torch.float16) # Ensure correct dtype, float32 for new params if rest is float16

        unet.set_attn_processor(attn_procs)

        # ControlNet attention processor setup (if ControlNet is part of the pipe)
        # This uses the original CNAttnProcessor, not modified here.
        if hasattr(self.pipe, "controlnet") and not isinstance(self.pipe, StyleContentStableDiffusionControlNetPipeline):
            CurrentCNAttnProcessor = CNAttnProcessor # From top-level import
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet_ in self.pipe.controlnet.nets:
                    controlnet_.set_attn_processor(CurrentCNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CurrentCNAttnProcessor(num_tokens=self.num_tokens))


    @torch.inference_mode()
    def get_image_embeds(self, style_image=None):
        style_patches = StyleProcessor(style_image, self.device) # Returns tuple of (H, M, L) patches
        # style_aware_encoder expects a tuple/list of patch tensors for its forward method if it processes them individually
        # Or if it expects a stacked tensor, adjust StyleProcessor or the input to style_aware_encoder
        # Assuming style_aware_encoder's forward can take the tuple from StyleProcessor:
        style_embeds_per_patch_level = self.style_aware_encoder(style_patches) # Output: (batch, num_patch_levels=3, projection_dim)
        style_embeds_per_patch_level = style_embeds_per_patch_level.to(self.device, dtype=torch.float32) # Ensure dtype

        style_ip_tokens_list = []
        uncond_style_ip_tokens_list = []

        # Iterate through each patch level (High, Middle, Low)
        for idx in range(style_embeds_per_patch_level.shape[1]): # Should be 3 levels
            current_level_embeds = style_embeds_per_patch_level[:, idx, :] # (batch, projection_dim)
            
            # Pass to the corresponding EnhancedImageProjModel
            # self.style_image_proj_modules is a ModuleList of 3 EnhancedImageProjModel instances
            projected_tokens = self.style_image_proj_modules[idx](current_level_embeds) # (batch, 2, cross_attn_dim)
            style_ip_tokens_list.append(projected_tokens)

            uncond_embeds = torch.zeros_like(current_level_embeds)
            uncond_projected_tokens = self.style_image_proj_modules[idx](uncond_embeds)
            uncond_style_ip_tokens_list.append(uncond_projected_tokens)
            
        # Concatenate tokens from all 3 patch levels along the sequence dimension (dim=1)
        # Each projected_tokens is (batch, 2, cross_attn_dim)
        # Result should be (batch, 6, cross_attn_dim)
        style_ip_tokens = torch.cat(style_ip_tokens_list, dim=1)
        uncond_style_ip_tokens = torch.cat(uncond_style_ip_tokens_list, dim=1)
        
        return style_ip_tokens, uncond_style_ip_tokens

    def set_scale(self, scale): # StyleShot's own set_scale
        for attn_processor in self.pipe.unet.attn_processors.values():
            is_ip_attn_proc = isinstance(attn_processor, IPAttnProcessor)
            if is_torch2_available() and not is_ip_attn_proc:
                 is_ip_attn_proc = attn_processor.__class__.__name__ == 'IPAttnProcessor2_0' and isinstance(attn_processor, globals().get('IPAttnProcessor2_0'))
            
            if is_ip_attn_proc:
                attn_processor.scale = scale # This is the base scale

    def samples(self, image_prompt_embeds, uncond_image_prompt_embeds, num_samples, device, prompt, negative_prompt,
                seed, guidance_scale, num_inference_steps, content_image, **kwargs, ):
        bs_embed, seq_len, _ = image_prompt_embeds.shape # image_prompt_embeds is style_ip_tokens
        # Repeat for num_samples (e.g. if generating multiple variations for the same style input)
        image_prompt_embeds_rpt = image_prompt_embeds.repeat_interleave(num_samples, dim=0)
        uncond_image_prompt_embeds_rpt = uncond_image_prompt_embeds.repeat_interleave(num_samples, dim=0)


        with torch.inference_mode():
            # Encode text prompts
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt, # prompt here is expected to be a list of strings, already repeated if needed by caller
                device=device,
                num_images_per_prompt=1, # Text prompts are usually 1-to-1 with batch items from style
                                         # If num_samples > 1, the repetition is on style embeddings
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # prompt_embeds_ shape: (batch_size_text_prompts * 2 for CFG, seq_len_text, dim)
            # image_prompt_embeds_rpt shape: (batch_size_style_input * num_samples, seq_len_style, dim)

            # Ensure batch sizes align or broadcast correctly.
            # If prompt is a single string and num_samples > 1, encode_prompt gives (2, seq, dim)
            # and image_prompt_embeds_rpt is (num_samples, seq_style, dim).
            # The concatenation expects batch dim to be the same for CFG.

            # If a single style image generates multiple samples with the *same* text prompt:
            if prompt_embeds_.shape[0] == 2 and image_prompt_embeds_rpt.shape[0] == num_samples: # Common case for single prompt, N samples
                prompt_embeds_expanded = prompt_embeds_.repeat_interleave(num_samples, dim=0) # Now (2*num_samples, seq_text, dim)
                negative_prompt_embeds_expanded = negative_prompt_embeds_.repeat_interleave(num_samples, dim=0)
            elif prompt_embeds_.shape[0] // 2 == image_prompt_embeds_rpt.shape[0]: # Prompts match samples (e.g. batch of style images)
                 prompt_embeds_expanded = prompt_embeds_
                 negative_prompt_embeds_expanded = negative_prompt_embeds_
            else: # Fallback or error
                print(f"Warning: Batch size mismatch between text embeds ({prompt_embeds_.shape[0]//2}) and style embeds ({image_prompt_embeds_rpt.shape[0]}). Using direct cat, may error.")
                prompt_embeds_expanded = prompt_embeds_
                negative_prompt_embeds_expanded = negative_prompt_embeds_


            # Concatenate text and style embeddings
            # prompt_embeds = torch.cat([prompt_embeds_expanded, image_prompt_embeds_rpt], dim=1) # This was for original IP-Adapter where style is text-conditioned
            # For StyleShot, the IPAttnProcessor expects encoder_hidden_states to be [text_embeddings, style_embeddings]
            # So, the prompt_embeds to the pipe should be the concatenated ones.
            
            # The UNet's IPAttnProcessor will split encoder_hidden_states into text and IP parts.
            # So, we need to provide the combined embeddings to the pipe's prompt_embeds argument.
            
            # Correct CFG setup for concatenated embeddings:
            # [negative_text_embeds, negative_style_embeds]
            # [positive_text_embeds, positive_style_embeds]
            
            # encode_prompt already does CFG for text.
            # prompt_embeds_ has [neg_text, pos_text]
            # We need to combine with [uncond_style, cond_style]
            
            # neg_text_embeds = negative_prompt_embeds_expanded # This already comes from encode_prompt's negative part if CFG is True
            # pos_text_embeds = prompt_embeds_expanded # This already comes from encode_prompt's positive part if CFG is True

            # This assumes prompt_embeds_ and negative_prompt_embeds_ are [neg_text_cfg, text_cfg] from encode_prompt
            # And image_prompt_embeds_rpt is [style_cond1, style_cond2, ...]
            # And uncond_image_prompt_embeds_rpt is [style_uncond1, style_uncond2, ...]
            
            # If encode_prompt returns (pos_embeds, neg_embeds) when CFG=True:
            # Let's assume self.pipe.encode_prompt returns (prompt_embeds, negative_prompt_embeds)
            # where prompt_embeds = actual text embeds, negative_prompt_embeds = uncond text embeds.

            # Final prompt_embeds for the pipeline call:
            # Positive part: [positive_text_embeds, positive_style_embeds]
            # Negative part: [negative_text_embeds, negative_style_embeds]
            
            # Let's re-verify how encode_prompt and pipeline expect these.
            # Standard diffusers pipeline:
            # prompt_embeds is [uncond_text, cond_text] if do_classifier_free_guidance.
            # We need to append our IP embeddings to both parts.
            
            # Assuming prompt_embeds_ from encode_prompt is already [neg_text_batch, pos_text_batch]
            # And we have image_prompt_embeds_rpt (positive style) and uncond_image_prompt_embeds_rpt (negative style)

            # Split text embeds if they are already CFG-doubled by encode_prompt
            if prompt_embeds_.shape[0] == 2 * image_prompt_embeds_rpt.shape[0] : # CFG for text, matching style batch
                neg_text_embeds, pos_text_embeds = torch.chunk(prompt_embeds_, 2, dim=0)
            else: # Assume prompt_embeds_ is just pos_text_embeds, and negative_prompt_embeds_ is neg_text_embeds
                  # This happens if encode_prompt takes num_images_per_prompt=1 for text, and we handle CFG batching manually
                  # For StyleShot, prompts are often single.
                pos_text_embeds = prompt_embeds_ # Shape: (batch_style*num_samples_text, seq_text, dim)
                neg_text_embeds = negative_prompt_embeds_ # Shape: (batch_style*num_samples_text, seq_text, dim)

            # Ensure batch sizes are compatible for concatenation.
            # Typically, for N samples from one style, text embeds are repeated N times.
            # image_prompt_embeds_rpt is (N, style_seq, dim)
            # pos_text_embeds should be (N, text_seq, dim)
            if pos_text_embeds.shape[0] != image_prompt_embeds_rpt.shape[0]:
                # This case might occur if prompt was a single string for multiple style samples.
                # Example: 1 style image, 4 samples, 1 text prompt.
                # pos_text_embeds from encode_prompt might be (1, text_seq, dim)
                # image_prompt_embeds_rpt is (4, style_seq, dim)
                # We need to repeat text embeds.
                if pos_text_embeds.shape[0] == 1 and image_prompt_embeds_rpt.shape[0] > 1:
                    pos_text_embeds = pos_text_embeds.repeat(image_prompt_embeds_rpt.shape[0], 1, 1)
                    neg_text_embeds = neg_text_embeds.repeat(image_prompt_embeds_rpt.shape[0], 1, 1)
                else:
                    # Fallback or error for unhandled batch size mismatches
                    print(f"Warning: Text ({pos_text_embeds.shape[0]}) and style ({image_prompt_embeds_rpt.shape[0]}) batch size mismatch in StyleShot.samples. Check prompt handling.")


            # Positive embeddings for the pipe: [text_positive, style_positive]
            final_pos_prompt_embeds = torch.cat([pos_text_embeds, image_prompt_embeds_rpt], dim=1)
            # Negative embeddings for the pipe: [text_negative, style_negative]
            final_neg_prompt_embeds = torch.cat([neg_text_embeds, uncond_image_prompt_embeds_rpt], dim=1)


        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        if content_image is None:
            # The `prompt_embeds` argument to the pipe should contain the concatenated [text, style] embeddings.
            # The IPAttnProcessor inside the UNet will split these based on self.num_tokens.
            images = self.pipe(
                prompt_embeds=final_pos_prompt_embeds, # Positive part of CFG
                negative_prompt_embeds=final_neg_prompt_embeds, # Negative part of CFG
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                # ControlNet specific args are passed via **kwargs if pipe is ControlNet pipe
                **kwargs, 
            ).images
        else: # For StyleContentStableDiffusionControlNetPipeline
            images = self.pipe(
                prompt_embeds=final_pos_prompt_embeds, # Positive part for UNet
                negative_prompt_embeds=final_neg_prompt_embeds, # Negative part for UNet
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                image=content_image, # ControlNet condition image
                # The StyleContentStableDiffusionControlNetPipeline takes 'style_embeddings' and 'negative_style_embeddings'
                # These are *only* the style parts, which it passes to ControlNet's cross-attention.
                # The UNet still gets the combined [text, style] from prompt_embeds.
                # So, this part of the original StyleShot's `samples` needs careful review
                # with how StyleContentStableDiffusionControlNetPipeline actually uses these arguments.
                # For now, assuming the main prompt_embeds handles the UNet part.
                # If ControlNet also needs IP-Adapter style features, its IPAttnProcessor would handle it.
                # The provided StyleContentStableDiffusionControlNetPipeline seems to use 'encoder_hidden_states=style_embeddings_input' for ControlNet.
                # This means ControlNet is conditioned *only* on style embeddings, not text.
                # The UNet is conditioned on [text, style_from_IPAttnProcessor_split].
                # This seems like a mismatch or a very specific setup.

                # Let's assume `style_embeddings` and `negative_style_embeddings` are for the ControlNet part.
                # The UNet gets its combined embeddings from `prompt_embeds` and `negative_prompt_embeds` as constructed above.
                style_embeddings=image_prompt_embeds_rpt, # Only style part for ControlNet
                negative_style_embeddings=uncond_image_prompt_embeds_rpt, # Only style part for ControlNet
                **kwargs,
            ).images
        return images

    def generate(
            self,
            style_image=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=1, # num_samples per style_image and per text prompt
            seed=42,
            guidance_scale=7.5,
            num_inference_steps=50,
            content_image=None,
            **kwargs,
    ):
        self.set_scale(scale) # Sets the base scale for IPAttnProcessor

        # num_prompts refers to the number of text prompts if `prompt` is a list.
        # If style_image is a list, this needs to be handled. Assume single style_image for now.
        
        if prompt is None:
            prompt_list = ["best quality, high quality"]
        elif isinstance(prompt, str):
            prompt_list = [prompt]
        elif isinstance(prompt, List):
            prompt_list = prompt
        else:
            raise TypeError("Prompt must be a string or a list of strings.")

        if negative_prompt is None:
            negative_prompt_list = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt_list)
        elif isinstance(negative_prompt, str):
            negative_prompt_list = [negative_prompt] * len(prompt_list)
        elif isinstance(negative_prompt, List) and len(negative_prompt) == len(prompt_list):
            negative_prompt_list = negative_prompt
        else:
            raise ValueError("Negative prompt must be a string or a list of strings matching the prompt list length.")

        style_ip_tokens, uncond_style_ip_tokens = self.get_image_embeds(style_image)
        # style_ip_tokens shape: (1, 6, cross_attn_dim) if style_image is single

        all_generated_images = []
        for p_idx, p_text in enumerate(prompt_list):
            current_negative_prompt = negative_prompt_list[p_idx]
            # Pass the single text prompt and its negative to the samples method.
            # num_samples here means how many images to generate for *this specific text prompt*
            # using the given style_image.
            images_for_prompt = self.samples(
                image_prompt_embeds=style_ip_tokens, 
                uncond_image_prompt_embeds=uncond_style_ip_tokens, 
                num_samples=num_samples, # This is num_images_per_text_prompt_for_this_style
                device=self.device, 
                prompt=p_text, # Single text prompt string
                negative_prompt=current_negative_prompt, # Single negative text prompt string
                seed=seed + p_idx if seed is not None else None, # Vary seed per prompt or keep same
                guidance_scale=guidance_scale, 
                num_inference_steps=num_inference_steps, 
                content_image=content_image, 
                **kwargs, 
            )
            all_generated_images.extend(images_for_prompt) # Collect all images
        
        # The original returns a list of lists if multiple prompts. Let's stick to that for now.
        # Re-group if necessary, or flatten. The original was List[images] where images itself could be a list.
        # Current self.samples returns a flat list of images for that call.
        # If prompt_list had P prompts, and num_samples=N, self.samples will be called P times,
        # each time generating N images. total P*N images.
        # To match original structure [[img1_p1, img2_p1], [img1_p2, img2_p2]]
        # we need to group them.
        
        # Simple way: return a flat list of all images generated.
        # If len(prompt_list) > 1 or num_samples > 1, this is a list of PIL Images.
        # If len(prompt_list) == 1 and num_samples == 1, it's also a list (of 1 PIL Image).
        # The original StyleShot's `generate` returned `generate_images.append(images)`, where `images` was the list from `self.samples`.
        # So it was a list of lists.
        
        # Reconstruct list of lists:
        grouped_images = []
        idx = 0
        for _ in prompt_list:
            grouped_images.append(all_generated_images[idx : idx + num_samples])
            idx += num_samples
        return grouped_images


# StyleContentStableDiffusionControlNetPipeline definition remains the same
# as it's a pipeline structure, and modifications are within the components it uses.
class StyleContentStableDiffusionControlNetPipeline(StableDiffusionControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None, # ControlNet condition image
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None, # These are [text_cond, ip_cond] from StyleShot.samples
        negative_prompt_embeds: Optional[torch.FloatTensor] = None, # These are [text_uncond, ip_uncond]
        ip_adapter_image: Optional[PipelineImageInput] = None, # Not typically used by StyleShot if style comes via prompt_embeds
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None, # Ditto
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        style_embeddings: Optional[torch.FloatTensor] = None, # Passed from StyleShot.samples for ControlNet part
        negative_style_embeddings: Optional[torch.FloatTensor] = None, # Passed from StyleShot.samples for ControlNet part
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.
        ... (docstring as in original) ...
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        #    IMPORTANT: The original check_inputs might not expect style_embeddings, negative_style_embeddings.
        #    This might need adjustment in a custom pipeline. For now, assuming it passes or is handled.
        self.check_inputs(
            prompt,
            image, # ControlNet condition
            callback_steps, # Deprecated
            negative_prompt,
            prompt_embeds, # Combined text+style for UNet
            negative_prompt_embeds, # Combined text+style for UNet
            ip_adapter_image, # Standard IP-Adapter input (likely None for StyleShot)
            ip_adapter_image_embeds, # Standard IP-Adapter input (likely None for StyleShot)
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None: # prompt_embeds is [text+style_cond]
            batch_size = prompt_embeds.shape[0] # This should be after CFG doubling if handled by StyleShot.samples
                                                # Or, if prompt_embeds here is (batch, seq, dim) before CFG doubling
        else: # Should not happen if check_inputs is robust
            raise ValueError("Either prompt or prompt_embeds must be provided.")


        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions


        # 3. Encode input prompt (IF NOT PROVIDED AS EMBEDS)
        #    For StyleShot, prompt_embeds and negative_prompt_embeds are ALREADY provided with [text,style]
        #    So, the original self.encode_prompt call here should be skipped if embeds are given.
        
        # text_encoder_lora_scale = (
        #     self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        # )
        # prompt_embeds, negative_prompt_embeds = self.encode_prompt( # This would be for pure text prompts
        #     prompt,
        #     device,
        #     num_images_per_prompt, # This is for text prompt duplication
        #     self.do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds, # Pass through if already computed
        #     negative_prompt_embeds=negative_prompt_embeds, # Pass through
        #     lora_scale=text_encoder_lora_scale,
        #     clip_skip=self.clip_skip,
        # )
        # The StyleShot.samples method prepares prompt_embeds and negative_prompt_embeds
        # to already include the concatenated [text, style_token] information.
        # It also handles the CFG batching for these combined embeddings.
        # So, prompt_embeds here is [ (neg_text, neg_style), (pos_text, pos_style) ] if CFG is done outside,
        # or just [ (text, style) ] if CFG is handled below.

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            # Assuming prompt_embeds is (batch, seq, dim) positive, and negative_prompt_embeds is (batch, seq, dim) negative.
            # This is the standard input to self.pipe() from StyleShot.samples()
            unet_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            # For ControlNet, the style_embeddings might also need CFG doubling.
            # StyleShot.samples provides style_embeddings (positive) and negative_style_embeddings (negative)
            if style_embeddings is not None and negative_style_embeddings is not None:
                 controlnet_style_embeds = torch.cat([negative_style_embeddings, style_embeddings])
            else: # Should not happen if StyleShot.samples provides them for CFG
                 controlnet_style_embeds = None
        else:
            unet_prompt_embeds = prompt_embeds
            controlnet_style_embeds = style_embeddings


        # IP-Adapter specific image embeds (likely not used by StyleShot here)
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds( # This is for standard IP-Adapter
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt, # num_images_per_prompt is for text
                self.do_classifier_free_guidance,
            )
            # If using this, it would be added to added_cond_kwargs for the UNet.
            # StyleShot's IP mechanism is via the IPAttnProcessor handling concatenated embeddings.

        # 4. Prepare ControlNet image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image( # image is the ControlNet condition (e.g. Canny edge map)
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt, # num_images_per_prompt for text prompts
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            # ... (multi-controlnet image prep as in original) ...
            images_prepared = []
            if isinstance(image[0], list):image = [list(t) for t in zip(*image)] # Transpose
            for image_ in image:
                image_ = self.prepare_image(image=image_, width=width, height=height, batch_size=batch_size * num_images_per_prompt,
                                           num_images_per_prompt=num_images_per_prompt, device=device, dtype=controlnet.dtype,
                                           do_classifier_free_guidance=self.do_classifier_free_guidance, guess_mode=guess_mode,)
                images_prepared.append(image_)
            image = images_prepared
            height, width = image[0].shape[-2:]
        else:
            assert False, "ControlNet type not recognized for image preparation."


        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, # This batch_size should be the one after CFG if inputs are pre-CFG'd
                                                # If unet_prompt_embeds is already CFG-doubled (2*B, S, D), then latents should be (B, C, H, W)
                                                # The `latent_model_input` below handles the duplication for CFG.
                                                # So, latents here should be (Actual_Batch_Size, C, H, W)
                                                # Actual_Batch_Size = prompt_embeds.shape[0] if prompt_embeds was (B,S,D) positive part.
            num_channels_latents,
            height,
            width,
            unet_prompt_embeds.dtype, # Match dtype of processed embeddings
            device,
            generator,
            latents,
        )

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # added_cond_kwargs for UNet (e.g. for standard IP-Adapter image_embeds, if used)
        # For StyleShot, IP is part of unet_prompt_embeds, handled by IPAttnProcessor.
        added_cond_kwargs = (
            {"image_embeds": image_embeds} # This `image_embeds` is from `prepare_ip_adapter_image_embeds` if used
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and 'image_embeds' in locals()
            else None
        )


        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e) for s, e in zip(control_guidance_start, control_guidance_end)]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # ControlNet inference
                # ControlNet is conditioned on `image` (e.g. Canny) and `controlnet_style_embeds` (from StyleShot)
                if guess_mode and self.do_classifier_free_guidance:
                    control_model_input = latents # Use single batch for guess mode
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    # For guess mode, ControlNet uses only the conditional part of style embeddings
                    cn_style_embeds_input = controlnet_style_embeds.chunk(2)[1] if controlnet_style_embeds is not None else None
                else:
                    control_model_input = latent_model_input # Use doubled batch
                    cn_style_embeds_input = controlnet_style_embeds


                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list): controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                
                # The ControlNet's cross-attention for StyleShot is only on style_embeddings, not text.
                # Its IPAttnProcessor (if it were a StyleShot-aware ControlNet) would need to be set up.
                # The provided CNAttnProcessor only truncates combined embeddings.
                # Here, `encoder_hidden_states` to ControlNet is `cn_style_embeds_input`.
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=cn_style_embeds_input, # Style embeds for ControlNet's x-attn
                    controlnet_cond=image, # This is the Canny/Depth map etc.
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # UNet inference
                # UNet is conditioned on `unet_prompt_embeds` which are [text, style_tokens]
                # Its IPAttnProcessor splits this into text part and style_token part for attention.
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=unet_prompt_embeds, # Combined [text, style_token] embeds
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs, # For other IP-Adapter types, if any
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k_cb in callback_on_step_end_tensor_inputs: callback_kwargs[k_cb] = locals()[k_cb]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    # prompt_embeds might be changed by callback, ensure unet_prompt_embeds is updated if so
                    # For simplicity, not handling callback modifying prompt_embeds here.

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0: # Deprecated callback
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image_out = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0] # Renamed
            image_out, has_nsfw_concept = self.run_safety_checker(image_out, device, unet_prompt_embeds.dtype)
        else:
            image_out = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image_out.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image_out = self.image_processor.postprocess(image_out, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()
        if not return_dict: return (image_out, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image_out, nsfw_content_detected=has_nsfw_concept)
