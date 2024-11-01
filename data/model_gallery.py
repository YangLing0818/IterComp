from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from data.RPG.RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from data.RPG.mllm import local_llm,GPT4
import torch
import json
import argparse
from PIL import Image
import os


class ModelGallery:
    def __init__(self, model_paths, device="cuda"):
        self.device = device
        self.model_paths = model_paths

    def save_image(self, image: Image, save_path: str):
        """Helper function to save images."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    def generate_sd15(self, prompt, save_path, seed=0):
        self.sd15 = StableDiffusionPipeline.from_pretrained(self.model_paths['sd15'], torch_dtype=torch.float16).to(self.device)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.sd15(prompt=prompt, generator=generator).images[0]
        self.save_image(image, save_path)

    def generate_sd21(self, prompt, save_path, seed=0):
        self.sd21 = StableDiffusionPipeline.from_pretrained(self.model_paths['sd21'], torch_dtype=torch.float16).to(self.device)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.sd21(prompt=prompt, generator=generator).images[0]
        self.save_image(image, save_path)

    def generate_sdxl(self, prompt, save_path, seed=0):
        self.sdxl = DiffusionPipeline.from_pretrained(self.model_paths['sdxl'], torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.sdxl(prompt=prompt, generator=generator).images[0]
        self.save_image(image, save_path)

    def generate_sd3(self, prompt, save_path, seed=0, num_inference_steps=28, guidance_scale=7.0):
        self.sd3 = StableDiffusion3Pipeline.from_pretrained(self.model_paths['sd3'], torch_dtype=torch.float16).to(self.device)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.sd3(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        self.save_image(image, save_path)

    def generate_flux(self, prompt, save_path, seed=0, guidance_scale=3.5, num_inference_steps=50, max_sequence_length=512):
        self.flux = FluxPipeline.from_pretrained(self.model_paths['flux'], torch_dtype=torch.bfloat16).to(self.device)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.flux(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, max_sequence_length=max_sequence_length, generator=generator).images[0]
        self.save_image(image, save_path)
    
    def generate_rpg(self, prompt, save_path, seed=0):
        self.rpg = RegionalDiffusionXLPipeline.from_pretrained(self.model_paths['rpg'], torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
        self.rpg.scheduler = DPMSolverMultistepScheduler.from_config(self.rpg.scheduler.config, use_karras_sigmas=True)
        self.rpg.enable_xformers_memory_efficient_attention()
        ## User input
        para_dict = GPT4(prompt, key='...Put your api-key here...')
        ## MLLM based split generation results
        split_ratio = para_dict['Final split ratio']
        regional_prompt = para_dict['Regional Prompt']
        image = self.rpg(
            prompt = regional_prompt,
            split_ratio = split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
            batch_size = 1, #batch size
            base_ratio = 0.5, # The ratio of the base prompt    
            base_prompt = prompt,       
            num_inference_steps = 20, # sampling step
            height = 1024, 
            width = 1024, 
            seed = seed,# random seed
            guidance_scale = 7.0
        ).images[0]
        self.save_image(image, save_path)
        


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--compositional_metric", type=str, default='attribute_binding', help="we focus on three compositional metrics: Attribut Binding (color, shape, and texture), Spatial Relationship, and Non-spatial Relationship"
    )
    # --compositional_metric should in {'attribute_binding', 'spatial_relationship', 'non_spatial_relationship'}
    args = parser.parse_args()
    model_paths = {
        "sd15": "pt-sk/stable-diffusion-1.5",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd3": "stabilityai/stable-diffusion-3-medium",
        "flux": "black-forest-labs/FLUX.1-dev",
        "rpg": "Lykon/dreamshaper-xl-1-0"
    }

    # Initialize the model gallery
    gallery = ModelGallery(model_paths)

    # Load the prompt data
    with open(f'data/prompt/{args.compositional_metric}_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    for model in model_paths.keys():
        method_name = f"generate_{model}"  
        method = getattr(gallery, method_name, None)  
        output_dir = f"datasets/train/{args.compositional_metric}"
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(data)):
            prompt = data[i]
            method(prompt, f"{output_dir}/{model}_prompt_{i}.png", seed=0)    

