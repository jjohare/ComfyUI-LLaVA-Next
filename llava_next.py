import asyncio
import base64
import os
import re
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

import comfy.utils
import folder_paths

model_fmt = ".safetensors"
model_type = "llava"

defaults = {
    "model_dir": "/mnt/mldata/GenerativeAI/ComfyUI/custom_nodes/ComfyUI-LLaVA-Next/models",
    "temperature": 0.2,
    "max_tokens": 40,
    "prompt_format": "[INST] <image>\nWhat is shown in this image? [/INST]",
}


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_installed_models():
    model_dir = defaults["model_dir"]
    models = [f for f in os.listdir(model_dir) if f.endswith(model_fmt)]
    return [re.sub(rf"{model_fmt}$", "", m) for m in models]


model_cache = {}

async def get_llava_next(model_name):
    if model_name in model_cache:
        return model_cache[model_name]

    model_path = os.path.join(defaults["model_dir"], model_name + model_fmt)

    processor = LlavaNextProcessor.from_pretrained(defaults["model_dir"])
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda:0")

    model_cache[model_name] = (processor, model)
    return processor, model


def encode(image: Image.Image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_image}"
    return image_url


async def get_caption_llava_next(processor, model, image: Image.Image, prompt_format, temp, max_tokens=35):
    prompt = prompt_format.replace("<image>", encode(image))
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length", truncation=True).to("cuda:0")
    output = model.generate(**inputs, max_length=max_tokens, do_sample=True, temperature=temp)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption


def wait_for_async(async_fn, loop=None):
    res = []

    async def run_async():
        r = await async_fn()
        res.append(r)

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    loop.run_until_complete(run_async())

    return res[0]


class LlavaNextCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        all_models = get_installed_models()

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (all_models,),
                "prompt_format": (
                    "STRING",
                    {"default": defaults["prompt_format"], "multiline": True},
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": defaults["max_tokens"],
                        "min": 0,
                        "max": 200,
                        "step": 5,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": defaults["temperature"],
                        "min": 0.0,
                        "max": 1,
                        "step": 0.1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "caption"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def caption(self, image, model, prompt_format, max_tokens, temperature):
        assert isinstance(image, torch.Tensor), f"{image} {type(image)=}"
        assert isinstance(model, str), f"{model} {type(model)=}"
        assert isinstance(prompt_format, str), f"{prompt_format} {type(prompt_format)=}"
        assert isinstance(max_tokens, int), f"{max_tokens} {type(max_tokens)=}"
        assert isinstance(temperature, float), f"{temperature} {type(temperature)=}"

        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0] + 1)

        processor, model = wait_for_async(lambda: get_llava_next(model))
        pbar.update(1)

        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(
                wait_for_async(
                    lambda: get_caption_llava_next(
                        processor,
                        model,
                        image,
                        prompt_format,
                        temperature,
                        max_tokens,
                    )
                )
            )
            pbar.update(1)
        result = "\n".join(tags)
        return {"ui": {"tags": tags}, "result": (result,)}


NODE_CLASS_MAPPINGS = {
    "LlavaNextCaptioner": LlavaNextCaptioner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LlavaNextCaptioner": "LLaVA-NeXT Captioner 🌊",
}
