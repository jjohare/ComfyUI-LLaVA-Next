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
model_type = "llava-next"
system_message = (
    "You are an assistant who describes the content and composition of images. "
    "Describe only what you see in the image, not what you think the image is about. "
    "Be factual and literal. Do not use metaphors or similes. Be concise."
)


defaults = {
    "model": "llava-v1.6-mistral-7b-hf",
    "temperature": 0.2,
    "max_tokens": 40,
    "prompt": "[INST] <image>\nWhat is shown in this image?[/INST]",
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
    if model_type not in folder_paths.folder_names_and_paths:
        models_dir = get_ext_dir("models", mkdir=True)
        folder_paths.add_model_folder_path(model_type, models_dir)

    models = folder_paths.get_filename_list(model_type)
    return [
        re.sub(rf"{model_fmt}$", "", m)
        for m in models
        if m.endswith(model_fmt)
    ]


async def get_llava_next(model):
    assert isinstance(model, str), f"{model} {type(model)=}"
    model_path = folder_paths.get_full_path(model_type, model + model_fmt)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} does not exist")

    start = time.monotonic()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = LlavaNextProcessor.from_pretrained(model)
    model = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(device)

    print(f"LLaVA-NeXT loaded in {time.monotonic() - start:.1f}s")
    return processor, model


def encode(image: Image.Image):
    assert isinstance(image, Image.Image), f"{image} {type(image)}"
    with BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_image}"
    return image_url


async def get_caption(
    processor,
    model,
    image: Image.Image,
    prompt,
    temp,
    max_tokens=35,
):
    assert isinstance(image, Image.Image), f"{image} {type(image)=}"
    assert isinstance(prompt, str), f"{prompt} {type(prompt)=}"
    assert isinstance(temp, float), f"{temp} {type(temp)=}"
    assert isinstance(max_tokens, int), f"{max_tokens} {type(max_tokens)=}"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

    start = time.monotonic()
    output = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp)
    print(f"Response in {time.monotonic() - start:.1f}s")

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


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
    _model = None
    _processor = None

    @classmethod
    def INPUT_TYPES(s):
        all_models = get_installed_models()

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (all_models,),
                "prompt": (
                    "STRING",
                    {"default": defaults["prompt"], "multiline": True},
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

    def caption(self, image, model, prompt, max_tokens, temperature):
        assert isinstance(image, torch.Tensor), f"{image} {type(image)=}"
        assert isinstance(model, str), f"{model} {type(model)=}"
        assert isinstance(prompt, str), f"{prompt} {type(prompt)=}"
        assert isinstance(max_tokens, int), f"{max_tokens} {type(max_tokens)=}"
        assert isinstance(temperature, float), f"{temperature} {type(temperature)=}"

        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        if self._model is None or self._processor is None:
            self._processor, self._model = wait_for_async(lambda: get_llava_next(model))

        pbar = comfy.utils.ProgressBar(tensor.shape[0])

        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(
                wait_for_async(
                    lambda: get_caption(
                        self._processor,
                        self._model,
                        image,
                        prompt,
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
