# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="846pN2uHGJLx"
# # Image2Image Pipeline for Stable Diffusion using 🧨 Diffusers 
#
# This notebook shows how to create a custom `diffusers` pipeline for  text-guided image-to-image generation with Stable Diffusion model using  🤗 Hugging Face [🧨 Diffusers library](https://github.com/huggingface/diffusers). 
#
# For a general introduction to the Stable Diffusion model please refer to this [colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb).
#
#

# %% id="n9AmMcAGASDq" colab={"base_uri": "https://localhost:8080/"} outputId="a118f8c0-8267-4a91-e644-ddaef0a22ce3"
# !nvidia-smi
# !pip install diffusers==0.3.0 transformers ftfy
# !pip install -qq "ipywidgets>=7,<8"

# %% [markdown] id="DQ11k9JVIXII"
# You need to accept the model license before downloading or using the weights. In this post we'll use model version `v1-4`, so you'll need to  visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree. 
#
# You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

# %% id="AOe3MmjoAoSU" colab={"base_uri": "https://localhost:8080/", "height": 272, "referenced_widgets": ["d42d75703380494ea733a9210963a5e7", "cc35eb9fd8384293b1651026fdd1f9eb", "1d5606eec39a47bea1155c9c7fe515c7", "af926235511a4ca09284c408b29c1c5d", "a0132b2a4af04168ba3c1ae718d5cc03", "34dbaa46476d474fa797b33b6504142d", "25dd2b6c2be045af8de0961e65121d24", "3e5c6985e3324e2a95f467b047ca32e6", "e6633f3989d04db1b6be3bffc7d4b0c2", "db3b3665066a4cde8d65225423396b13", "ab9c64a8154e4d66ab72160a501af0c5", "ab9a004e821a479e85a58a60073c0e80", "fb0ab1c8f8de4b5aa29ec46427a064a3", "a80a9b614d234e6bb2c93f703e3c0d62"]} outputId="90d6b354-86e6-43ef-e485-289de18b2420"
from huggingface_hub import notebook_login
import inspect
import warnings
from typing import List, Optional, Union
import torch
from torch import autocast
from tqdm.auto import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

notebook_login()

# %% [markdown] id="YL179UtGKweb"
# Load the pipeline

# %% id="Fr2QIEzvCFH2" colab={"base_uri": "https://localhost:8080/"} outputId="de380bc9-7c91-4be9-cef0-52d001966a60"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
img_to_img = pipe.to(device)

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
text_to_img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
text_to_img = text_to_img.to("cuda")  

# %% [markdown] id="tFBvRqfCKzVQ"
# Download an initial image and preprocess it so we can pass it to the pipeline.

# %% colab={"base_uri": "https://localhost:8080/", "height": 529} id="quipipaCC1AP" outputId="b2d02591-c0db-4534-a630-f6add788f1be"
from diffusers import LMSDiscreteScheduler

lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe.scheduler = lms

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["1302890b2aed4f3597641f822748d9b2", "da996d076aed47b7a628a739b49d6b9b", "a0b8c0b066da4388b9d7a53c188db3cd", "6381940f422245bca1d7556a591d5bc9", "6fcc816e998549c4b1022d6eaaad8785", "96e7ac2741494f0394d99332281c3d30", "19ffabec3fe0447da075680b89d5aca1", "ca6fcdd3bdc54bd6a1d1578e421fcf8f", "14fdd2e6d7bf4d8888a881829e17b13a", "1465cb7856854eca9733884af794aa84", "48b9429809404c64910552f442053501"]} id="q7ixTMhfD6Ux" outputId="d9b2e9f1-c3cf-46c7-8a8a-4c4361065c6a"
# %% id="OenMItYozRA6" outputId="5b97c348-68e4-4476-9b3f-af26c18056a5" colab={"base_uri": "https://localhost:8080/", "height": 529}
import requests
from io import BytesIO
from PIL import Image

url = "https://i.imgur.com/XCoD24f.jpg"

response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
init_img

# %% id="UoJ8MIlFzrFv" outputId="c466cbdb-73c4-4744-c2d8-6c2f525fdb99" colab={"base_uri": "https://localhost:8080/", "height": 561, "referenced_widgets": ["3feffd6077e5407787b5a4e9ceb1b944", "27b14a707826457db8601713661ac545", "4a952d29dcb347a6b94588cec4e418ce", "cae75d12056d4f7fab5ed9ea37b24236", "00e94e4272db4a8292e9595a8586c4a9", "70b002348ac94f31bfa9433ee5f679ac", "4e0301d3c449471ea87d701a7d3a90e9", "a02f340d606b4881a9f244558f3b56bf", "d6595fc967c848adbbd0a9c2518fd554", "802313f24adf4f7087da815ba6ef6efe", "ae88d0c5b69b4ee5bc7b31bf60898c04"]}
generator = torch.Generator(device=device).manual_seed(1024)
prompt = "tall and very large magnificient multicolored flower with multicolored shaft, ice pack, two magical multicored igloos in the background, beautiful large stars in the sky. High quality sketch, trending on art station, magical, fairy tale"
with autocast("cuda"):
    image = img_to_img(prompt=prompt, init_image=init_img, strength=0.40, guidance_scale=20, generator=generator).images[0]
image

# %% colab={"base_uri": "https://localhost:8080/", "height": 235} id="WaAW4sSdV7vZ" outputId="fee7d789-c0d6-43f2-98ab-962ec3ca5fc5"
history = []
prompts_and_params = []
defaults =  {"seed": 1026, "num_inference_steps":40, "guidance_scale":8}
prompts_and_params.append({
    "prompt": "photorealistic attractive girl, art by monet, sharp details, the girl is an amazingly cute zombie"})

options = {**defaults, **prompts_and_params[-1]}
history.append(options)
generator = torch.Generator("cuda").manual_seed(1025)
with autocast("cuda"):
  image = text_to_img(**options, generator=generator).images[0]

image

# %% id="q4F6B8yk-d1M"
