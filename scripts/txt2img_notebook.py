# %%
import argparse, os, sys, glob
from dataclasses import dataclass
import cv2
import torch as t
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

MAIN = __name__ == "__main__"

# %%
def get_device():
    if(t.cuda.is_available()):
        return 'cuda'
    elif(t.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = t.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

# %%
if MAIN:
    @dataclass
    class StableDiffusionConfig():
        prompt = "a painting of a virus monster playing guitar"
        outdir = "../outputs/txt2img-samples"
        skip_grid = False
        skip_save = False
        ddim_steps = 100
        plms = True
        laion400m = False
        fixed_code = False
        ddim_eta = 0.0
        n_iter = 1
        height = 512
        weight = 512
        C = 4
        f = 8
        n_samples = 1
        n_rows = 0
        scale = 7.5
        from_file = ""
        config = "../configs/stable-diffusion/v1-inference.yaml"
        ckpt = "../models/ldm/stable-diffusion-v1/model.ckpt"
        seed = 42
        precision = ""

    opt = StableDiffusionConfig()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = t.device(get_device())
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    start_code = None
    if opt.fixed_code:
        start_code = t.randn(
            [opt.n_samples, opt.C, opt.height // opt.f, opt.weight // opt.f], device="cpu"
        ).to(t.device(device))

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if device.type == 'mps':
        print('Using mps backend')
        precision_scope = nullcontext # have to use f32 on mps

# %%
def txt2img(prompt: str) -> "list[Image.Image]":
    assert prompt is not None
    data = [batch_size * [prompt]]
    with t.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                all_samples = list()
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.height // opt.f, opt.weight // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = t.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu()
                    if not opt.skip_save:
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            # base_count += 1
                            all_samples.append(img)
    return all_samples
    # print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
    #       f" \nEnjoy.")

# %%

image_out = txt2img('an astronaut riding on a horse')[0]

# %%
