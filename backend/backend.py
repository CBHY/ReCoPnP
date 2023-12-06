import os
import sys
sys.path.append('/home/cvlserver/ssd2tb/hkt/backend/ReCo/')
sys.path.append('/home/cvlserver/ssd2tb/hkt/pnp-diffusers')
import uuid
import torch
from fastapi import FastAPI, File, UploadFile

import argparse, os, sys, glob, re
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPTokenizer, CLIPTextModel

from pydantic import BaseModel
from typing import Optional, Union

class Arguments(BaseModel):
    prompt: Optional[str] = "A cat is eating a fish. <546> <621> <885> <861> white cat <654> <512> <739> <779> Another cat"
    outdir: Optional[str] = "outputs"
    laion400m : bool = False
    plms : bool = True
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    n_iter: int = 4
    H: int = 512
    W: int = 512
    C: int = 4
    f: int = 8
    n_samples: int = 1
    n_rows: int = 0
    scale: float = 2.0
    fixed_code : bool = False
    skip_grid : bool = False
    skip_save : bool = False
        
#     from_file: Optional[str] = None
    from_file:bool = False
    config: str = "ReCo/configs/stable-diffusion/v1-inference-box.yaml"
    ckpt: str = "ReCo/logs/reco_laion_1232.ckpt"
    seed: int = 42
    precision: str = "autocast"
#     embedding_path: Optional[str] = None
    embedding_path: bool = False


app = FastAPI()

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"


@app.post("/image_gen")
async def image_gen(arg: Arguments):
    res_image_path_list = []
    def pre_caption(caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())


    def load_model_from_config(config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
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

        model.cuda()
        model.eval()
        return model


    def main():
        
        if arg.laion400m:
            print("Falling back to LAION 400M model...")
            arg.config = "ReCo/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            arg.ckpt = "ReCo/logs/reco_laion_1232.ckpt"
            arg.outdir = "ReCo/outputs/txt2img-samples-laion400m"

        seed_everything(arg.seed)

        config = OmegaConf.load(f"{arg.config}")

        if arg.config != 'ReCo/configs/stable-diffusion/v1-inference.yaml':
            max_src_length = 1232 if '1232' in arg.ckpt else 616
            config.model.params.cond_stage_config.params['extend_outputlen']=max_src_length
            config.model.params.cond_stage_config.params['max_length']=max_src_length

        model = load_model_from_config(config, f"{arg.ckpt}")
        #model.embedding_manager.load(arg.embedding_path)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if arg.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(arg.outdir, exist_ok=True)
        outpath = arg.outdir

        batch_size = arg.n_samples
        n_rows = arg.n_rows if arg.n_rows > 0 else batch_size
        if not arg.from_file:
            prompt = arg.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {arg.from_file}")
            with open(arg.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))


        ## box processor prepare
        cliptext_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        box_offset = len(cliptext_tokenizer)
        coco_hw = json.load(open('ReCo/dataset/coco_wh.json', 'r'))
        coco_od = json.load(open('ReCo/dataset/coco_allbox_gitcaption.json', 'r')) ## 'tag' 'pad_caption' 'crop_caption'
        num_bins, centercrop = 1000, True
        num_withbox = 0
        for key in coco_od:
            if 'box' in coco_od[key]:
                if len(coco_od[key]['box'])!=0: num_withbox+=1
        ## end of box processor prepare

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if arg.fixed_code:
            start_code = torch.randn([arg.n_samples, arg.C, arg.H // arg.f, arg.W // arg.f], device=device)

        precision_scope = autocast if arg.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(arg.n_iter, desc="Sampling"):
                        seed_everything(arg.seed+n)
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if arg.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            ##
                            token_prompt = []
                            prompt_seq = []
                            for x in prompts[0].split(' <'):
                                prompt_seq += [y for y in x.split('>') if y!='' and y!=' ']

                            tokenized_text = []
                            str_caption = prompt_seq[0]
                            text = [pre_caption(prompt_seq[0].lower(), max_src_length)]
                            text_enc = cliptext_tokenizer(text, truncation=True, max_length=max_src_length, return_length=True,
                                                        return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                            tokenized_text.append(text_enc[0,:])
                            # tokenized_text = []
                            # str_caption = ''

                            for ii in range(1,len(prompt_seq),5):
                                box, caption = [float(x) for x in prompt_seq[ii:ii+4]], prompt_seq[ii+4][1:]

                                box = [float(x)/num_bins for x in box]
                                quant_x0 = int(round((box[0] * (num_bins - 1)))) + box_offset
                                quant_y0 = int(round((box[1] * (num_bins - 1)))) + box_offset
                                quant_x1 = int(round((box[2] * (num_bins - 1)))) + box_offset
                                quant_y1 = int(round((box[3] * (num_bins - 1)))) + box_offset
                                region_coord = torch.tensor([quant_x0,quant_y0,quant_x1,quant_y1]).to(text_enc.device)
                                caption = pre_caption(caption.lower(), max_src_length)
                                region_text = cliptext_tokenizer(caption, truncation=True, max_length=max_src_length, return_length=True,
                                                            return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                                tokenized_text.append(region_coord)
                                tokenized_text.append(region_text[0,:])
                                str_caption += ' <%d> <%d> <%d> <%d> '%(quant_x0-box_offset,quant_y0-box_offset,quant_x1-box_offset,quant_y1-box_offset) + caption
                            tokenized_text = torch.cat(tokenized_text, dim=0)[:max_src_length]
                            pad_tokenized_text = torch.tensor([box_offset-1]*max_src_length).to(text_enc.device)
                            pad_tokenized_text[:len(tokenized_text)] = tokenized_text
                            prompts = pad_tokenized_text.unsqueeze(0)
                            arg.prompt, prompt = str_caption[:200], str_caption[:200]

                            c = model.get_learned_conditioning(prompts)
                            shape = [arg.C, arg.H // arg.f, arg.W // arg.f]
                            samples_ddim, _ = sampler.sample(S=arg.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=arg.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=arg.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=arg.ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not arg.skip_save:
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    # Image.fromarray(x_sample.astype(np.uint8)).save(
                                    #     os.path.join(sample_path, f"{base_count:05}.jpg"))
                                    repeat=0
                                    while os.path.isfile(os.path.join('/home/cvlserver/ssd2tb/hkt/backend/ReCo/'+outpath, "test_%d.png"%(repeat))):
                                        repeat+=1
                                    pred_image = Image.fromarray(x_sample.astype(np.uint8))
                                    res_image_path = os.path.join('/home/cvlserver/ssd2tb/hkt/backend/ReCo/'+outpath, "test_%d.png"%(repeat))
                                    pred_image.save(res_image_path)
                                    res_image_path_list.append(res_image_path)

                                    base_count += 1


                            draw = ImageDraw.Draw(pred_image)
                            for ii in range(1,len(prompt_seq),5):
                                box, caption = [float(x) for x in prompt_seq[ii:ii+4]], prompt_seq[ii+4][1:]
                                box = [float(x)/num_bins for x in box]
                                pred_dim = pred_image.size[0]
                                left, top, right, bottom = int(box[0]*pred_dim), int(box[1]*pred_dim), int(box[2]*pred_dim), int(box[3]*pred_dim)
                                left, top, right, bottom = min(max(0,left),pred_dim), min(max(0,top),pred_dim), min(max(0,right),pred_dim), min(max(0,bottom),pred_dim)
                                if left>=right or top>=bottom:
                                    continue
                                draw.rectangle(((left, top), (right, bottom)), fill=None, outline=(184,134,11), width=int(8*512/500))
                            pred_image.save(os.path.join('/home/cvlserver/ssd2tb/hkt/backend/ReCo/'+outpath, "bbox/test_%d.png"%(repeat)))



                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")
        print(pad_tokenized_text)
        
    main()
    print(res_image_path_list)
    return {'res_image_path':res_image_path_list[0]}


import pnp_diffusers.preprocess as pnp_pre
import pnp_diffusers.pnp as PNP_


class Arguments_pre(BaseModel):
    data_path: str = "/home/cvlserver/ssd2tb/hkt/backend/ReCo/outputs/test_0.png"


@app.post("/style_gen")
async def style_gen_pnp_pre(arg_pre : Arguments_pre):
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=arg_pre.data_path)
    parser.add_argument('--save_dir', type=str, default='latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=999)
    parser.add_argument('--save-steps', type=int, default=1000)
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    opt = parser.parse_args(args=[])
    pnp_pre.run(opt)
    return {}

import yaml

class Arguments_pnp(BaseModel):
    prompt:str = "a photo of a colorful lego horse on the forest"
    image_path:str = "/home/cvlserver/ssd2tb/hkt/backend/ReCo/outputs/test_0.png"
    output_path:str = "/home/cvlserver/ssd2tb/hkt/backend/pnp_diffusers/PNP-results/final_result"
    negative_prompt:str = "ugly, blurry, black, low res, unrealistic"


@app.post("/style_gen2")
async def style_gen_pnp(arg_pnp : Arguments_pnp):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='pnp_diffusers/config_pnp.yaml')
    opt = parser.parse_args(args=[])
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    seed_everything(config["seed"])
    print(config,type(config))
    print(arg_pnp,type(arg_pnp))
    config['prompt'] = arg_pnp.prompt
    config['image_path'] = arg_pnp.image_path
    config['output_path'] = arg_pnp.output_path
    config['negative_prompt'] = arg_pnp.negative_prompt
    pnp = PNP_.PNP(config)
    pnp.run_pnp()
    return {'final_output_path':arg_pnp.output_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
