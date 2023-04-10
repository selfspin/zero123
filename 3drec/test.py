import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
from torchvision import transforms

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig, camera_pose, sample_near_eye

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img
from voxnerf.data import load_blender
from my3d import get_T, depth_smooth_loss

from finetune.data_objaverse import Objaverse, DiffusionLoss
from finetune.distributed_utils import get_rank, get_world_size, is_main_process, reduce_value
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import numpy as np
import time
import argparse
import random
import torch.backends.cudnn
import os
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import pickle
from torch.optim import lr_scheduler
import torch.distributed as dist
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device_glb = torch.device("cuda")


def load_im(im_path):
    from PIL import Image
    from io import BytesIO
    import requests
    from torchvision import transforms
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
    tforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inp = tforms(im).unsqueeze(0)
    return inp * 2 - 1


def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }


class SJC(BaseConf):
    family: str = "sd"
    sd: SD = SD(
        variant="objaverse",
        scale=100.0
    )
    lr: float = 0.05
    n_steps: int = 10000
    vox: VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose: PoseConfig = PoseConfig(rend_hw=32, FoV=49.1, R=2.0)

    emptiness_scale: int = 10
    emptiness_weight: int = 0
    emptiness_step: float = 0.5
    emptiness_multiplier: float = 20.0

    grad_accum: int = 1

    depth_smooth_weight: float = 1e5
    near_view_weight: float = 1e5

    depth_weight: int = 0

    var_red: bool = True

    train_view: bool = True
    scene: str = 'chair'
    index: int = 2

    view_weight: int = 10000
    prefix: str = 'exp'
    nerf_path: str = "data/nerf_wild"

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox)


def sjc_3d(poser, vox, model: ScoreAdapter,
           lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
           depth_weight, var_red, train_view, scene, index, view_weight, prefix, nerf_path,
           depth_smooth_weight, near_view_weight, grad_accum, **kwargs):
    bs = 1
    ts = model.us[30:-10]

    if is_main_process():
        print("Loading training data ...")

    save_root = "/SSD_DISK/datasets/objaverse_animal_render/"
    root = '/EXT_DISK/datasets/objaverse_animal_render/'

    print('Loading BLIP2 model ...')
    text_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    text_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                               torch_dtype=torch.float16)
    text_model.cuda()
    print('BLIP2 loaded!')

    picture_size = 256
    test_transforms = transforms.Compose([
        transforms.Resize([picture_size]),
        transforms.CenterCrop([picture_size, picture_size]),
        transforms.ToTensor()
    ])

    world_size = get_world_size()
    local_rank = get_rank()

    uids = os.listdir(root)
    uids.sort()

    per_len = len(uids) // world_size
    uids = uids[int(per_len * local_rank):int(per_len * (local_rank + 1))]
    if is_main_process():
        uids = tqdm(uids)

    for uid in uids:
        for str in ['img_emb', 'clip_emb', 'vae_emb', 'prompt_emb']:
            os.makedirs(os.path.join(save_root, uid, str), exist_ok=True)

        img_path = os.path.join(root, uid, 'image', '000_img_0001.png')
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = test_transforms(img)

        inputs = text_processor(img, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = text_model.generate(**inputs, max_new_tokens=20)
        generated_text = text_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        prompt_emb = model.prompts_emb_2([generated_text]).detach()
        np.save(os.path.join(save_root, uid, 'prompt_emb', f'{"%03d" % 0}.npy'), prompt_emb.cpu())

        for id in range(42):
            refer_img_path = os.path.join(root, uid, 'image', f'{"%03d" % id}_img_0001.png')
            refer_img = Image.open(refer_img_path)
            refer_img = refer_img.convert('RGB')
            refer_img = test_transforms(refer_img)[None].cuda()

            clip_emb = model.model.get_learned_conditioning(refer_img.float()).tile(1, 1, 1).detach()
            vae_emb = model.model.encode_first_stage(refer_img.float()).mode().detach()
            img_emb = model.encode(refer_img).detach()

            np.save(os.path.join(save_root, uid, 'clip_emb', f'{"%03d" % id}.npy'), clip_emb[0].cpu())
            np.save(os.path.join(save_root, uid, 'vae_emb', f'{"%03d" % id}.npy'), vae_emb[0].cpu())
            np.save(os.path.join(save_root, uid, 'img_emb', f'{"%03d" % id}.npy'), img_emb[0].cpu())


@torch.no_grad()
def evaluate(score_model, vox, poser):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "view_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    weights = rearrange(weights, "N (h w) 1 -> N h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights
    else:
        return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    # y = torch.nn.functional.interpolate(y, 512, mode='bilinear', antialias=True)
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", lambda fn: imwrite(fn, im))
    metric.put_artifact("depth", ".npy", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    cfg = optional_load_config(fname="full_config_objaverse.yml")
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    pbar = tqdm(range(1))

    with EventStorage(model.prompt.replace(' ', '-')), HeartBeat(pbar):
        ckpt_fname = latest_ckpt()
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        with EventStorage(model.prompt.replace(' ', '-') + "_test"):
            evaluate(model, vox, poser)


def latest_ckpt():
    ts, ys = read_stats("./", "ckpt")
    assert len(ys) > 0
    return ys[-1]


def main_worker(local_rank, nprocs):
    torch.cuda.set_device(local_rank)
    seed_everything(0)
    dist.init_process_group(backend="nccl",
                            init_method=f'tcp://127.0.0.1:{15765}',
                            world_size=nprocs,
                            rank=local_rank)
    dispatch(SJC)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    gpus = '0,1,2,3,4,6,7'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    nprocs = len(gpus.split(','))

    mp.spawn(main_worker, nprocs=nprocs, args=(nprocs,))
