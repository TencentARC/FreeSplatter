import os
import json
import uuid
import time
import rembg
import numpy as np
import trimesh
import torch
import fpsample
import fast_simplification
import matplotlib.pyplot as plt
cmap = plt.get_cmap("hsv")
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
from scipy.spatial.transform import Rotation
from safetensors import safe_open
from huggingface_hub import hf_hub_download

from transformers import AutoModelForImageSegmentation
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from freesplatter.hunyuan.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from freesplatter.utils.mesh_optim import optimize_mesh
from freesplatter.utils.camera_util import *
from freesplatter.utils.recon_util import *
from freesplatter.utils.infer_util import *
from freesplatter.webui.camera_viewer.visualizer import CameraVisualizer


def inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def save_gaussian(latent, gs_vis_path, model, opacity_threshold=None, pad_2dgs_scale=True):
    if latent.ndim == 3:
        latent = latent[0]

    sh_dim = model.sh_dim
    scale_dim = 2 if model.use_2dgs else 3
    xyz, features, opacity, scaling, rotation = latent.split([3, sh_dim, 1, scale_dim, 4], dim=-1)
    features = features.reshape(features.shape[0], sh_dim//3, 3)

    if opacity_threshold is not None:
        index = torch.nonzero(opacity.sigmoid() > opacity_threshold)[:, 0]
        xyz = xyz[index]
        features = features[index]
        opacity = opacity[index]
        scaling = scaling[index]
        rotation = rotation[index]
    
    # transform gaussians from reference view to world view
    cam2world = create_camera_to_world(torch.tensor([0, -2, 0]), camera_system='opencv').to(latent)
    R, T = cam2world[:3, :3], cam2world[:3, 3].reshape(1, 3)
    xyz = xyz @ R.T + T
    rotation = rotation.detach().cpu().numpy()
    rotation = Rotation.from_quat(rotation[:, [1, 2, 3, 0]]).as_matrix()
    rotation = R.detach().cpu().numpy() @ rotation
    rotation = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
    rotation = torch.from_numpy(rotation).to(latent)
    
    # pad 2DGS with an additional z-scale for visualization
    if scaling.shape[-1] == 2 and pad_2dgs_scale:
        z_scaling = inv_sigmoid(torch.ones_like(scaling[:, :1]) * 0.001)
        scaling = torch.cat([scaling, z_scaling], dim=-1)
    pc_vis = model.gs_renderer.gaussian_model.set_data(
        xyz.float(), features.float(), scaling.float(), rotation.float(), opacity.float())
    pc_vis.save_ply_vis(gs_vis_path)


class FreeSplatterRunner:
    def __init__(self, device):
        self.device = device

        # background remover
        self.rembg = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True,
            cache_dir='ckpts/',
        ).to(device)
        self.rembg.eval()

        # diffusion models
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", 
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
            cache_dir="ckpts/",
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.zero123plus_v11 = pipeline.to(device)

        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
            cache_dir="ckpts/",
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.zero123plus_v12 = pipeline.to(device)

        pipeline = HunYuan3D_MVD_Std_Pipeline.from_pretrained(
            './ckpts/Hunyuan3D-1/mvd_std',
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.hunyuan3d_mvd_std = pipeline.to(device)

        # freesplatter
        config_file = 'configs/freesplatter-object.yaml'
        ckpt_path = hf_hub_download('TencentARC/FreeSplatter', repo_type='model', filename='freesplatter-object.safetensors', local_dir='./ckpts/FreeSplatter')
        model = instantiate_from_config(OmegaConf.load(config_file).model)
        state_dict = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict, strict=True)
        self.freesplatter = model.eval().to(device)

        config_file = 'configs/freesplatter-object-2dgs.yaml'
        ckpt_path = hf_hub_download('TencentARC/FreeSplatter', repo_type='model', filename='freesplatter-object-2dgs.safetensors', local_dir='./ckpts/FreeSplatter')
        model = instantiate_from_config(OmegaConf.load(config_file).model)
        state_dict = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict, strict=True)
        self.freesplatter_2dgs = model.eval().to(device)

        config_file = 'configs/freesplatter-scene.yaml'
        ckpt_path = hf_hub_download('TencentARC/FreeSplatter', repo_type='model', filename='freesplatter-scene.safetensors', local_dir='./ckpts/FreeSplatter')
        model = instantiate_from_config(OmegaConf.load(config_file).model)
        state_dict = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict, strict=True)
        self.freesplatter_scene = model.eval().to(device)

    @torch.inference_mode()
    def run_segmentation(
        self, 
        image, 
        do_rembg=True,
    ):
        torch.cuda.empty_cache()

        if do_rembg:
            image = remove_background(image, self.rembg)

        return image

    def run_img_to_3d(
        self, 
        image_rgba, 
        model='Zero123++ v1.2', 
        diffusion_steps=30, 
        guidance_scale=4.0,
        seed=42, 
        view_indices=[],
        gs_type='2DGS',
        mesh_reduction=0.5,
        cache_dir=None,
    ):
        torch.cuda.empty_cache()

        self.output_dir = os.path.join(cache_dir, f'output_{uuid.uuid4()}')
        os.makedirs(self.output_dir, exist_ok=True)

        # image-to-multiview
        input_image = resize_foreground(image_rgba, 0.9)
        seed_everything(seed)
        if model == 'Zero123++ v1.1':
            output_image = self.zero123plus_v11(
                input_image, 
                num_inference_steps=diffusion_steps, 
                guidance_scale=guidance_scale,
            ).images[0]
        elif model == 'Zero123++ v1.2':
            output_image = self.zero123plus_v12(
                input_image, 
                num_inference_steps=diffusion_steps, 
                guidance_scale=guidance_scale,
            ).images[0]
        elif model == 'Hunyuan3D Std':
            output_image = self.hunyuan3d_mvd_std(
                input_image, 
                num_inference_steps=diffusion_steps, 
                guidance_scale=guidance_scale, 
                guidance_curve=lambda t:2.0,
            ).images[0]
        else:
            raise ValueError(f'Unknown model: {model}')
        
        # preprocess images
        image, alpha = rgba_to_white_background(input_image)
        image = v2.functional.resize(image, 512, interpolation=3, antialias=True).clamp(0, 1)
        alpha = v2.functional.resize(alpha, 512, interpolation=0, antialias=True).clamp(0, 1)

        output_image_rgba = remove_background(output_image, self.rembg)
        if 'Zero123++' in model:
            images, alphas = rgba_to_white_background(output_image_rgba)
        else:
            _, alphas = rgba_to_white_background(output_image_rgba)
            images = torch.from_numpy(np.asarray(output_image) / 255.0).float()
            images = rearrange(images, 'h w c -> c h w')

        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        alphas = rearrange(alphas, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        if model == 'Hunyuan3D Std':
            images = images[[0, 2, 4, 5, 3, 1]]
            alphas = alphas[[0, 2, 4, 5, 3, 1]]
        images_vis = v2.functional.to_pil_image(rearrange(images, 'nm c h w -> c h (nm w)'))
        images = v2.functional.resize(images, 512, interpolation=3, antialias=True).clamp(0, 1)
        alphas = v2.functional.resize(alphas, 512, interpolation=0, antialias=True).clamp(0, 1)

        images = torch.cat([image.unsqueeze(0), images], dim=0)     # 7 x 3 x 512 x 512
        alphas = torch.cat([alpha.unsqueeze(0), alphas], dim=0)     # 7 x 1 x 512 x 512

        # run reconstruction
        view_indices = [1, 2, 3, 4, 5, 6] if len(view_indices) == 0 else view_indices
        images, alphas = images[view_indices], alphas[view_indices]
        legends = [f'V{i}' if i != 0 else 'Input' for i in view_indices]

        gs_vis_path, video_path, mesh_fine_path, fig = self.run_freesplatter_object(
            images, alphas, legends=legends, gs_type=gs_type, mesh_reduction=mesh_reduction)

        return images_vis, gs_vis_path, video_path, mesh_fine_path, fig
    
    def run_views_to_3d(
        self, 
        image_files, 
        do_rembg=False,
        gs_type='2DGS',
        mesh_reduction=0.5,
        cache_dir=None,
    ):
        torch.cuda.empty_cache()

        self.output_dir = os.path.join(cache_dir, f'output_{uuid.uuid4()}')
        os.makedirs(self.output_dir, exist_ok=True)

        # preprocesss images
        images, alphas = [], []
        for image_file in image_files:
            if isinstance(image_file, tuple):
                image_file = image_file[0]
            image = Image.open(image_file)
            w, h = image.size

            image_rgba = self.run_segmentation(image, do_rembg=do_rembg)
            if image.mode == 'RGBA':
                image, alpha = rgba_to_white_background(image_rgba)
                image = v2.functional.center_crop(image, min(h, w))
                alpha = v2.functional.center_crop(alpha, min(h, w))
            else:
                image_rgba = resize_foreground(image_rgba, 0.9)
                image_rgba.save('test.png')
                image, alpha = rgba_to_white_background(image_rgba)
            
            image = v2.functional.resize(image, 512, interpolation=3, antialias=True).clamp(0, 1)
            alpha = v2.functional.resize(alpha, 512, interpolation=0, antialias=True).clamp(0, 1)

            images.append(image)
            alphas.append(alpha)

        images = torch.stack(images, dim=0)
        alphas = torch.stack(alphas, dim=0)
        images_vis = v2.functional.to_pil_image(rearrange(images, 'n c h w -> c h (n w)'))

        # run reconstruction
        legends = [f'V{i}' for i in range(1, 1+len(images))]

        gs_vis_path, video_path, mesh_fine_path, fig = self.run_freesplatter_object(
            images, alphas, legends=legends, gs_type=gs_type, mesh_reduction=mesh_reduction)

        return images_vis, gs_vis_path, video_path, mesh_fine_path, fig
    
    def run_freesplatter_object(
        self, 
        images, 
        alphas, 
        legends=None, 
        gs_type='2DGS', 
        mesh_reduction=0.5,
    ):
        torch.cuda.empty_cache()
        device = self.device

        freesplatter = self.freesplatter_2dgs if gs_type == '2DGS' else self.freesplatter

        images, alphas = images.to(device), alphas.to(device)
        
        t0 = time.time()
        with torch.inference_mode():
            gaussians = freesplatter.forward_gaussians(images.unsqueeze(0))
        t1 = time.time()

        # estimate camera parameters and visualize
        c2ws_pred, focals_pred = freesplatter.estimate_poses(images, gaussians, masks=alphas, use_first_focal=True, pnp_iter=10)
        fig = self.visualize_cameras_object(images, c2ws_pred, focals_pred, legends=legends)
        t2 = time.time()
        
        # save gaussians
        gs_vis_path = os.path.join(self.output_dir, 'gs_vis.ply')
        save_gaussian(gaussians, gs_vis_path, freesplatter, opacity_threshold=5e-3, pad_2dgs_scale=True)
        print(f'Save gaussian at {gs_vis_path}')

        # render video
        with torch.inference_mode():
            c2ws_video = get_circular_cameras(N=120, elevation=0, radius=2.0, normalize=True).to(device)
            fx = fy = focals_pred.mean() / 512.0
            cx = cy = torch.ones_like(fx) * 0.5
            fxfycxcy_video = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).repeat(c2ws_video.shape[0], 1).to(device)

            video_frames = freesplatter.forward_renderer(
                gaussians,
                c2ws_video.unsqueeze(0),
                fxfycxcy_video.unsqueeze(0),
            )['image'][0].clamp(0, 1)

        video_path = os.path.join(self.output_dir, 'gs.mp4')
        save_video(video_frames, video_path, fps=30)
        print(f'Save video at {video_path}')
        t3 = time.time()

        # extract mesh
        with torch.inference_mode():
            c2ws_fusion = get_fibonacci_cameras(N=120, radius=2.0)
            c2ws_fusion, _ = normalize_cameras(c2ws_fusion, camera_position=torch.tensor([0., -2., 0.]), camera_system='opencv')
            c2ws_fusion = c2ws_fusion.to(device)
            c2ws_fusion_reference = torch.linalg.inv(c2ws_fusion[0:1]) @ c2ws_fusion
            fx = fy = focals_pred.mean() / 512.0
            cx = cy = torch.ones_like(fx) * 0.5
            fov = np.rad2deg(np.arctan(0.5 / fx.item())) * 2
            fxfycxcy_fusion = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).repeat(c2ws_fusion.shape[0], 1).to(device)

            fusion_render_results = freesplatter.forward_renderer(
                gaussians,
                c2ws_fusion_reference.unsqueeze(0),
                fxfycxcy_fusion.unsqueeze(0),
            )
            images_fusion = fusion_render_results['image'][0].clamp(0, 1).permute(0, 2, 3, 1)
            alphas_fusion = fusion_render_results['alpha'][0].permute(0, 2, 3, 1)
            depths_fusion = fusion_render_results['depth'][0].permute(0, 2, 3, 1)

            fusion_images = (images_fusion.detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)
            fusion_depths = depths_fusion.detach().cpu().numpy()
            fusion_alphas = alphas_fusion.detach().cpu().numpy()
            fusion_masks = (fusion_alphas > 1e-2).astype(np.uint8)
            fusion_depths = fusion_depths * fusion_masks - np.ones_like(fusion_depths) * (1 - fusion_masks)

            fusion_c2ws = c2ws_fusion.detach().cpu().numpy()

            mesh_path = os.path.join(self.output_dir, 'mesh.obj')
            rgbd_to_mesh(
                fusion_images, fusion_depths, fusion_c2ws, fov, mesh_path, cam_elev_thr=-90)    # use all angles for tsdf fusion
            print(f'Save mesh at {mesh_path}')
            t4 = time.time()

        # optimize texture
        cam_pos = c2ws_fusion[:, :3, 3].cpu().numpy()
        cam_inds = torch.from_numpy(fpsample.fps_sampling(cam_pos, 16).astype(int)).to(device=device)

        alphas_bake = alphas_fusion[cam_inds]
        images_bake = (images_fusion[cam_inds] - (1 - alphas_bake)) / alphas_bake.clamp(min=1e-6)

        fxfycxcy = fxfycxcy_fusion[cam_inds].clone()
        intrinsics = torch.eye(3).unsqueeze(0).repeat(len(cam_inds), 1, 1).to(fxfycxcy)
        intrinsics[:, 0, 0] = fxfycxcy[:, 0]
        intrinsics[:, 0, 2] = fxfycxcy[:, 2]
        intrinsics[:, 1, 1] = fxfycxcy[:, 1]
        intrinsics[:, 1, 2] = fxfycxcy[:, 3]

        # out_mesh = Mesh.load(str(mesh_path), auto_uv=False, device='cpu')
        out_mesh = trimesh.load(str(mesh_path), process=False)
        out_mesh = optimize_mesh(
            out_mesh, 
            images_bake, 
            alphas_bake.squeeze(-1), 
            c2ws_fusion[cam_inds].inverse(), 
            intrinsics,
            simplify=mesh_reduction,
            verbose=False
        )
        mesh_fine_path = os.path.join(self.output_dir, 'mesh.glb')

        out_mesh.export(mesh_fine_path)
        print(f"Save optimized mesh at {mesh_fine_path}")
        t5 = time.time()

        print(f'Generate Gaussians: {t1-t0:.2f} seconds.')
        print(f'Estimate poses: {t2-t1:.2f} seconds.')
        print(f'Generate video: {t3-t2:.2f} seconds.')
        print(f'Generate mesh: {t4-t3:.2f} seconds.')
        print(f'Optimize mesh: {t5-t4:.2f} seconds.')

        return gs_vis_path, video_path, mesh_fine_path, fig

    def visualize_cameras_object(
        self, 
        images, 
        c2ws, 
        focal_length, 
        legends=None,
    ):
        images = v2.functional.resize(images, 128, interpolation=3, antialias=True).clamp(0, 1)
        images = (images.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)

        cam2world = create_camera_to_world(torch.tensor([0, -2, 0]), camera_system='opencv').to(c2ws)
        transform = cam2world @ torch.linalg.inv(c2ws[0:1])
        c2ws = transform @ c2ws
        c2ws = c2ws.detach().cpu().numpy()
        c2ws[:, :, 1:3] *= -1   # opencv to opengl

        focal_length = focal_length.mean().detach().cpu().numpy()
        fov = np.rad2deg(np.arctan(256.0 / focal_length)) * 2

        colors = [cmap(i / len(images))[:3] for i in range(len(images))]

        legends = [None] * len(images) if legends is None else legends

        viz = CameraVisualizer(c2ws, legends, colors, images=images)
        fig = viz.update_figure(
            3, 
            height=320,
            line_width=5,
            base_radius=1, 
            zoom_scale=1, 
            fov_deg=fov, 
            show_grid=True, 
            show_ticklabels=True, 
            show_background=True, 
            y_up=False,
        )
        return fig
    
    # FreeSplatter-S
    def run_views_to_scene(
        self, 
        image1,
        image2,
        cache_dir=None,
    ):
        torch.cuda.empty_cache()

        self.output_dir = os.path.join(cache_dir, f'output_{uuid.uuid4()}')
        os.makedirs(self.output_dir, exist_ok=True)

        # preprocesss images
        images = []
        for image in [image1, image2]:
            w, h = image.size
            image = torch.from_numpy(np.asarray(image) / 255.0).float()
            image = rearrange(image, 'h w c -> c h w')
            image = v2.functional.center_crop(image, min(h, w))
            image = v2.functional.resize(image, 512, interpolation=3, antialias=True).clamp(0, 1)
            images.append(image)

        images = torch.stack(images, dim=0)
        images_vis = v2.functional.to_pil_image(rearrange(images, 'n c h w -> c h (n w)'))

        # run reconstruction
        legends = [f'V{i}' for i in range(1, 1+len(images))]

        gs_vis_path, video_path, fig = self.run_freesplatter_scene(images, legends=legends)

        return images_vis, gs_vis_path, video_path, fig
    
    def run_freesplatter_scene(
        self, 
        images, 
        legends=None, 
    ):
        torch.cuda.empty_cache()

        freesplatter = self.freesplatter_scene

        device = self.device
        images = images.to(device)
        
        t0 = time.time()
        with torch.inference_mode():
            gaussians = freesplatter.forward_gaussians(images.unsqueeze(0))
        t1 = time.time()

        # estimate camera parameters
        c2ws_pred, focals_pred = freesplatter.estimate_poses(images, gaussians, use_first_focal=True, pnp_iter=10)
        # rescale cameras to make the baseline equal to 1.0
        baseline_pred = (c2ws_pred[:, :3, 3] - c2ws_pred[:1, :3, 3]).norm() + 1e-2
        scale_factor = 1.0 / baseline_pred
        c2ws_pred = c2ws_pred.clone()
        c2ws_pred[:, :3, 3] *= scale_factor
        # visualize cameras
        fig = self.visualize_cameras_scene(images, c2ws_pred, focals_pred, legends=legends)
        t2 = time.time()
        
        # save gaussians
        gs_vis_path = os.path.join(self.output_dir, 'gs_vis.ply')
        save_gaussian(gaussians, gs_vis_path, freesplatter, opacity_threshold=5e-3)
        print(f'Save gaussian at {gs_vis_path}')

        # render video
        with torch.inference_mode():
            c2ws_video = generate_interpolated_path(c2ws_pred.detach().cpu().numpy()[:, :3, :], n_interp=120)
            c2ws_video = torch.cat([
                torch.from_numpy(c2ws_video), 
                torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(c2ws_video.shape[0], 1, 1)
            ], dim=1).to(gaussians)
            fx = fy = focals_pred.mean() / 512.0
            cx = cy = torch.ones_like(fx) * 0.5
            fxfycxcy_video = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).repeat(c2ws_video.shape[0], 1).to(device)

            video_frames = freesplatter.forward_renderer(
                gaussians,
                c2ws_video.unsqueeze(0),
                fxfycxcy_video.unsqueeze(0),
                rescale=scale_factor.reshape(1).to(gaussians)
            )['image'][0].clamp(0, 1)

        video_path = os.path.join(self.output_dir, 'gs.mp4')
        save_video(video_frames, video_path, fps=30)
        print(f'Save video at {video_path}')
        t3 = time.time()

        print(f'Generate Gaussians: {t1-t0:.2f} seconds.')
        print(f'Estimate poses: {t2-t1:.2f} seconds.')
        print(f'Generate video: {t3-t2:.2f} seconds.')

        return gs_vis_path, video_path, fig

    def visualize_cameras_scene(
        self, 
        images, 
        c2ws, 
        focal_length, 
        legends=None,
    ):
        images = v2.functional.resize(images, 128, interpolation=3, antialias=True).clamp(0, 1)
        images = (images.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)

        c2ws = c2ws.detach().cpu().numpy()
        c2ws[:, :, 1:3] *= -1

        focal_length = focal_length.mean().detach().cpu().numpy()
        fov = np.rad2deg(np.arctan(256.0 / focal_length)) * 2

        colors = [cmap(i / len(images))[:3] for i in range(len(images))]

        legends = [None] * len(images) if legends is None else legends

        viz = CameraVisualizer(c2ws, legends, colors, images=images)
        fig = viz.update_figure(
            2, 
            height=320,
            line_width=5,
            base_radius=1, 
            zoom_scale=1, 
            fov_deg=fov, 
            show_grid=True, 
            show_ticklabels=True, 
            show_background=True, 
            y_up=False,
        )
        return fig
