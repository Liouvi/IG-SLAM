
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from gaussian_splatting.scene.gaussian_model import GaussianModel
import random
from munch import munchify
from gui import gui_utils
import torch
from tqdm import tqdm
import cv2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj

from utils.slam_utils import get_loss_mapping
from utils.camera_utils import Camera
torch.multiprocessing.set_sharing_strategy('file_system')
from .backend import Backend
from colorama import Fore, Style
from time import time
import torch.nn as nn
import torchvision.transforms.functional as F
from functools import reduce



class Mapper(object):
    """
    Mpper thread.
    """

    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args
        self.slam = slam
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.last_t = -1
        self.warmup = cfg['tracking']['warmup']
        # backend process
        self.verbose = slam.verbose
        self.first_time = True
        self.video = slam.video
        self.net = slam.net
        self.kf_indices = []
        self.kfs = []
        self.reload_map = slam.reload_map
        self.device = "cuda:0"
        self.output = slam.output
        self.last_idx = 0
        # Camera Parameters
        self.iteration_count = 0
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.fovx = focal2fov(self.fx, self.W)
        self.fovy = focal2fov(self.fy, self.H)
        # Gaussian Parameters
        self.first_frame = True

        self.pyr_scaling = 0.8


        self.pyr_depth = 3
        self.gaussians = None
        self.background = None
        self.pipeline_params = None
        self.opt_params = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.q_map2main = None
        self.last_visit = 0
        self.viewpoint_pyrs = {}
        self.cur_viewpoint = None
        self.current_window = []
        self.monocular = (self.cfg["mode"] == "mono")
        self.occ_aware_visibility = {}
        model_params = munchify(cfg["model_params"])
        opt_params = munchify(cfg["opt_params"])
        pipeline_params = munchify(cfg["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.use_spherical_harmonics = self.cfg["Training"]["spherical_harmonics"]
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.cfg)
        self.gaussians.init_lr(6.0)

        self.gaussians.training_setup(opt_params)

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.cameras_extent = 6.0
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.W,
            H=self.H,
        ).transpose(0, 1)

        self.set_hyperparams()

    def create_color_gaussian_pyramid(self,image, levels):
        pyramid = [image]
        current_image = image
        
        for i in range(1, levels):
            width = int(current_image.shape[2] * self.pyr_scaling)
            height = int(current_image.shape[1] * self.pyr_scaling)
            new_size = (height, width)
            current_image= current_image.unsqueeze(0)
            current_image = F.resize(current_image, new_size)
            current_image= current_image.squeeze(0)
            pyramid.append(current_image)        
        return pyramid


    def create_depth_gaussian_pyramid(self,image, levels):
        pyramid = [image]
        current_image = image
        for i in range(1, levels):
            width = int(current_image.shape[1] * self.pyr_scaling)
            height = int(current_image.shape[0] * self.pyr_scaling)
            new_size = (height, width)
            current_image= current_image.unsqueeze(0)
            current_image = F.resize(current_image, new_size)
            current_image= current_image.squeeze(0)
            pyramid.append(current_image)
        return pyramid
    
    def viewpoint_pyr_from_video(self,color,depth,cov,pose,cur_idx,levels):
        viewpoint_pyr = []
        color_pyr = self.create_color_gaussian_pyramid(color,levels)
        depth_pyr = self.create_depth_gaussian_pyramid(depth,levels)
        cov_pyr = self.create_depth_gaussian_pyramid(cov,levels)





        for idx,(color,depth,cov) in enumerate(zip(color_pyr,depth_pyr,cov_pyr)):

            uid = cur_idx*1000 + idx
            fx = self.fx / (pow(1 / self.pyr_scaling,idx))
            fy = self.fy / (pow(1 / self.pyr_scaling,idx))
            cx = self.cx / (pow(1 / self.pyr_scaling,idx))
            cy = self.cy / (pow(1 / self.pyr_scaling,idx))
            image_height = int(self.H / (pow(1 / self.pyr_scaling,idx)))
            image_width = int(self.W / (pow(1 / self.pyr_scaling,idx)))

            FoVx = focal2fov(fx, image_width)
            FoVy = focal2fov(fy, image_height)

            projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            W=image_width,
            H=image_height,
            ).transpose(0, 1)
            projection_matrix = projection_matrix.to(device=self.device)

            viewpoint = Camera(
            uid,
            color,
            depth,
            cov,
            pose,
            projection_matrix,
            fx,
            fy,
            cx,
            cy,
            FoVx,
            FoVy,
            image_height,
            image_width,
            device=self.device,
            )
            viewpoint_pyr.append(viewpoint)

        viewpoint_pyr.reverse()
        return viewpoint_pyr

    def set_hyperparams(self):
        
        self.mapping_itr_num = self.cfg["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.cfg["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.cfg["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.cfg["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.cfg["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.cfg["Training"]["gaussian_reset"]
        self.size_threshold = self.cfg["Training"]["size_threshold"]
        self.window_size = self.cfg["Training"]["window_size"]
    
    def add_next_kf(self, frame_idx,viewpoint,pyr_level, init=False, scale=2.0, depth_map=None):

        self.gaussians.extend_from_pcd_seq(
            viewpoint,pyr_level,self.pyr_scaling, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
    
    def map(self, current_window,pyr_idx,cur_viewpoint, prune=False, iters=1,depth_grads=None):

        if len(current_window) == 0:
            return
        
        viewpoint_stack = [self.viewpoint_pyrs[kf_idx][pyr_idx] for kf_idx in current_window]
        random_viewpoint_stack = []

        current_window_set = set(current_window)
        for cam_idx in self.kfs:
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(self.viewpoint_pyrs[cam_idx][-1])

        for _ in range(iters):
            self.iteration_count += 1
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []


            if self.q_main2vis.empty():
                self.push_to_gui(cur_viewpoint,self.current_window)

            for cam_idx in reversed(current_window):
                viewpoint = self.viewpoint_pyrs[cam_idx][pyr_idx]
                depth_weights = torch.sqrt(torch.pow(viewpoint.cov, -1))
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                #if pyr_idx == self.pyr_depth - 1 and depth_grads is not None:
                #    depth.grad = depth_grads[cam_idx].unsqueeze(0)
                loss_mapping += get_loss_mapping(
                    self.cfg, image, depth, depth_weights, viewpoint, opacity,initialization=False
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                depth_weights = torch.sqrt(torch.pow(viewpoint.cov, -1))
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.cfg, image, depth,depth_weights, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling

            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.cfg["Training"]["window_size"]:
                        prune_mode = self.cfg["Training"]["prune_mode"]
                        prune_coviz = 0
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[-1]
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True


                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0:
                    
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

        return gaussian_split
                
    def color_refinement(self):
        Log("Starting color refinement")
        cur_t = self.video.counter.value
        for idx in range(cur_t):
            frame_items = self.video.get_mapping_item(idx, self.device)
            image, depth, depth_cov, c2w = frame_items
            viewpoint_pyr = self.viewpoint_pyr_from_video(image,depth,depth_cov,c2w.matrix().inverse(),idx,levels=self.pyr_depth)
            for i in range(len(viewpoint_pyr)):
                viewpoint_pyr[i].T = c2w.matrix().inverse()
            self.viewpoint_pyrs[idx] = viewpoint_pyr

        iteration_total = self.cfg["Training"]["post_process_iters"]
        for iteration in tqdm(range(1, iteration_total + 1)):
            
            viewpoint_idx_stack = [kf_idx for kf_idx in range(cur_t)]
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )


            
            viewpoint_cam = self.viewpoint_pyrs[viewpoint_cam_idx][-1]

            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )

            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            depth_weights = torch.sqrt(torch.pow(viewpoint_cam.cov, -1))
            loss = get_loss_mapping(self.cfg, image, depth, depth_weights, viewpoint_cam, opacity,initialization=False)

            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.gaussians.max_steps - 2000 + iteration)
            #if (iteration + 1) % 20 == 0:
            #    self.push_to_gui(self.cur_viewpoint,self.kfs)
                

        Log("Map refinement done")

    def info(self, msg):
        print(Fore.GREEN)
        print(msg)
        print(Style.RESET_ALL)

    def push_to_gui(self, viewpoint,window):
        
        current_window_dict = {}
        current_window_dict[window[0]] = window[1:]
        keyframes = [self.viewpoint_pyrs[kf_idx][-1] for kf_idx in self.kfs]
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                current_frame=viewpoint,
                keyframes=keyframes,
                kf_window=current_window_dict,
                gtcolor=viewpoint.original_image,
                gtdepth=viewpoint.depth.cpu().numpy(),
                cov = viewpoint.cov.cpu().numpy()
            )
        )

    def downsample(self,viewpoint,scale = 8):

        viewpoint.fx = viewpoint.fx / scale
        viewpoint.fy = viewpoint.fy / scale
        viewpoint.cx = viewpoint.cx / scale
        viewpoint.cy = viewpoint.cy / scale
        viewpoint.image_height = viewpoint.image_height // scale
        viewpoint.image_width = viewpoint.image_width // scale

        viewpoint.depth = cv2.resize(viewpoint.depth, (viewpoint.image_width, viewpoint.image_height), interpolation=cv2.INTER_LINEAR)
        viewpoint.cov = cv2.resize(viewpoint.cov, (viewpoint.image_width, viewpoint.image_height), interpolation=cv2.INTER_LINEAR)
        viewpoint.original_image =  viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
        viewpoint.original_image = cv2.resize(viewpoint.original_image, (viewpoint.image_width, viewpoint.image_height), interpolation=cv2.INTER_AREA)
        viewpoint.original_image = torch.from_numpy(viewpoint.original_image).permute(2, 0, 1).to(self.device)

        viewpoint.FoVx = focal2fov(viewpoint.fx, viewpoint.image_width)
        viewpoint.FoVy = focal2fov(viewpoint.fy, viewpoint.image_height)
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=viewpoint.fx,
            fy=viewpoint.fy,
            cx=viewpoint.cx,
            cy=viewpoint.cy,
            W=viewpoint.image_width,
            H=viewpoint.image_height,
            ).transpose(0, 1)
        
        viewpoint.projection_matrix = projection_matrix.to(device=self.device)

        return viewpoint

    def add_to_window(self,idx):
        if idx not in self.current_window:
            self.current_window.append(idx)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

    def add_to_kfs(self,idx):
        if idx not in self.kfs:
            self.kfs.append(idx)

    def init(self,cur_idx):

        for idx in range(0,cur_idx+1):

            self.current_window.append(idx)
            frame_items = self.video.get_mapping_item(idx, self.device)
            image, depth, depth_cov, c2w = frame_items
            viewpoint_pyr = self.viewpoint_pyr_from_video(image,depth,depth_cov,c2w.matrix().inverse(),idx,levels=self.pyr_depth)
            for i in range(len(viewpoint_pyr)):
                viewpoint_pyr[i].T = c2w.matrix().inverse()
            self.cur_viewpoint = viewpoint_pyr[-1]
            self.viewpoint_pyrs[idx] = viewpoint_pyr


        for idx in self.current_window:

            frame_items = self.video.get_mapping_item(idx, self.device)
            image, depth, depth_cov, c2w = frame_items
            viewpoint_pyr = self.viewpoint_pyr_from_video(image,depth,depth_cov,c2w.matrix().inverse(),idx,levels=self.pyr_depth)
            for i in range(len(viewpoint_pyr)):
                viewpoint_pyr[i].T = c2w.matrix().inverse()
            self.cur_viewpoint = viewpoint_pyr[-1]

            self.viewpoint_pyrs[idx] = viewpoint_pyr
            
        print("current_window: ",self.current_window)
        for pyr_idx in range(self.pyr_depth):
            pyr_level = len(viewpoint_pyr) - pyr_idx
            for idx in self.current_window:
                self.add_next_kf(
                    cur_idx, self.viewpoint_pyrs[idx][pyr_idx],pyr_level, depth_map=self.viewpoint_pyrs[idx][pyr_idx].depth, init=False
                )
            iter_per_kf = self.mapping_itr_num

            self.map(self.current_window,pyr_idx,self.cur_viewpoint, iters=iter_per_kf) 
        

        self.current_window = self.current_window[-self.window_size:]
        self.first_time = False



    def __call__(self):
        cur_idx = self.video.counter.value - 2

        if cur_idx > self.last_idx and cur_idx > self.warmup - 2:
            self.slam.mapping_now += 1

            timestamp = int(self.video.timestamp[cur_idx].cpu().numpy().item())
            
            if self.first_time:
                self.init(cur_idx)
                self.slam.mapping_now -= 1
                return

            if len(self.current_window) != 0 and cur_idx == self.current_window[-1]:
                self.slam.mapping_now -= 1
                return 
            #self.rigid_transformation()
            self.kf_indices.append(timestamp)
            self.add_to_window(cur_idx)
            self.add_to_kfs(cur_idx)

            

            #if self.iteration_count > self.track_start:
            #    print("track!")
            #    self.track(self.viewpoint_pyrs[cur_idx])

            for idx in self.current_window:
                frame_items = self.video.get_mapping_item(idx, self.device)
                image, depth, depth_cov, c2w = frame_items
                viewpoint_pyr = self.viewpoint_pyr_from_video(image,depth,depth_cov,c2w.matrix().inverse(),idx,levels=self.pyr_depth)
                for i in range(len(viewpoint_pyr)):
                    viewpoint_pyr[i].T = c2w.matrix().inverse()
                self.cur_viewpoint = viewpoint_pyr[-1]
                self.viewpoint_pyrs[idx] = viewpoint_pyr

            for idx in self.kfs:
                start_time =time()
                frame_items = self.video.get_mapping_item(idx, self.device)
                image, depth, depth_cov, c2w = frame_items

                for i in range(self.pyr_depth):
                    self.viewpoint_pyrs[idx][i].T = c2w.matrix().inverse()

            for pyr_idx in range(self.pyr_depth):
                pyr_level = len(viewpoint_pyr) - pyr_idx

                self.add_next_kf(
                    cur_idx, self.viewpoint_pyrs[cur_idx][pyr_idx],pyr_level, depth_map=self.viewpoint_pyrs[cur_idx][pyr_idx].depth, init=False
                )

                print("current_window: ",self.current_window)



                iter_per_kf = self.mapping_itr_num

                self.map(self.current_window,pyr_idx,self.cur_viewpoint, iters=iter_per_kf,depth_grads=None) 


            self.map(self.current_window,pyr_idx,self.cur_viewpoint, iters=1,prune=True)
            self.slam.mapping_now.data -= 1
        self.last_idx = cur_idx
