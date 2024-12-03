import os
import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, Style
from collections import OrderedDict
from lietorch import SE3
from time import gmtime, strftime, time, sleep
import torch.multiprocessing as mp
from gui import gui_utils, slam_gui
from .droid_net import DroidNet
from .frontend import Frontend
from .backend import Backend
from .depth_video import DepthVideo
from .motion_filter import MotionFilter
from .trajectory_filler import PoseTrajectoryFiller
from .mapping import Mapper
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
import wandb
from datetime import datetime
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from utils.multiprocessing_utils import clone_obj
import time

class Tracker(nn.Module):
    def __init__(self, cfg, args, slam):
        super(Tracker, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose
        self.cameras = dict()
        # filter incoming frames so that there is enough motion
        self.frontend_window = cfg['tracking']['frontend']['window']
        filter_thresh = cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)

        # frontend process
        self.frontend = Frontend(self.net, self.video, self.args, self.cfg)

    def forward(self, timestamp, image, depth, intrinsic, gt_pose=None):
        with torch.no_grad():

            ### check there is enough motion
            self.motion_filter.track(timestamp, image, depth, intrinsic, gt_pose=gt_pose)

            # local bundle adjustment
            self.frontend()

class BundleAdjustment(nn.Module):
    def __init__(self, cfg, args, slam):
        super(BundleAdjustment, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.last_t = -1
        self.ba_counter = -1

        # backend process
        self.backend = Backend(self.net, self.video, self.args, self.cfg)

    def info(self, msg):
        print(Fore.GREEN)
        print(msg)
        print(Style.RESET_ALL)

    def forward(self):
        cur_t = self.video.counter.value
        t = cur_t
        t_start = 0
        if cur_t > self.frontend_window and cur_t - self.last_t > 9:
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Full BA'
            msg = f'\n\n {now} : [{t_start}, {t}]; Current Keyframe is {cur_t}, last is {self.last_t}.'

            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False)
            self.info(msg+'\n')

            self.last_t = cur_t
class SLAM:
    def __init__(self, args, cfg):
        super(SLAM, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.verbose = cfg['verbose']
        self.mode = cfg['mode']
        self.output = cfg['data']['output']
        self.start_idx = cfg["data"]["start_idx"]
        self.kf_indices = []
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(f'{self.output}/logs/', exist_ok=True)
        self.cameras = dict()
        self.update_cam(cfg)
        self.net = DroidNet()
        self.load_pretrained(cfg['tracking']['pretrained'])
        self.post_processing = cfg['data']['post_processing']
        self.single_process = cfg["Dataset"]["single_process"]
        self.net.to(self.device).eval()
        self.net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()
        self.tracking_finished = torch.zeros((1)).int()
        self.tracking_finished.share_memory_()
        self.optimizing_finished = torch.zeros((1)).int()
        self.optimizing_finished.share_memory_()
        self.mapping_now = torch.zeros((1)).int()
        self.mapping_now.share_memory_()
        self.GlobalBA = False

        self.hang_on = torch.zeros((1)).int()
        self.hang_on.share_memory_()

        self.reload_map = torch.zeros((1)).int()
        self.reload_map.share_memory_()


        # store images, depth, poses, intrinsics (shared between process)
        self.video = DepthVideo(cfg, args)

        self.tracker = Tracker(cfg, args, self)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)
        self.ba = BundleAdjustment(cfg, args, self)
        self.mapper = Mapper(cfg,args,self)

        q_main2vis = mp.Queue()
        q_vis2main = mp.Queue()

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        model_params = munchify(cfg["model_params"])
        opt_params = munchify(cfg["opt_params"])
        pipeline_params = munchify(cfg["pipeline_params"])
        self.use_spherical_harmonics = self.cfg["Training"]["spherical_harmonics"]
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )


        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.mapper.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )



        self.mapper.q_main2vis = q_main2vis
        self.mapper.q_vis2main = q_vis2main

        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        run = wandb.init(
            project="IG-SLAM",
            name=f"{current_datetime}",
            config=cfg,
            mode=None
        )

    def update_cam(self, cfg):
        """
        Update the camera intrinsics according to the pre-processing config,
        such as resize or edge crop
        """
        # resize the input images to crop_size(variable name used in lietorch)
        H, W = cfg['cam']['H'], cfg['cam']['W']
        fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
        cx, cy = cfg['cam']['cx'], cfg['cam']['cy']

        h_edge, w_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
        H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']

        self.fx = fx * (W_out + w_edge * 2) / W
        self.fy = fy * (H_out + h_edge * 2) / H
        self.cx = cx * (W_out + w_edge * 2) / W
        self.cy = cy * (H_out + h_edge * 2) / H
        self.H, self.W = H_out, W_out

        self.cx = self.cx - w_edge
        self.cy = self.cy - h_edge


                
                
                
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

        self.fovx = focal2fov(self.fx, self.W)
        self.fovy = focal2fov(self.fy, self.H)

    def load_pretrained(self, pretrained):
        
        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(pretrained).items()
        ])

        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]

        self.net.load_state_dict(state_dict)

    def tracking(self,rank,stream,queue):
        print('Tracking Triggered!')
        
        self.all_trigered += 1
        while(self.all_trigered < self.num_running_thread):
            pass
        start_time = time.time()
        idx = self.start_idx
        while (self.tracking_finished < 1):
                
            if not self.mapping_now < 1 and self.single_process:
                sleep(0.01)
                continue


            #if self.init_start == True and self.init_success==True:
            if(idx < len(stream)):
                timestamp, image, depth, intrinsic, gt_pose = stream[idx] 

                viewpoint = Camera(
                        timestamp,
                        image,
                        depth,
                        None,
                        gt_pose.inverse(),
                        self.projection_matrix,
                        self.fx,
                        self.fy,
                        self.cx,
                        self.cy,
                        self.fovx,
                        self.fovy,
                        self.H,
                        self.W,
                        device=self.device,
                    )

                self.cameras[timestamp] = viewpoint

                if self.mode != 'rgbd':
                    depth = None
                
                self.tracker(timestamp, image, depth, intrinsic, gt_pose)
                idx += 1

            else:
                print('Tracking Done!')
                
                self.tracking_finished += 1

        msg = ["cameras",self.cameras]
        queue.put(msg)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"The function took {elapsed_time:.4f} seconds to run.")
        print(f"fps: ", timestamp / elapsed_time)
        
    def terminate(self,rank,stream=None):

        os.makedirs(f'{self.output}/checkpoints/', exist_ok=True)
        torch.save({
            'tracking_net': self.net.state_dict(),
            'keyframe_timestamps': self.video.timestamp,
        }, f'{self.output}/checkpoints/go.ckpt')


        print("Calculating psnr,ssim,lpips")
        camera_trajectory = self.traj_filler(stream)

        for idx in range(self.start_idx,len(self.cameras)+self.start_idx):
           pose = camera_trajectory[idx].matrix().data.cpu()
           self.cameras[idx].T = pose
        
        rendering_result = eval_rendering(
            self.cameras,
            self.gaussians,
            stream,
            self.output,
            self.pipeline_params,
            self.background,
            kf_indices=self.kf_indices,
            start_idx=self.start_idx,
            iteration="before_opt",
        )
        columns = ["tag", "psnr", "ssim", "lpips","l1_depth"]
        metrics_table = wandb.Table(columns=columns)
        metrics_table.add_data(
            self.cfg['data']['input_folder'],
            rendering_result["mean_psnr"],
            rendering_result["mean_ssim"],
            rendering_result["mean_lpips"],
            rendering_result["mean_l1depth"]
        )
        wandb.log({"Metrics": metrics_table})

        do_ate = True
        if do_ate:
            print("Calculating ATE")
            from evo.core.trajectory import PoseTrajectory3D
            import evo.main_ape as main_ape
            from evo.core.metrics import PoseRelation
            from evo.core.trajectory import PosePath3D
            import numpy as np

            print("#"*20 + f" Results for {stream.input_folder} ...")

            timestamps = [i for i in range(len(stream))]

            w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
            camera_trajectory =  w2w * camera_trajectory.inv()
            traj_est = camera_trajectory.data.cpu().numpy()
            estimate_c2w_list = camera_trajectory.matrix().data.cpu()
            np.save(
                f'{self.output}/checkpoints/est_poses.npy',
                  estimate_c2w_list.numpy(), # c2ws
            )

            traj_ref = []
            traj_est_select = []
            if stream.poses is None:  # for eth3d submission
                if stream.image_timestamps is not None:
                    submission_txt = f'{self.output}/submission.txt'
                    with open(submission_txt, 'w') as fp:
                        for tm, pos in zip(stream.image_timestamps, traj_est.tolist()):
                            str = f'{tm:.9f}'
                            for ps in pos:  # timestamp tx ty tz qx qy qz qw
                                str += f' {ps:.14f}'
                            fp.write(str+'\n')
                    print('Poses are save to {}!'.format(submission_txt))

                print("Terminate: no GT poses found!")
                trans_init = None
                gt_c2w_list = None
            else:
                for i in range(len(stream.poses)):
                    val = stream.poses[i].sum()
                    if np.isnan(val) or np.isinf(val):
                        print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
                        continue
                    traj_est_select.append(traj_est[i])
                    traj_ref.append(stream.poses[i])

                traj_est = np.stack(traj_est_select, axis=0)
                gt_c2w_list = torch.from_numpy(np.stack(traj_ref, axis=0))

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:,:3],
                    orientations_quat_wxyz=traj_est[:,3:],
                    timestamps=np.array(timestamps))

                traj_ref =PosePath3D(poses_se3=traj_ref)

                result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                                      pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

                out_path=f'{self.output}/metrics_traj.txt'
                with open(out_path, 'a') as fp:
                    fp.write(result.pretty_str())
        print("Saving Gaussians")
        save_gaussians(self.gaussians,self.output)
        wandb.finish() 

    def mapping(self,rank,queue, dont_run=False):
        print('Dense Mapping Triggered!')
        self.all_trigered += 1
        while(self.tracking_finished < 1):
            self.mapper()
        
        print('Dense Mapping Done!')

        while not self.optimizing_finished:
            sleep(1.0)

        if self.post_processing:
            print('Post Processing!')
            self.mapper.color_refinement()

        msg = ["gaussians", clone_obj(self.mapper.gaussians),self.mapper.kf_indices]
        queue.put(msg)

    def optimizing(self, rank, dont_run=False):
        print('Full Bundle Adjustment Triggered!')
        self.all_trigered += 1
        while(self.tracking_finished < 1):

            self.ba()
        self.ba.last_t = 0
        self.ba()
        self.optimizing_finished += 1
        print('Full Bundle Adjustment Done!')
        




    def run(self, stream):

        queue = mp.Queue()

        tracking_process = mp.Process(target=self.tracking,args=(3,stream,queue))
        mapping_process = mp.Process(target=self.mapping, args=(1,queue,False))
        optimizing_process = mp.Process(target=self.optimizing, args=(4,False))

        if self.cfg["vis"]:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.num_running_thread[0] = 4
        else:
            self.num_running_thread[0] = 3

        tracking_process.start()
        mapping_process.start()
        optimizing_process.start()

        if self.cfg["vis"]:
            gui_process.start()
            self.all_trigered += 1

        
        outputs = []
        while len(outputs) < 2:
            if not queue.empty():
                result = queue.get(timeout=2)  # Adding timeout to avoid indefinite blocking
                outputs.append(result)



        for output in outputs:
    
            if output[0] == "gaussians":
                self.gaussians = output[1]
                self.kf_indices = output[2]
            
            elif output[0] == "cameras":
        
                self.cameras = output[1]

            else:
                raise

        
        tracking_process.join()
        while not self.mapper.q_main2vis.empty():
                self.mapper.q_main2vis.get()
        self.mapper.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
        optimizing_process.join()
        if self.cfg["vis"]:
            gui_process.join()
        mapping_process.join()

        self.terminate(rank=-1, stream=stream)

