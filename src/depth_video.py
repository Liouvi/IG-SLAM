import torch
import lietorch
import droid_backends
from copy import deepcopy
import numpy as np
from torch.multiprocessing import Value
import cv2
from .droid_net import cvx_upsample
from .geom import projective_ops as pops

class DepthVideo:
    def __init__(self, cfg, args):
        self.cfg =cfg
        self.args = args
        self.compute_covs = True
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.mapping = Value('i', 0)
        self.ba_lock = {
            'dense': Value('i', 0),
            'loop': Value('i', 0),
        }
        self.global_ba_lock = Value('i', 0)
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.stereo = (cfg['mode'] == 'stereo')
        device = args.device
        self.device = device
        c = 1 if not self.stereo else 2
        self.scale_factor = 8
        s = self.scale_factor
        buffer = cfg['tracking']['buffer']

        self.dht = self.ht // s 
        self.dwd = self.wd // s

        ### Set initial disparity cov ###
        self.sigma_disp = torch.tensor(0.1, device=self.device)
        self.disp_prior_cov = torch.pow(self.sigma_disp, 2)

        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.float)
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_gt = torch.zeros(buffer, 4, 4, device=device, dtype=torch.float).share_memory_()  # c2w matrix
        self.disps = torch.ones(buffer, ht//s, wd//s, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//s, wd//s, device=device, dtype=torch.float).share_memory_()
        self.depths_gt = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()
        self.depths_cov = torch.ones(buffer,  ht//s, wd//s, dtype=torch.float, device=self.device).share_memory_()
        self.depths_cov_up  = torch.ones(buffer,  ht, wd, dtype=torch.float, device=self.device).share_memory_()

        # Uncertainty sigmas#
        self.optimizing = torch.zeros((1)).int().share_memory_()
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//s, wd//s, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//s, wd//s, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//s, wd//s, dtype=torch.half, device=device).share_memory_()

        ### initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses_gt[:] = torch.eye(4, dtype=torch.float, device=device)

        ### pose compensation from vitural to real
        self.pose_compensate = torch.zeros(1, 7, dtype=torch.float, device=device).share_memory_()
        self.pose_compensate[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.dummy_loss = torch.tensor(0.0, dtype=torch.float, device=device,requires_grad=True).share_memory_()

    def get_lock(self):
        return self.counter.get_lock()

    def get_ba_lock(self, ba_type):
        return self.ba_lock[ba_type].get_lock()

    def get_mapping_lock(self):
        return self.mapping.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1]


        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            self.depths_gt[index] = item[4]
            depth = item[4][..., 3::8, 3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)
            self.disps[index] = self.disps_sens[index].clone()

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.poses_gt[index] = item[9].to(self.poses_gt.device)

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index > 0:
                index = self.counter.value + index
            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
            )

        return item

    def append(self, *item):
         with self.get_lock():
             self.__item_setter(self.counter.value, item)


    ###  dense mapping operations ###
    def get_mapping_item(self, index, device='cuda:0'):
        with self.mapping.get_lock():
            image = self.images[index].clone()  # [h, w, 3]
            est_disp = self.disps_up[index].clone()  # [h, w]  # [h, w]
            est_depth_cov = self.depths_cov_up[index].clone()

            gt_depth = self.depths_gt[index].clone()  # [h, w]
            est_depth = 1.0 / (est_disp + 1e-7)
            
            # origin alignment
            w2c = lietorch.SE3(self.poses[index].clone()) # Tw(droid)_to_c
            c2w = lietorch.SE3(self.pose_compensate[0].clone()) * w2c.inv() 

            depth = est_depth

            depth_cov = est_depth_cov

            return image, depth, depth_cov,c2w
        
    
    def set_item_from_mapping(self, index, pose=None, depth=None):
        with self.get_lock():
            pass

    ### geometric operations ###

    @staticmethod
    def format_indices(ii, jj, device='cuda'):
        """ to device, long, {-1}"""
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(dim=-1), mask) # [b, h, w, 1]
        self.disps_up[ix] = disps_up.squeeze()  # [b, h, w]
        self.disps_up[ix] = cvx_upsample(self.disps[ix].unsqueeze(-1), mask).squeeze()
        self.depths_cov_up[ix] = cvx_upsample(self.depths_cov[ix].unsqueeze(-1), mask).squeeze()

    def normalize(self):
        """ normalize depth and poses """
        with self.get_lock():
            cur_ix = self.counter.value
            s = self.disps[:cur_ix].mean()
            self.disps[:cur_ix] /= s
            self.poses[:cur_ix, :3] *= s  # [tx, ty, tz, qx, qy, qz, qw]
            self.dirty[:cur_ix] = True

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indices(ii, jj, self.device)
        Gs = lietorch.SE3(self.poses[None, ...])

        coords, valid_mask = pops.projective_transform(
            poses=Gs, depths=self.disps[None, ...], intrinsics=self.intrinsics[None, ...],
            ii=ii, jj=jj, jacobian=False, return_depth=False,
        )

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric, where distance = sqrt((u(ii) - u(jj->ii))^2 + (v(ii) - v(jj->ii))^2) """
        return_matrix = False
        N = self.counter.value
        if ii is None:
            return_matrix = True
            ii, jj = torch.meshgrid(
                torch.arange(N),
                torch.arange(N),
                indexing='ij'
            )

        ii, jj = DepthVideo.format_indices(ii, jj)

        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        if bidirectional:
            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta
            )

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], jj, ii, beta
            )

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=0, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, ba_type=None,fill=False):
        """ dense bundle adjustment (DBA) """
        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        lock = self.get_lock() if ba_type is None else self.get_ba_lock(ba_type)
        with lock:
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1
            
            N = t1 - t0
            HW = self.dht * self.dwd

            droid_backends.ba(self.poses, self.disps, self.intrinsics[intrinsic_common_id], self.disps_sens,
                              target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)

            #At that point you have to implement Hessian 
            
            #eta = damping
            if self.compute_covs and not fill and ba_type != 'loop' and ba_type != 'dense':
                w2c = lietorch.SE3(self.poses.clone()).to(self.device)
                c2w = w2c.inv()

                w2c = w2c.vec()
                c2w = c2w.vec()

                identity_transform = lietorch.SE3(torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float)).to(self.device)

                H, v, Q, E, w = droid_backends.reduced_camera_matrix(
                    c2w,
                    w2c, # TODO(remove, unnecessary, given the math)
                    self.disps,
                    self.intrinsics[0],
                    identity_transform.vec(),
                    self.disps_sens,
                    target, 
                    weight, # TODO: we should pass as well previous pose covariances!, so to not lose information when optimizing
                    eta,
                    ii, jj, t0, t1)
                
                #Cholesky decomposition
                
                try:
                    L = torch.linalg.cholesky(torch.as_tensor(H, device=self.device, dtype=torch.float))# from double to float...
                except:
                    L = None
                if L is not None:
                    identity = torch.eye(L.shape[0], device=L.device) # L has shape (PD,PD) 
                    L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
                    if torch.isnan(L_inv).any():
                        print("NANs in L_inv!!")
                        raise
                    # We only care about block diagonals of sigma_g though, here we are calculating everything...
                    sigma_gg = L_inv.transpose(-2,-1) @ L_inv 
                    # TODO: this is the same as optimizeDensely in gtsam....
                    # delta = sigma @ torch.as_tensor(v, device=self.device, dtype=torch.float)

                    # Calculate sigmas
                    # Extract only the block-diagonal of size D from sigma_g
                    P = N
                    D = L.shape[0] // P
                    assert D == 6

                    sigma_gg = sigma_gg.view(P, D, P, D).permute(0,2,1,3) # P x P x D x D
                    sigma_g = torch.diagonal(sigma_gg, dim1=0, dim2=1).permute(2,0,1).view(P, D, D) # P x D x D

                    Ei = E[:P]
                    Ejz = E[P:P+ii.shape[0]]
                    M = Ejz.shape[0]
                    assert M == ii.shape[0]
                    kx, kk = torch.unique(ii, return_inverse=True)
                    K = kx.shape[0] # !!!! Aixo es different de N o rather de P i probablement K = P + fixed_poses, so K>P!

                    # Ejz contains all the psi(D)*z(HW) pairs of products (M in total)
                    # These are populating the off-diagonal of E,
                    # The diagonal is populated by Ei
                    min_ii_jj = min(ii.min(),jj.min())
                    #ic(K)
                    #ic(P)
                    Ej = torch.zeros(K, K, D, HW, device=self.device) # HUGE MEMORY CONSUMPTION, this should be P, K, D, HW
                    # The equation is E[jj[m], ii[m]] = Ejz[m] for m in M, but if we take into account the fixed poses, and indices, we get:
                    Ej[jj - min_ii_jj, ii - min_ii_jj] = Ejz
                    Ej = Ej[t0-min_ii_jj:t1-min_ii_jj].view(P,K,D,HW) # Keep only the keyframes we are optimizing over, and remove the fixed ones, but add all the depth-maps...
                    # The diagonal is populated by Ei
                    Ej[range(P), t0-min_ii_jj:t1-min_ii_jj, :, :] = Ei[range(P), :, :]
                    
                    E_sum = Ej
                    E_sum = E_sum.view(P, K, D, HW)
                    E_sum = E_sum.permute(0,2,1,3).reshape(P*D, K*HW)
                    Q_ = Q.view(K*HW,1)
                    F = torch.matmul(Q_ * E_sum.t(), L_inv) # K*HW x D*P
                    F2 = torch.pow(F, 2)
                    delta_cov = F2.sum(dim=-1) # K*HW
                    z_cov = Q_.squeeze() + delta_cov # K*HW
                    z_cov = z_cov.view(K, self.dht, self.dwd)

                    # Update depths_sigma, clamp?
                    depth_cov = z_cov / self.disps[kx]**4
                    self.depths_cov[kx] = depth_cov

            ###########################