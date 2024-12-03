from scipy.ndimage import maximum_filter1d
import numpy as np

def diff_x(x, r):
    left = x[r : 2 * r + 1]
    middle = x[2 * r + 1 :] - x[: -2 * r - 1]
    right = x[-1:] - x[-2 * r - 1 : -r - 1]
    return np.concatenate([left, middle, right], axis=0)


def diff_y(x, r):
    left = x[:, r : 2 * r + 1]
    middle = x[:, 2 * r + 1 :] - x[:, : -2 * r - 1]
    right = x[:, -1:] - x[:, -2 * r - 1 : -r - 1]
    return np.concatenate([left, middle, right], axis=1)


def box_filter(x, r):
    return diff_y(diff_x(x.cumsum(axis=0), r).cumsum(axis=1), r)

def decision_map(img1, img2, ks):
    # maximum filter is separable so we perform two 1D filters in sequence
    max1 = maximum_filter1d(maximum_filter1d(img1, axis=0, size=ks, mode="mirror"), axis=1, size=ks, mode="mirror")
    max2 = maximum_filter1d(maximum_filter1d(img2, axis=0, size=ks, mode="mirror"), axis=1, size=ks, mode="mirror")
    return max1 > max2


def majority_filter(map, ks):
    # because the the map is binary, comparing the sum with the area of the kernel is equivalent to majority vote
    radius = int((ks - 1) / 2)
    return box_filter(map.T, radius).T > (ks ** 2) / 2


def get_depth_mask(cov,max_depth_sigma_thresh,ks=32):
    depth_cov = cov
    depth_cov = (depth_cov - depth_cov.min()) / (depth_cov.max() - depth_cov.min())
    th = np.ones_like(depth_cov) * max_depth_sigma_thresh
    
    mask = decision_map(depth_cov, th, ks)
    mask = 1 - majority_filter(
    1 - majority_filter(mask, ks), ks)

    

    return mask.astype(int)