import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import open3d as o3d
import pickle
import imageio
import sys
import os
import tqdm
import collections
import matplotlib.pylab as plt 
import scipy
import cProfile

# Point cloud from depth (by Konstantin)
def pointcloudify_depth(depth, intrinsics, dist_coeff=[], undistort=True):
    shape = depth.shape[::-1]
    
    if undistort:
        undist_intrinsics, _ = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeff, shape, 1, shape)
        inv_undist_intrinsics = np.linalg.inv(undist_intrinsics)

    else:
        inv_undist_intrinsics = np.linalg.inv(intrinsics)

    if undistort:
        # undist_depthi = cv2.undistort(depthi, intrinsics, dist_coeff, None, undist_intrinsics)
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, dist_coeff, None
                                                  , undist_intrinsics, shape, cv2.CV_32FC1)
        undist_depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)

    # Generate x,y grid for H x W image
    grid_x, grid_y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    grid = np.concatenate([np.expand_dims(grid_x, -1),
                           np.expand_dims(grid_y, -1)], axis=-1)

    grid = np.concatenate([grid, np.ones((shape[1], shape[0], 1))], axis=-1)

    # To normalized image coordinates
    local_grid = inv_undist_intrinsics @ grid.reshape(-1, 3).transpose()  # 3 x H * W

    # Raise by undistorted depth value from image plane to local camera space
    if undistort:
        local_grid = local_grid.transpose() * np.expand_dims(undist_depth.reshape(-1), axis=-1)

    else:
        local_grid = local_grid.transpose() * np.expand_dims(depth.reshape(-1), axis=-1)
        
    return local_grid.astype(np.float32)


def project_pcd_to_depth(pcd, undist_intrinsics, img_size): 
    I = np.zeros(img_size, np.float32)
    h, w = img_size
    points = np.asarray(pcd.points)
    d = points[:, 2] #np.linalg.norm(points, axis=1)
    normalized_points = points / np.expand_dims(points[:, 2], axis=1)
    proj_pcd = np.round(undist_intrinsics @ normalized_points.T).astype(np.int64)[:2].T
    proj_mask = (proj_pcd[:, 0] >= 0) & (proj_pcd[:, 0] < w) & (proj_pcd[:, 1] >= 0) & (proj_pcd[:, 1] < h)
    proj_pcd = proj_pcd[proj_mask, :]
    d = d[proj_mask]
    pcd_image = np.zeros((1080, 1920))
    pcd_image[proj_pcd[:, 1], proj_pcd[:, 0]] = d
    return pcd_image

def smooth_depth(depth):
    MAX_DEPTH_VAL = 1e6
    KERNEL_SIZE = 11
    depth[depth == 0] = MAX_DEPTH_VAL
    smoothed_depth = scipy.ndimage.minimum_filter(depth, KERNEL_SIZE)
    smoothed_depth[smoothed_depth == MAX_DEPTH_VAL] = 0
    return smoothed_depth

def initial_filter(gray_mask, depth):
    mask_arr = np.array(gray_mask).reshape(1920*1080, 1)
    depth_arr = np.array(depth).reshape(1920*1080, 1)
    depth_filtered_arr = np.fromiter((d  if mask_arr[num] > 0 else 0 for num, d in enumerate(depth_arr)), dtype=depth_arr.dtype)
    return depth_filtered_arr.reshape(1080, 1920)

def convert_from_depth(depth_arr, rgb_cnf, shp):
    with open('bandeja_standard.pickle', 'rb') as config:
        config_dict = pickle.load(config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloudify_depth(depth_arr, config_dict['depth']['dist_mtx'],
                                                config_dict['depth']['dist_coef']))
    T = np.load('azure2s10_standard_extrinsics(1).npy')

    # Align point cloud with depth reference frame
    pcd.transform(T)
    # o3d.visualization.draw_geometries([pcd])
    # Project aligned point cloud to rgb
    #?
    aligned_depth = project_pcd_to_depth(pcd, rgb_cnf['undist_intrinsics'], shp)
    return aligned_depth
    

def convert_pcd(depth, rgb_cnf):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloudify_depth(depth, rgb_cnf['undist_intrinsics'], undistort=False)) # H * W X 3
    return pcd

def filter_pcd(pcd):
    cld = np.asarray(pcd.points)
    return [(p) for num, p in enumerate(cld) if p[0] > 0  or p[1] > 0 or p[2] > 0]

def filter_pcd_points(pcd, pcd_mask):
    cld = np.asarray(pcd.points)
    cld_mask = np.asarray(pcd_mask.points)

    return [(p) for num, p in enumerate(cld) if cld_mask[num][2] > 0 and (p[0] > 0  or p[1] > 0 or p[2] > 0)]

def count_labels(filtered, labels):
    uniq = {}
    depth_p = {}
    for pos, a in enumerate(labels):
        if not (a in uniq):
            uniq[a] = 0
            depth_p[a] = filtered[pos][2]
        else:
            uniq[a] +=1
    return (uniq, depth_p)

def filter_small_clusters(uniq, depth_p):
    filtered_depth_p = {}

    for key, dist in depth_p.items():
        if abs(dist) > 0.0001 and uniq[key] > 1000:
            filtered_depth_p[key] = dist
    return filtered_depth_p

def choose_mask_points(labels, filtered, mink):
    ready = []
    for n, x in enumerate(labels): 
        if x == mink:
            ready.append(filtered[n])
    return ready

def back_to_png(ready, rgb_cnf, shp):
    r = o3d.geometry.PointCloud()
    r.points = o3d.utility.Vector3dVector(ready)
    aligned_depth = project_pcd_to_depth(r, rgb_cnf['undist_intrinsics'], shp)
    return aligned_depth

def upd_mask(aligned_depth):
    for num_c, rows in enumerate(aligned_depth):
        for num_r, col in enumerate(rows):
            if (aligned_depth[num_c][num_r] > 0):
                aligned_depth[num_c][num_r] = 255

def main():
    path = sys.argv[1]
    # s = line.strip()
    s = "test8"

    rgb_cnf = np.load('s10_standard_intrinsics(1).npy', allow_pickle=True).item()
    mask = imageio.imread(os.path.join(path, s, "mask.png"))
    depth = imageio.imread(os.path.join(path, s, "depth.png"))
    depth_array = np.load(os.path.join(path, s, "pc.npy"), allow_pickle=True)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    depth_sparce = convert_from_depth(depth_array, rgb_cnf, mask.shape[:2])

    filtered = np.asarray(convert_pcd(depth_sparce * (gray_mask > 0), rgb_cnf).points)

    filtered = filtered[~np.all(filtered == 0, axis=1)]

    clustering = DBSCAN(eps = 0.5)
    # clustering = AgglomerativeClustering(n_clusters =None, distance_threshold = 50)
    labels = clustering.fit_predict(filtered)


    uniq, depth_p = count_labels(filtered, labels)
    print(uniq)

    filtered_depth_p = filter_small_clusters(uniq, depth_p)

    mink = min(filtered_depth_p, key=depth_p.get)

    ready = choose_mask_points(labels, filtered, mink)

    aligned_depth = back_to_png(ready, rgb_cnf, depth.shape[:2]) 
    aligned_depth = smooth_depth(aligned_depth)

    upd_mask(aligned_depth)

    imageio.imwrite(os.path.join(path, s, "res.png"), (aligned_depth).astype(np.uint16))

if __name__ == "__main__":
    main()
    


