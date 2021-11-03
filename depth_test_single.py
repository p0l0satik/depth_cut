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

if __name__ == "__main__":
    path = sys.argv[1]
    with open(os.path.join(path, "annotation.txt")) as asoc:
        lines = asoc.readlines()
        for line in lines:
            s = line.strip().split(" ")
            mask = imageio.imread(os.path.join(path, s[0]))
            depth = imageio.imread(os.path.join(path, s[1]))
            depth2 = cv2.imread(os.path.join(path, s[1]))

            gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            depth_gray = cv2.cvtColor(depth2, cv2.COLOR_RGB2GRAY)
            only_mask = cv2.bitwise_and(depth_gray.copy(), gray_mask)
            
            depth_dist = {}
            for num_c, col in enumerate(only_mask):
                for num_r, val in enumerate(col):
                    if not (val in depth_dist):
                        depth_dist[val] = 1
                    else:
                        depth_dist[val] += 1
            
            # plt.plot(depth_dist)
            del depth_dist[0]
            lists = sorted(depth_dist.items()) # sorted by key, return a list of tuples

            x, y = zip(*lists) # unpack a list of pairs into two tuples
            plt.figure()
            plt.scatter(x, y)
            # plt.show()
            plt.savefig(os.path.join(path, s[1][:6], "dist"))
            # for k in sorted(depth_dist.items()):
            #     print(k)
            # print()
            # print()
            # print()
            # print()

            with open('bandeja_standard.pickle', 'rb') as config:
                config_dict = pickle.load(config)


            pcd = o3d.geometry.PointCloud()
            rgb_cnf = np.load('s10_standard_intrinsics(1).npy', allow_pickle=True).item()
            pcd.points = o3d.utility.Vector3dVector(pointcloudify_depth(depth, rgb_cnf['undist_intrinsics'], undistort=False)) # H * W X 3
            

            mask_pc = o3d.geometry.PointCloud()

            mask_pc.points = o3d.utility.Vector3dVector(pointcloudify_depth(gray_mask, rgb_cnf['undist_intrinsics'], undistort=False)) # H * W X 3

            cld = np.asarray(pcd.points)
            mask = np.asarray(mask_pc.points)

            filtered = []
            for n, x in enumerate(cld): 
                if mask[n][2] > 0:
                    filtered.append(cld[n])


            f = o3d.geometry.PointCloud()
            f.points = o3d.utility.Vector3dVector(filtered)

            clustering = DBSCAN(eps = 50)
            # clustering = AgglomerativeClustering(n_clusters =None, distance_threshold = 5000)
            labels = clustering.fit_predict(filtered)
            uniq = {}
            depth_p = {}
            for pos, a in enumerate(labels):
                if not (a in uniq):
                    uniq[a] = 0
                    depth_p[a] = filtered[pos][2]
                else:
                    uniq[a] +=1
            

            ready = []
            #the first cluster is of zero depth so:
            # print(uniq)
            # print(depth_p)
            filtered_depth_p = {}

            for key, dist in depth_p.items():
                if abs(dist) > 0.0001 and uniq[key] > 1000:
                    filtered_depth_p[key] = dist
            mink = min(filtered_depth_p, key=depth_p.get)
            # print(mink)
            for n, x in enumerate(labels): 
                if x == mink:
                    ready.append(filtered[n])

            r = o3d.geometry.PointCloud()
            r.points = o3d.utility.Vector3dVector(ready)
            aligned_depth = project_pcd_to_depth(r, rgb_cnf['undist_intrinsics'], depth.shape[:2])
            for num_c, rows in enumerate(aligned_depth):
                for num_r, col in enumerate(rows):
                    if (aligned_depth[num_c][num_r] > 0):
                        aligned_depth[num_c][num_r] = 255

            imageio.imwrite(os.path.join(path, s[2]), (aligned_depth).astype(np.uint16))


