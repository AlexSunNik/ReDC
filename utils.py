from scipy.interpolate import Rbf
import os
import numpy as np
import dask.array as da
from scipy.interpolate import griddata


def grid_interpolation(sparse, method='nearest'):
    nz_idx = np.nonzero(sparse)
    X = nz_idx[0]
    Y = nz_idx[1]
    assert len(X) == len(Y)
    vals = np.array([sparse[X[i], Y[i]] for i in range(len(X))])
    points = np.zeros((len(X), 2))
    points[:,0] = Y
    points[:,1] = X
    ti_x = np.arange(sparse.shape[1])
    ti_y = np.arange(sparse.shape[0])
    xx, yy = np.meshgrid(ti_x, ti_y)
    zz = griddata(points, vals, (xx, yy), method=method, rescale=False, fill_value=0)
    return zz

def euclidean_norm_numpy(x1, x2):
    return np.linalg.norm(x1 - x2, axis=0)

def radial_interpolation(pc_rgb_from_sparse, scale, sparse, sr=1):
    """
    RBF.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    -------------
    Parameters:
        pc_rgb_from_sparse (arr): data points from sparse raw dataset, converted to 
                RBG coordinates.
        scale (arr): factor by which to divide default RBF epsilon parameter.
                Set empirically.
        sparse (arr): depth map.
    Returns:
        (float): RMSE over points where both gt and d have data points
    """
    x = pc_rgb_from_sparse[:, 0]
    y = np.max(pc_rgb_from_sparse[:, 1]) - pc_rgb_from_sparse[:, 1]
    z = pc_rgb_from_sparse[:, 2]
    
    x = x[::sr]
    y = y[::sr]
    z = z[::sr]
    
    xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                            for a in [x,y]])
    ximax = np.amax(xi, axis=1)
    ximin = np.amin(xi, axis=1)
    edges = ximax - ximin
    edges = edges[np.nonzero(edges)]
    epsilon = np.power(
        np.prod(edges)/xi.shape[-1], 1.0/edges.size) / scale

    rbf = Rbf(x, y, z, epsilon=epsilon)

    ti_x = np.arange(sparse.shape[1])
    ti_y = np.arange(sparse.shape[0])
    
    xx, yy = np.meshgrid(ti_x, ti_y)
    n1 = xx.shape[1]
    ix = da.from_array(xx, chunks=(5, n1))
    iy = da.from_array(yy, chunks=(5, n1))
    iz = da.map_blocks(rbf, ix, iy)
    zz = iz.compute()

    interpolated_sparse = np.zeros((sparse.shape))
#     print(interpolated_sparse.shape)
#     mask = (np.max(pc_rgb_from_sparse[:, 1]) - yy.flatten()).astype(int), xx.flatten().astype(int)
#     print(len(mask))
    
    interpolated_sparse[(np.max(pc_rgb_from_sparse[:, 1]) - yy.flatten()).astype(int), xx.flatten().astype(int)] = zz.flatten()

    return interpolated_sparse


def depth_map_to_lidar(velo_to_cam_matrix, depth_map, rgb_image=None):
    """
    Projects 2D depth map to LiDAR coordinates.
    i.e. similar data points as the ones from the KITTI raw dataset, except the car's
    egomotion has already been compensated for in the KITTI depth dataset.
    -------------
    Parameters:
        velo_to_cam_matrix (arr): found in KITTI dataset.
        depth_map (arr): 2D depth map from the KITTI depth completion dataset.
    Returns:
        dict_res (dic): dictionary containing LiDAR point cloud
                {"velo": array of points in LiDAR coordinate system, 
                "pc": array of points in RGB coordinate system,
                "rgb feats": array of RGB features.}
    """
    dict_res = {}

    #find data points in depth map (i.e. where > 0)
    indices = np.where(depth_map > 0)
    indices_ = list(indices).copy()
    indices_[0] = indices[1]
    indices_[1] = indices[0]
    
    #create array of these points, in RGB coordinates
    velo_pts_im_ = np.zeros((len(indices[0]), 3))
    velo_pts_im_[:, 2] = depth_map[indices]
    velo_pts_im_[:, :2] = np.asarray(indices_).T
    velo_pts_im = velo_pts_im_.copy()
    velo_pts_im[:, :2] = velo_pts_im_[:, :2] * depth_map[indices][..., np.newaxis]
    velo_pts_im = velo_pts_im.T
 
    #inverse transformation to project points to LiDAR coordinates
    A = velo_to_cam_matrix[:3, :3]
    B = velo_to_cam_matrix[:, 3].reshape((-1, 1))
    velo = np.linalg.solve(A, velo_pts_im - B).T

    dict_res['velo'] = velo
    dict_res['pc'] = velo_pts_im_
    if rgb_image is not None:
        dict_res['rgb feats'] = rgb_image[indices]

    return dict_res

def load_calib_matrices(args, calib_name='calib_velo_to_cam.txt'):
        calib_matrices = {}
        path = args.data_folder_rgb
        
        days = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
        for day in days:
            path_day = os.path.join(
                path, day)

            cam_to_velo_matrix_left = calib_read(path_day, 2, calib_name)
            cam_to_velo_matrix_right = calib_read(path_day, 3, calib_name)
            calib_matrices[day] = (cam_to_velo_matrix_left,
                                   cam_to_velo_matrix_right)
        return calib_matrices

    
def calib_read(calib_folder, projection_id, calib_name='calib_velo_to_cam.txt'):
    #print('PROJ ID', projection_id)
    #projection_id = 2
    # TODO INVESTIGATE WHY PROJ ID = 3 IS SET

    calib_velo_to_cam_file = os.path.join(
        calib_folder, calib_name)
    with open(calib_velo_to_cam_file, 'r') as f:
        f.readline()
        R_cam_velo_str = f.readline()
        t_cam_velo_str = f.readline()
    R_cam_velo_split = R_cam_velo_str.split(' ')[1:]
    R_cam_velo_split = [float(r) for r in R_cam_velo_split]
    R_cam_velo = np.array(R_cam_velo_split).reshape(3, 3)
    t_cam_velo_split = t_cam_velo_str.split(' ')[1:]
    t_cam_velo_split = [float(t) for t in t_cam_velo_split]
    t_cam_velo = np.array(t_cam_velo_split)

    T_cam_velo = np.zeros((4, 4))
    T_cam_velo[:3, :3] = R_cam_velo
    T_cam_velo[:3, 3] = t_cam_velo
    T_cam_velo[3, 3] = 1
    res = T_cam_velo

    if calib_name == 'calib_velo_to_cam.txt':
        calib_cam_to_cam_file = os.path.join(calib_folder, 'calib_cam_to_cam.txt')
        with open(calib_cam_to_cam_file, 'r') as f:
            lines = f.readlines()
            lines = [line.replace('\n', '') for line in lines]

            R_0_rect_str = lines[8]
            if projection_id == 2:
                P_i_rect_str = lines[25]
            elif projection_id == 3:
                P_i_rect_str = lines[33]

            R_0_rect_split = R_0_rect_str.split(' ')
            assert R_0_rect_split[0][:-1] == 'R_rect_00'
            R_0_rect_split = R_0_rect_split[1:]
            R_0_rect_split = [float(r) for r in R_0_rect_split]
            R_0_rect = np.array(R_0_rect_split).reshape(3, 3)

            P_i_rect_split = P_i_rect_str.split(' ')
            assert (P_i_rect_split[0][:-1] ==
                    'P_rect_02' if projection_id == 2 else 'P_rect_03')
            P_i_rect_split = P_i_rect_split[1:]
            P_i_rect_split = [float(r) for r in P_i_rect_split]
            P_i_rect = np.array(P_i_rect_split).reshape(3, 4)

        R_0_rect_expanded = np.zeros((4, 4))
        R_0_rect_expanded[:3, :3] = R_0_rect
        R_0_rect_expanded[3, 3] = 1

        RT = np.dot(R_0_rect_expanded, T_cam_velo)
        res = np.dot(P_i_rect, RT)

    return res