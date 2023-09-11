from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
from unimatch.geometry import forward_backward_consistency_check

def sintel_read(binary, n_channels):
    TAG_FLOAT = 202021.25
    c = n_channels
    with open(binary, 'rb') as f:
        # Parse biany file header
        tag = float(np.fromfile(f, np.float32, count=1)[0])
        assert(tag == TAG_FLOAT)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]

        # Read in flow data and reshape it
        flow = np.fromfile(f, np.float32, count=-1).reshape(h, w, c).transpose(2, 0, 1)
    
    if c == 2:
        return np.flip(flow, 0).copy()
    
    return flow

def sintel_cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    TAG_FLOAT = 202021.25
    with open(filename,'rb') as f:
        tag = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert(tag == TAG_FLOAT)
        M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
        N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def vkitti_read_flow(f):
    # read png to bgr in 16 bit unsigned short
    bgr = cv2.imread(f, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 1:].astype('f4') - 1
    out_flow[..., 0] *= h - 1
    out_flow[..., 1] *= w - 1
    out_flow[invalid] = 0  # or another value (e.g., np.nan)
    return out_flow.transpose(2, 0, 1)

def vkitti_read_depth(f):
    depth = cv2.imread(f, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100
    return np.expand_dims(depth.astype('f4'), 0)


def get_sintel_data(root, video_name, start_frame=None, num_frames=None,
                    use_normal=True):
    rgbs = sorted(glob(f'{root}/training/clean/{video_name}/*.png'))
    flows = sorted(glob(f'{root}/training/flow/{video_name}/*.flo'))
    occlusions = sorted(glob(f'{root}/training/occlusions/{video_name}/*.png'))
    depths = sorted(glob(f'{root}/training/depth/{video_name}/*.dpt'))
    cams = sorted(glob(f'{root}/training/camdata_left/{video_name}/*.cam'))

    # trim
    if start_frame is not None:
        end_frame = num_frames and start_frame + num_frames
        end_frame_minus_one = end_frame and end_frame - 1
        rgbs = rgbs[start_frame:end_frame]
        flows = flows[start_frame:end_frame_minus_one]
        occlusions = occlusions[start_frame:end_frame_minus_one]
        depths = depths[start_frame:end_frame]
        cams = cams[start_frame:end_frame]

    #load
    rgbs = torch.stack([F.pil_to_tensor(Image.open(f)) for f in rgbs])
    flows = torch.stack([torch.from_numpy(sintel_read(f, 2)) for f in flows])
    occlusions = torch.stack([F.pil_to_tensor(Image.open(f)) for f in occlusions])
    depths = torch.stack([torch.from_numpy(sintel_read(f, 1)) for f in depths])
    cams = torch.stack([torch.from_numpy(sintel_cam_read(f)[0]) for f in cams])

    # remove outliers from depth
    valid_max = depths[(depths-depths.mean()).abs() < 4*depths.std()].max()
    depths = torch.where((depths-depths.mean()).abs() < 4*depths.std(), depths, valid_max)

    # preprocess
    rgbs = rgbs / 255
    flows = flows.permute(0, 2, 3, 1)
    occlusions = occlusions.permute(0, 2, 3, 1).squeeze(-1) > 1
    conds = depths / depths.max()
    conds = 1 - conds
    conds = conds.repeat(1, 3, 1, 1)

    # # dilate the occlusion to make it cover the boundry area
    # occlusions = (torch.nn.functional.conv2d(occlusions[:, None]*1., torch.ones(1, 1, 3, 3)*(1/9), padding='same') > 0).squeeze(1)

    if use_normal:
        dx, dy = torch.gradient(depths, dim=[2, 3])
        dx *= cams[:, 0, 0].view(-1, *([1]*len(depths.shape[1:]))) / depths
        dy *= cams[:, 1, 1].view(-1, *([1]*len(depths.shape[1:]))) / depths
        normal = torch.cat((-dx, -dy, torch.ones_like(depths)), dim=1)
        normal /= ((normal ** 2).sum(dim=1, keepdims=True) ** 0.5)
        conds = (normal * 0.5 + 0.5).clip(0, 1)

    return {
        'rgbs': rgbs,
        'flows': flows,
        'occlusions': occlusions,
        'conds': conds,
        'is_backward_flow': False,
    }

def get_vkitti2_data(root, video_name, start_frame=None, num_frames=None,
                    use_normal=True, use_backward_flow=False):
    rgbs = sorted(glob(f'{root}/{video_name}/clone/frames/rgb/Camera_0/*.jpg'))
    flows = sorted(glob(f'{root}/{video_name}/clone/frames/forwardFlow/Camera_0/*.png'))
    flows_bwd = sorted(glob(f'{root}/{video_name}/clone/frames/backwardFlow/Camera_0/*.png'))
    depths = sorted(glob(f'{root}/{video_name}/clone/frames/depth/Camera_0/*.png'))
    cam = torch.tensor(
        [[725.0087,        0, 620.5],
         [       0, 725.0087, 187.0],
         [       0,        0,     1]]
    ) # https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/

    # trim
    if start_frame is not None:
        num_frames_minus_one = num_frames and num_frames - 1
        rgbs = rgbs[start_frame:start_frame + num_frames]
        flows = flows[start_frame:start_frame + num_frames_minus_one]
        flows_bwd = flows_bwd[start_frame:start_frame + num_frames_minus_one]
        depths = depths[start_frame:start_frame + num_frames]

    # load
    rgbs = torch.stack([F.pil_to_tensor(Image.open(f)) for f in rgbs])
    flows = torch.stack([torch.from_numpy(vkitti_read_flow(f))for f in flows])
    flows_bwd = torch.stack([torch.from_numpy(vkitti_read_flow(f))for f in flows_bwd])
    depths = torch.stack([torch.from_numpy(vkitti_read_depth(f))for f in depths])

    # remove outliers from depth
    valid_max = depths[(depths-depths.mean()).abs() < 4*depths.std()].max()
    depths = torch.where((depths-depths.mean()).abs() < 4*depths.std(), depths, valid_max)

    # derive occlusions
    occlusions, occlusions_bwd = forward_backward_consistency_check(flows.flip(1), flows_bwd.flip(1))

    if use_backward_flow:
        flows = flows_bwd
        occlusions = occlusions_bwd

    # preprocess
    rgbs = rgbs / 255
    flows = flows.permute(0, 2, 3, 1)
    occlusions = (occlusions > 0.5)
    conds = depths / depths.max()
    conds = 1 - conds
    conds = conds.repeat(1, 3, 1, 1)

    # # dilate the occlusion to make it cover the boundry area
    # occlusions = (torch.nn.functional.conv2d(occlusions[:, None]*1., torch.ones(1, 1, 3, 3)*(1/9), padding='same') > 0).squeeze(1)

    if use_normal:
        dx, dy = torch.gradient(depths, dim=[2, 3])
        dx *= cam[0, 0] / depths
        dy *= cam[1, 1] / depths
        normal = torch.cat((-dx, -dy, torch.ones_like(depths)), dim=1)
        normal /= ((normal ** 2).sum(dim=1, keepdims=True) ** 0.5)
        conds = (normal * 0.5 + 0.5).clip(0, 1)

    return {
        'rgbs': rgbs,
        'flows': flows,
        'occlusions': occlusions,
        'conds': conds,
        'is_backward_flow': use_backward_flow,
    }

