import copy
import os
os.environ["PYOPENGL_PLATFORM"] = "OSMesa"
import sys

import torch
from tqdm import tqdm
import cv2
import time
import torchvision.transforms as standard

sys.path.append(os.path.abspath(os.path.dirname(__file__)))     # append current dir to PATH
from base import Tester
from config import cfg
from utils.visualize import draw_2d_skeleton
from data.processing import inference_extraHM, augmentation


def _get_input(frame):
    ### load image from recorded files ###
    load_filepath = './recorded_files/'

    color = cv2.imread(load_filepath + 'color_%d.png' % frame)
    color = cv2.resize(color, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    return color

def _visualize(color, coords_uvd):
    vis = draw_2d_skeleton(color, coords_uvd[cfg.num_vert:])
    vis = cv2.resize(vis, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    color = cv2.resize(color, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("vis", vis)
    cv2.imshow("img", color)
    cv2.waitKey(50)

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

class HandTracker():
    def __init__(self):
        self.tester = Tester()

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])

        if cfg.extra:
            self.extra_uvd = np.zeros((21, 3), dtype=np.float32)
            self.idx = 0

    def run(self, img):
        # input
        rgb = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
        # init_data = pil_to_tensor(rgb).cuda()     ~ include ToTensor(), Normalize(*mean_std)

        # need to check below processing available in gpu location. (previously done in cpu area)



        ### crop image ###

        bbox = [bb_x_min, bb_y_min, bbox_size, bbox_size]

        img, img2bb_trans, bb2img_trans, _, _, = \
            augmentation(img, bbox, 'evaluation', exclude_flip=False)

        # img = self.transform(img)     # already done in server.py
        img = torch.unsqueeze(img, 0).type(torch.float32)
        inputs = {'img': img}

        if cfg.extra:
            # affine transform x,y coordinates with current cropinfo
            uv1 = np.concatenate((self.extra_uvd[:, :2], np.ones_like(self.extra_uvd[:, :1])), 1)
            self.extra_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize uv, depth is already relative value
            self.extra_uvd[:, :2] = self.extra_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            extra_hm = inference_extraHM(self.extra_uvd, self.idx, reinit_num=10)
            inputs['extra'] = torch.unsqueeze(torch.from_numpy(extra_hm), dim=0)
            self.idx += 1

        with torch.no_grad():
            outs = self.tester.model(inputs)

        outs = {k: v.cpu().numpy() for k, v in outs.items()}
        coords_uvd = outs['coords'][0]

        # normalized value to uv(pixel) range
        coords_uvd[:, :2] = (coords_uvd[:, :2] + 1) * (cfg.input_img_shape[0] // 2)

        # back to original image
        uv1 = np.concatenate((coords_uvd[:, :2], np.ones_like(coords_uvd[:, :1])), 1)
        coords_uvd[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

        if cfg.extra:
            self.extra_uvd = copy.deepcopy(coords_uvd[cfg.num_vert:])

        # restore depth value after passing extra pose
        coords_uvd[:, 2] = coords_uvd[:, 2] * cfg.depth_box + root_depth

        all_uvd = copy.deepcopy(coords_uvd)
        return all_uvd



def main():
    torch.backends.cudnn.benchmark = True

    tracker = HandTracker()

    cam_intrinsic = None

    frame = 1
    while True:
        color = _get_input(frame)

        all_uvd = tracker.run(color)

        ### if required uvd format ##
        mesh_uvd = copy.deepcopy(all_uvd[:cfg.num_vert])       # (778, 3)
        joint_uvd = copy.deepcopy(all_uvd[cfg.num_vert:])       # (21, 3)

        ### if required xyz format ###
        # all_xyz = uvd2xyz(all_uvd, cam_intrinsic)

        _visualize(color, joint_uvd)

        frame += 1


if __name__ == '__main__':
    main()



