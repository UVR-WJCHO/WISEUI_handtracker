import os
os.environ["PYOPENGL_PLATFORM"] = "OSMesa"
import torch
from tqdm import tqdm
import cv2
import time
import torchvision.transforms as standard

from base import Tester
from config import cfg
from utils.visualize import *
from data.processing import *


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


def generate_extra(extra_uvd, idx, reinit_num=10):
    extra_width = cfg.extra_width

    # re-initialize extra heatmap at every n frames
    if idx % int(reinit_num) == 0:
        extra_hm = np.zeros((1, extra_width, extra_width), dtype=np.float32)
        return extra_hm

    else:
        extra_hm = np.zeros((extra_width, extra_width), dtype=np.float32)
        ratio = int(256 / extra_width)

        for i in range(21):
            u = int(np.clip(extra_uvd[i, 0], 0, 255) / float(ratio))
            v = int(np.clip(extra_uvd[i, 1], 0, 255) / float(ratio))
            extra_hm[u, v] = extra_uvd[i, 2]

            if u - 2 >= 0 and v - 2 >= 0 and u + 2 < extra_width and v + 2 < extra_width:
                val = extra_uvd[i, 2] / 2.0
                extra_hm[u - 2, v] = val
                extra_hm[u + 2, v] = val
                extra_hm[u, v - 2] = val
                extra_hm[u, v + 2] = val

                val = extra_uvd[i, 2] / 3.0
                extra_hm[u - 2, v - 2] = val
                extra_hm[u - 2, v + 2] = val
                extra_hm[u + 2, v - 2] = val
                extra_hm[u + 2, v + 2] = val

            if u - 4 >= 0 and v - 4 >= 0 and u + 4 < extra_width and v + 4 < extra_width:
                val = extra_uvd[i, 2] / 5.0
                extra_hm[u - 4, v] = val
                extra_hm[u + 4, v] = val
                extra_hm[u, v - 4] = val
                extra_hm[u, v + 4] = val

        return np.expand_dims(extra_hm, axis=0)


class HandTracker():
    def __init__(self):
        self.tester = Tester()

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])

        if cfg.extra:
            self.extra_uvd = np.zeros((21, 3), dtype=np.float32)
            self.extra_hm = np.zeros((1, cfg.extra_width, cfg.extra_width), dtype=np.float32)
            self.idx = 0

    def run(self, img):
        img = self.transform(img)
        img = torch.unsqueeze(img, 0).type(torch.float32)
        inputs = {'img': img}

        ### crop image ###


        if cfg.extra:
            self.extra_hm = generate_extra(self.extra_uvd, self.idx, reinit_num=10)
            inputs['extra'] = torch.unsqueeze(torch.from_numpy(self.extra_hm), dim=0)
            self.idx += 1

        with torch.no_grad():
            outs = self.tester.model(inputs)

        outs = {k: v.cpu().numpy() for k, v in outs.items()}
        coords_uvd = outs['coords'][0]
        coords_uvd[:, :2] = (coords_uvd[:, :2] + 1) * cfg.input_img_shape[0] // 2

        if cfg.extra:
            self.extra_uvd = np.copy(coords_uvd[cfg.num_vert:])

        return coords_uvd


    def get_record(self, img):
        img = self.transform(img)
        inputs = np.float32(img)
        targets = {}

        image = torch.from_numpy(inputs)
        img = torch.unsqueeze(image, 0).type(torch.float32)
        inputs = {'img': img}
        return inputs, targets


def main():
    torch.backends.cudnn.benchmark = True

    tracker = HandTracker()

    frame = 1
    while True:
        color = _get_input(frame)

        coords_uvd = tracker.run(color)

        _visualize(color, coords_uvd)

        frame += 1


if __name__ == '__main__':
    main()


