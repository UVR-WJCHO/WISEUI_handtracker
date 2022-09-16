import os
import json
import numpy as np
import torch
import cv2
import pickle
import time
from torch.utils.data import Dataset
from torchvision import transforms, utils

from data.vis_utils import *
from data.processing import xyz2uvd, uvd2xyz, read_img, load_db_annotation, augmentation, cv2pil, _assert_exist
from data.processing import get_focal_pp

from config import cfg
from utils.visualize import render_mesh_multi_views, draw_2d_skeleton, render_mesh
from utils.mano import MANO
from manopth import argutils
from manopth.manolayer import ManoLayer
import statistics

MANO_MODEL_PATH = './data/mano/models/MANO_RIGHT.pkl'

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]
VISIBLE_PARAM = 0.025

def generate_fake_prevpose(joint_uvd, weight=1.0):
    # (21, 3), uv range : (0~256), d range : (-0.10 ~ 0.10)

    extra_uvd = np.copy(joint_uvd)
    extra_uvd = random_translate_pose(extra_uvd, weight=weight)

    noise_w = weight / 2.0
    ref_value = 5.0
    extra_uvd[:, 0] += np.random.normal(-1 * ref_value * noise_w, ref_value * noise_w, 21)
    extra_uvd[:, 1] += np.random.normal(-1 * ref_value * noise_w, ref_value * noise_w, 21)
    extra_uvd[:, 2] += np.random.normal(-0.003 * noise_w, 0.003 * noise_w, 21)

    return extra_uvd, weight

def random_translate_pose(joint_uvd, weight=1.0):
    extra_uvd = np.copy(joint_uvd)
    ref_value = 10.0
    extra_uvd[:, 0] += np.random.normal(-1 * ref_value * weight, ref_value * weight, 1)
    extra_uvd[:, 1] += np.random.normal(-1 * ref_value * weight, ref_value * weight, 1)
    extra_uvd[:, 2] += np.random.normal(-0.01 * weight, 0.01 * weight, 1)

    return extra_uvd


def generate_extraFeature(curr_uvd, ratio=[0.2, 0.3, 0.2, 0.3], debug=None):
    flag = int(np.random.choice(4, 1, p=ratio))
    if flag is 0:
        extra_uvd = np.copy(curr_uvd)
        w_aug = 0.0
    elif flag is 1:
        extra_uvd, w_aug = generate_fake_prevpose(curr_uvd, weight=1.0)
    elif flag is 2:
        w = np.random.uniform(2., 4.)
        extra_uvd, w_aug = generate_fake_prevpose(curr_uvd, weight=w)
    else:
        extra_uvd = np.zeros((21, 3))
        w_aug = 99.0

    ### debug ###
    # debug = debug / 255.0
    # vis_curr = draw_2d_skeleton(np.copy(debug), curr_uvd)
    # vis_extra = draw_2d_skeleton(np.copy(debug), extra_uvd)
    # vis_curr = cv2.resize(vis_curr, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    # vis_extra = cv2.resize(vis_extra, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("vis_curr", vis_curr)
    # cv2.imshow("vis_extra", vis_extra)
    # cv2.waitKey(0)

    extra_width = cfg.extra_width
    ratio = int(256. / extra_width)

    extra_hm = np.zeros((extra_width, extra_width), dtype=np.float32)
    extra_uvd_hm = np.copy(extra_uvd)
    # extra_hm = np.zeros((128, 128), dtype=np.float32)
    for i in range(21):
        u = int(np.clip(extra_uvd_hm[i, 0], 0, 255) / float(ratio))
        v = int(np.clip(extra_uvd_hm[i, 1], 0, 255) / float(ratio))
        extra_hm[u, v] = extra_uvd_hm[i, 2]

        if u - 2 >= 0 and v - 2 >= 0 and u + 2 < extra_width and v + 2 < extra_width:
            val = extra_uvd_hm[i, 2] / 2.0
            extra_hm[u - 2, v] = val
            extra_hm[u + 2, v] = val
            extra_hm[u, v - 2] = val
            extra_hm[u, v + 2] = val

            val = extra_uvd_hm[i, 2] / 3.0
            extra_hm[u - 2, v - 2] = val
            extra_hm[u - 2, v + 2] = val
            extra_hm[u + 2, v - 2] = val
            extra_hm[u + 2, v + 2] = val

        if u - 4 >= 0 and v - 4 >= 0 and u + 4 < extra_width and v + 4 < extra_width:
            val = extra_uvd_hm[i, 2] / 5.0
            extra_hm[u - 4, v] = val
            extra_hm[u + 4, v] = val
            extra_hm[u, v - 4] = val
            extra_hm[u, v + 4] = val

    extra_hm = np.expand_dims(extra_hm, axis=0)

    # w = 0 if optimal feature, w = 1~2.5 as noise scale, ...
    w_aug = 1.0 / (w_aug * 1.5 + 1.0)

    return extra_uvd, extra_hm, w_aug


class FreiHAND(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.extra = cfg.extra
        assert self.mode in ['training', 'evaluation'], 'mode error'

        # load annotations
        self.anno_all = load_db_annotation(root, self.mode)
        if self.mode == 'evaluation':
            root = os.path.join(root, 'bbox_root_freihand_output.json')
            self.root_result = []
            with open(root) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                self.root_result.append(np.array(annot[i]['root_cam']))
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
        self.versions = ['gs', 'hom', 'sample', 'auto']
        self.img_load = None

    def __getitem__(self, idx):
        if self.mode == 'training':
            version = self.versions[idx // len(self.anno_all)]
        else:
            version = 'gs'
        idx = idx % len(self.anno_all)
        img = read_img(idx, self.root, self.mode, version)# / 255.0

        # img_debug = np.copy(img)
        # cv2.imshow("Frei img before augment", img_debug)
        # cv2.waitKey(1)

        bbox_size = 130
        bbox = [112 - bbox_size//2, 112 - bbox_size//2, bbox_size, bbox_size]
        img, img2bb_trans, bb2img_trans, _, _,  = \
            augmentation(img, bbox, self.mode, exclude_flip=True)

        # img_debug = np.copy(img) / 255.0
        # cv2.imshow("Frei img", img_debug)
        # cv2.waitKey(1)

        img_cv = np.copy(img)
        img_pil = cv2pil(img)
        img = self.transform(img_pil)

        if self.mode == 'training':
            K, mesh_xyz, pose_xyz, scale = self.anno_all[idx]
            K, mesh_xyz, pose_xyz, scale = [np.array(x) for x in [K, mesh_xyz, pose_xyz, scale]]
            # concat mesh and pose label
            all_xyz = np.concatenate((mesh_xyz, pose_xyz), axis=0)
            all_uvd = xyz2uvd(all_xyz, K)   # (799, 3)
            # affine transform x,y coordinates
            uv1 = np.concatenate((all_uvd[:, :2], np.ones_like(all_uvd[:, :1])), 1)
            all_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # wrist is the relative joint
            root_depth = all_uvd[cfg.num_vert:cfg.num_vert + 1, 2:3].copy()
            all_uvd[:, 2:3] = (all_uvd[:, 2:3] - root_depth)
            # box to normalize depth
            all_uvd[:, 2:3] /= cfg.depth_box

            if self.extra:
                extra_uvd, extra_hm, w_aug = generate_extraFeature(all_uvd[cfg.num_vert:, :], debug=img_cv)
                # normalize
                extra_uvd[:, :2] = extra_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            # normalize
            all_uvd[:, :2] = all_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            if self.extra:
                inputs = {'img': np.float32(img), 'extra': np.float32(extra_hm)}
                targets = {'mesh_pose_uvd': np.float32(all_uvd), 'extra_uvd': np.float32(extra_uvd), 'weight_aug': np.float32(w_aug)}
            else:
                inputs = {'img': np.float32(img)}
                targets = {'mesh_pose_uvd': np.float32(all_uvd)}

            meta_info = {}
        else:
            K, scale = self.anno_all[idx]
            K, scale = [np.array(x) for x in [K, scale]]

            inputs = {'img': np.float32(img)}
            targets = {}
            meta_info = {
                'img2bb_trans': np.float32(img2bb_trans),
                'bb2img_trans': np.float32(bb2img_trans),
                'root_depth': np.float32(self.root_result[idx][2][None]),
                'K': np.float32(K),
                'scale': np.float32(scale)}

        return inputs, targets, meta_info

    def __len__(self):
        if self.mode == 'training':
            return len(self.anno_all) * 4
        else:
            return len(self.anno_all)

    def get_img(self):
        return self.img_load

    def get_record(self, img):
        img = self.transform(img)
        inputs = np.float32(img)
        targets = {}
        return inputs, targets

    def evaluate(self, outs, meta_info, cur_sample_idx):
        coords_uvd = outs['coords']
        batch = coords_uvd.shape[0]
        eval_result = {'pose_out': list(), 'mesh_out': list()}
        for i in range(batch):
            coord_uvd_crop, root_depth, img2bb_trans, bb2img_trans, K, scale = \
                coords_uvd[i], meta_info['root_depth'][i], meta_info['img2bb_trans'][i], \
                meta_info['bb2img_trans'][i], meta_info['K'][i], meta_info['scale'][i]
            coord_uvd_crop[:, 2] = coord_uvd_crop[:, 2] * cfg.depth_box + root_depth
            coord_uvd_crop[:, :2] = (coord_uvd_crop[:, :2] + 1) * cfg.input_img_shape[0] // 2
            # back to original image
            coord_uvd_full = coord_uvd_crop.copy()
            uv1 = np.concatenate((coord_uvd_full[:, :2], np.ones_like(coord_uvd_full[:, :1])), 1)
            coord_uvd_full[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
            coord_xyz = uvd2xyz(coord_uvd_full, K)
            pose_xyz = coord_xyz[cfg.num_vert:]
            mesh_xyz = coord_xyz[:cfg.num_vert]
            eval_result['pose_out'].append(pose_xyz.tolist())
            eval_result['mesh_out'].append(mesh_xyz.tolist())
            if cfg.vis:
                mesh_xyz_crop = uvd2xyz(coord_uvd_crop[:cfg.num_vert], K)
                vis_root = os.path.join(cfg.output_root, 'FreiHAND_vis')
                if not os.path.exists(vis_root):
                    os.makedirs(vis_root)
                idx = cur_sample_idx + i
                img_full = read_img(idx, self.root, 'evaluation', 'gs')
                img_crop = cv2.warpAffine(img_full, img2bb_trans, cfg.input_img_shape, flags=cv2.INTER_LINEAR)
                focal, pp = get_focal_pp(K)
                cam_param = {'focal': focal, 'princpt': pp}

                # img_mesh, view_1, view_2 = render_mesh_multi_views(img_crop, mesh_xyz_crop, self.face, cam_param)
                # path_mesh_img = os.path.join(vis_root, 'render_mesh_img_{}.png'.format(idx))
                # cv2.imwrite(path_mesh_img, img_mesh)
                # path_mesh_view1 = os.path.join(vis_root, 'render_mesh_view1_{}.png'.format(idx))
                # cv2.imwrite(path_mesh_view1, view_1)
                # path_mesh_view2 = os.path.join(vis_root, 'render_mesh_view2_{}.png'.format(idx))
                # cv2.imwrite(path_mesh_view2, view_2)
                path_joint = os.path.join(vis_root, 'joint_{}.png'.format(idx))
                vis = draw_2d_skeleton(img_crop, coord_uvd_crop[cfg.num_vert:])
                cv2.imwrite(path_joint, vis)
                path_img = os.path.join(vis_root, 'img_{}.png'.format(idx))
                cv2.imwrite(path_img, img_crop)

        return eval_result

    def print_eval_result(self, eval_result):
        output_json_save_path = os.path.join('./', 'pred_frei.json')
        with open(output_json_save_path, 'w') as fo:
            json.dump([eval_result['pose_out'], eval_result['mesh_out']], fo)
        print('Dumped %d joints and %d verts predictions to %s'
              % (len(eval_result['pose_out']), len(eval_result['mesh_out']), output_json_save_path))


class HO3D_v2(Dataset):
    def __init__(self, root='../../dataset/HO3D_v2', mode='training', loadit=True, validset=False):
        ###
        # initial setting
        # 640*480 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        # hand joint order : Mano
        ###
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.root = root
        if mode == 'training':
            self.mode = 'train'
            self.name = 'train'
            self._mode = mode
        if mode == 'evaluation':
            self.mode = mode
            self.name = 'valid'
            self._mode = mode

        self.loadit = loadit
        self.extra = cfg.extra

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

        self.IMAGE_WIDTH = 640.
        self.IMAGE_HEIGHT = 480.
        self.camera_intrinsics = np.array([[617.627, 0, 320.262],
                                           [0, 617.101, 238.469],
                                           [0, 0, 1]])
        """
        # load meshes to memory
        # object_root = os.path.join(self.root, 'Object_models')
        # self.objects = self.load_objects(object_root)

        # self.camera_pose = np.array(
        #     [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
        #      [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
        #      [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
        #      [0, 0, 0, 1]])

        # self.camera_pose = np.array(
        #     [[1., 0., 0., 0.],
        #      [0., 1., 0., 0.],
        #      [0., 0., 1., 0.],
        #      [0, 0, 0, 1]])
        """

        print('Loading HO3D_v2 dataset index ...')
        t = time.time()

        if not loadit:
            layer = ManoLayer(
                side='right',
                mano_root='./data/mano/models',
                ncomps=6,
                root_rot_mode='axisang',
                joint_rot_mode='axisang')

            layer.cuda()

            # subjects = [1, 2, 3, 4, 5, 6]
            # subject = "Subject_1"
            # subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
            # actions = os.listdir(subject)

            subject_path = os.path.join(root, self.mode)
            subjects = os.listdir(subject_path)

            dataset = dict()
            dataset['train'] = dict()
            dataset['valid'] = dict()

            for subject in subjects:
                subject = str(subject)

                dataset['train'][subject] = list()
                dataset['valid'][subject] = list()

                rgb_set = list(os.listdir(os.path.join(root, self.mode, subject, 'rgb')))
                frames = len(rgb_set)

                if validset:
                    data_split = int(frames)  # * 1 / 30) + 1
                    data_split_train = int(data_split * 7 / 8) + 1
                    data_split_valid = data_split - data_split_train

                    for i in range(frames):
                        if i < data_split_train:
                            dataset['train'][subject].append(rgb_set[i])
                        else:
                            dataset['valid'][subject].append(rgb_set[i])

                    # if required random valid
                    # valid_set = np.random.choice(data_split, data_split_valid)
                    # for i in range(frames):
                    #     if i in valid_set:
                    #         dataset['test'][subject].append(rgb_set[i])
                    #     else:
                    #         dataset['train'][subject].append(rgb_set[i])

                else:
                    for i in range(frames):
                        dataset['train'][subject].append(rgb_set[i])

            #print(yaml.dump(dataset))

            MKA_list = list()

            modes = ['train', 'valid']
            for i in range(2):
                self.samples = dict()
                self.name = modes[i]
                idx = 0
                for subject in list(dataset[modes[i]]):

                    flag_start = 0
                    kps_list = np.zeros((3, 21, 3), dtype=float)

                    for frame in dataset[modes[i]][subject]:
                        sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                        }
                        _, depth, meta = self.read_data(sample)

                        # get the hand Mesh from MANO model for the current pose
                        self.camera_intrinsics = meta['camMat']

                        #_, handMesh = forwardKinematics(meta['handPose'], meta['handTrans'], meta['handBeta'])
                        pose_params, beta, trans = meta['handPose'], meta['handBeta'], meta['handTrans']
                        pose_params = torch.from_numpy(np.expand_dims(pose_params, axis=0)).cuda()
                        beta = torch.from_numpy(np.expand_dims(beta, axis=0)).cuda()
                        trans = torch.from_numpy(np.expand_dims(trans, axis=0)).cuda()

                        verts, Jtr = layer(pose_params, th_betas=beta, th_trans=trans)
                        verts = np.squeeze(verts.cpu().numpy()) / 1000.
                        Jtr = np.squeeze(Jtr.cpu().numpy())

                        verts = verts.dot(self.coord_change_mat.T)
                        Jtr = Jtr.dot(self.coord_change_mat.T)

                        verts_uvd = xyz2uvd(verts, self.camera_intrinsics)

                        objCorners = meta['objCorners3DRest']
                        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(meta['objRot'])[0].T) + meta['objTrans']
                        objCornersTrans = objCornersTrans.dot(self.coord_change_mat.T) * 1000.
                        objJoints3D = self.get_box_3d_control_points(objCornersTrans)

                        handJoints3D = meta['handJoints3D']
                        handJoints3D = handJoints3D.dot(self.coord_change_mat.T) * 1000.

                        objKps = project_3D_points(meta['camMat'], objJoints3D, is_OpenGL_coords=False)
                        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

                        handKps = handKps[jointsMapManoToSimple]
                        handJoints3D = handJoints3D[jointsMapManoToSimple]

                        ### GT MKA calculation ###

                        kps3D = np.copy(handJoints3D)

                        if flag_start is 0:
                            flag_start = 1
                            kps_list[0] = kps3D
                        elif flag_start is 1:
                            flag_start = 2
                            kps_list[1] = kps3D
                        else:
                            kps_list[2] = kps3D

                            prev_xyz = np.copy(kps_list[0])
                            curr_xyz = np.copy(kps_list[1])
                            futu_xyz = np.copy(kps_list[2])

                            acc = prev_xyz + futu_xyz - 2 * curr_xyz
                            acc_3d = np.square(acc[:, 0]) + np.square(acc[:, 1]) + np.square(acc[:, 2])
                            acc_3d = np.sqrt(acc_3d)  # should be [21, ]
                            acc_avg = np.average(acc_3d)

                            if not acc_avg > 10:
                                MKA_list.append(acc_avg)

                            kps_list[0] = np.copy(kps_list[1])
                            kps_list[1] = np.copy(kps_list[2])

                        ### generate visibility value ###
                        handKps_ = np.copy(handKps)
                        handJoints3D_ = np.copy(handJoints3D) / 1000.  # mm to GT unit
                        handKps_ = np.round(handKps_).astype(np.int)

                        # depth : (256, 256)
                        # handKps_ : (21, 2)
                        # handJoints3D_ : (21, 3)
                        visible = []
                        for i_vis in range(21):
                            if handKps_[i_vis][0] >= self.IMAGE_WIDTH or handKps_[i_vis][1] >= self.IMAGE_HEIGHT:
                                continue
                            d_img = depth[handKps_[i_vis][1], handKps_[i_vis][0]]
                            d_gt = handJoints3D_[i_vis][-1]
                            if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                                visible.append(i_vis)

                        ### generate BoundingBox ###
                        x_min = int(np.min(handKps[:, 0]))
                        x_max = int(np.max(handKps[:, 0]))
                        y_min = int(np.min(handKps[:, 1]))
                        y_max = int(np.max(handKps[:, 1]))

                        bbox = [x_min, y_min, x_max, y_max]

                        new_sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                            'handJoints3D': handJoints3D,
                            'handKps': handKps,
                            'objJoints3D': objJoints3D,
                            'objKps': objKps,
                            'visible': visible,
                            'bb': bbox,
                            'verts_uvd': verts_uvd,
                            'camMat': meta['camMat']
                        }
                        self.samples[idx] = new_sample
                        idx += 1
                        if idx % 1000 == 0:
                            print("preprocessing idx : ", idx)

                self.clean_data()
                self.save_samples()
                print('Saving done, cfg_HO3D_v2')

            total_MKA_avg = statistics.mean(MKA_list)
            print("total_MKA_avg : ", total_MKA_avg)

        else:
            self.samples = self.load_samples()
            ### test meta data has missing annotation, only acquire images in 'train' folder ###

            print('Loading of %d samples done in %.2f seconds' % (len(self.samples), time.time() - t))

    def load_samples(self):
        with open('data/cfg_HO3D_v2/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self):
        with open('data/cfg_HO3D_v2/{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(list(self.samples), f, pickle.HIGHEST_PROTOCOL)

    def clean_data(self):
        print("Size beforing cleaning: {}".format(len(self.samples.keys())))

        for key in list(self.samples):
            try:
                self.__getitem__(key)
            except Exception as e:
                print(e)
                print("Index failed: {}".format(key))
                del self.samples[key]

        self.samples = list(self.samples.values())
        print("Size after cleaning: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.preprocess(idx)

    def get_image(self, sample):
        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)

        img = cv2.imread(img_path)
        return img

    """
    def get_image(self, sample):
        img = self.fetch_image(sample)
        # if self.mode == 'train':
        #     img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(0)
        return img
    
    def fetch_image(self, sample):
        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)

        img = Image.open(img_path)
        return img
    
    def get_depth(self, sample):
        img = self.fetch_depth(sample)
        img = Image.fromarray(img)

        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        # cv2.imshow("depth in dataset", img)
        # cv2.waitKey(0)
        return img
    
    def fetch_depth(self, sample):
        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'depth', file_name)
        _assert_exist(img_path)
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(img_path)

        dpt = depth[:, :, 2] + depth[:, :, 1] * 256
        dpt = dpt * depth_scale

        return dpt
    """

    def read_data(self, sample):
        file_name = sample['frame_idx'] + '.pkl'
        meta_path = os.path.join(self.root, 'train', sample['subject'], 'meta', file_name)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)
        rgb = cv2.imread(img_path)

        img_path = os.path.join(self.root, 'train', sample['subject'], 'depth', file_name)
        _assert_exist(img_path)
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(img_path)

        dpt = depth[:, :, 2] + depth[:, :, 1] * 256
        dpt = dpt * depth_scale

        return rgb, dpt, meta


    def preprocess(self, idx):
        """
        objTrans: A 3x1 vector representing object translation
        objRot: A 3x1 vector representing object rotation in axis-angle representation
        handPose: A 48x1 vector represeting the 3D rotation of the 16 hand joints including the root joint in axis-angle representation. The ordering of the joints follow the MANO model convention (see joint_order.png) and can be directly fed to MANO model.
        handTrans: A 3x1 vector representing the hand translation
        handBeta: A 10x1 vector representing the MANO hand shape parameters
        handJoints3D: A 21x3 matrix representing the 21 3D hand joint locations
        objCorners3D: A 8x3 matrix representing the 3D bounding box corners of the object
        objCorners3DRest: A 8x3 matrix representing the 3D bounding box corners of the object before applying the transformation
        objName: Name of the object as given in YCB dataset
        objLabel: Object label as given in YCB dataset
        camMat: Intrinsic camera parameters
        """
        # idx = idx % (self.sample_len)
        sample = self.samples[idx]

        # frame_idx = int(sample['frame_idx'])
        # objKps = np.copy(sample['objKps'])
        # objJoints3D = np.copy(sample['objJoints3D'])

        handKps = np.copy(sample['handKps'])
        handJoints3D = np.copy(sample['handJoints3D'])
        verts_uvd = np.copy(sample['verts_uvd'])
        visible = np.copy(sample['visible'])

        img = self.get_image(sample)

        # img_debug = np.copy(img)
        # cv2.imshow("HO3D img before augment", img_debug)
        # cv2.waitKey(1)

        bbox = sample['bb']
        x_min, y_min, x_max, y_max = bbox

        bbox_size = np.clip(max(x_max - x_min, y_max - y_min) + 160, 0, 480)
        x_min = np.clip(x_min - 80, 0, 640)
        y_min = np.clip(y_min - 80, 0, 480)

        if (x_min + bbox_size) > 640:
            margin = (x_min + bbox_size) - 640
            x_min = x_min - margin
        if (y_min + bbox_size) > 480:
            margin = (y_min + bbox_size) - 640
            y_min = y_min - margin

        bbox = [x_min, y_min, bbox_size, bbox_size]

        img, img2bb_trans, bb2img_trans, _, _, = \
            augmentation(img, bbox, self._mode, exclude_flip=True)

        # img_debug = np.copy(img) / 255.0
        # cv2.imshow("HO3D img", img_debug)
        # cv2.waitKey(5)
        img_cv = np.copy(img)
        img_pil = cv2pil(img)
        self.img_load = img_pil
        img = self.transform(img_pil)

        if self.mode == 'train':
            all_uvd = np.zeros((799, 3), dtype=np.float32)
            all_uvd[:cfg.num_vert, :] = np.copy(verts_uvd)
            all_uvd[cfg.num_vert:, :-1] = np.copy(handKps)
            all_uvd[cfg.num_vert:, -1] = np.copy(handJoints3D[:, -1] / 1000.)

            # affine transform x,y coordinates
            uv1 = np.concatenate((all_uvd[:, :2], np.ones_like(all_uvd[:, :1])), 1)
            all_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # # imgAnno = showHandJoints(img_debug, pose_uvd[:, :2])
            # imgAnno = showHandJoints_vis(img_debug, all_uvd[cfg.num_vert:, :-1], visible)
            # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
            # cv2.imshow("rgb pred", imgAnno_rgb)
            # cv2.waitKey(0)

            # wrist is the relative joint & box to normalize depth
            root_depth = all_uvd[cfg.num_vert:cfg.num_vert + 1, 2:3].copy()
            all_uvd[:, 2:3] = (all_uvd[:, 2:3] - root_depth)
            all_uvd[:, 2:3] /= cfg.depth_box

            if self.extra:
                extra_uvd, extra_hm, w_aug = generate_extraFeature(all_uvd[cfg.num_vert:, :], debug=img_cv)
                # normalize
                extra_uvd[:, :2] = extra_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            # img normalize
            all_uvd[:, :2] = all_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1

            if self.extra:
                inputs = {'img': np.float32(img), 'extra': np.float32(extra_hm)}
                targets = {'mesh_pose_uvd': np.float32(all_uvd), 'extra_uvd': np.float32(extra_uvd), 'weight_aug': np.float32(w_aug)}
            else:
                inputs = {'img': np.float32(img)}
                targets = {'mesh_pose_uvd': np.float32(all_uvd)}

            meta_info = {}
        else:
            # K, scale = self.anno_all[idx]
            # K, scale = [np.array(x) for x in [K, scale]]
            K = self.camera_intrinsics
            # if self.extra:
            #     extra_hm = np.zeros((1, 1, 64, 64), dtype=np.float32)
            #     inputs = {'img': np.float32(img), 'extra': extra_hm}
            # else:
            inputs = {'img': np.float32(img)}
            targets = {}
            meta_info = {
                'img2bb_trans': np.float32(img2bb_trans),
                'bb2img_trans': np.float32(bb2img_trans),
                'K': np.float32(K),
                'root_depth': np.float32(handJoints3D[2])
                }

        return inputs, targets, meta_info


    def load_objects(self, obj_root):
        object_names = ['juice', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
            all_models[obj_name] = {
                'corners': corners
            }
        return all_models

    def get_object_pose(self, sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def get_box_3d_control_points(self, corners):

        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)

        edge_01 = (corners[0] + corners[1]) / 2.
        edge_12 = (corners[1] + corners[2]) / 2.
        edge_23 = (corners[2] + corners[3]) / 2.
        edge_30 = (corners[3] + corners[0]) / 2.
        edge_45 = (corners[4] + corners[5]) / 2.
        edge_56 = (corners[5] + corners[6]) / 2.
        edge_67 = (corners[6] + corners[7]) / 2.
        edge_74 = (corners[7] + corners[4]) / 2.
        edge_04 = (corners[0] + corners[4]) / 2.
        edge_15 = (corners[1] + corners[5]) / 2.
        edge_26 = (corners[2] + corners[6]) / 2.
        edge_37 = (corners[3] + corners[7]) / 2.

        center = np.mean(corners, axis=0)

        control_points = np.vstack((center, corners,
                                    edge_01, edge_12, edge_23, edge_30,
                                    edge_45, edge_56, edge_67, edge_74,
                                    edge_04, edge_15, edge_26, edge_37))

        return control_points



def get_dataset(dataset, mode):
    if dataset == 'FreiHAND':
        return FreiHAND(os.path.join('../../dataset', dataset), mode)

    if dataset == 'HO3D_v2':
        return HO3D_v2(os.path.join('../../dataset', dataset), mode)

