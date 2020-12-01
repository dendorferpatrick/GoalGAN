import os, sys

sys.path.append(os.getcwd())

import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import copy
import math
import logging
import data.experiments
from utils.utils import re_im, image_json
import torchvision.transforms.functional as TF
from torchvision import transforms
import random


# Helper functions for creating trajectory dataset
def rotate(X, center, alpha):
    XX = torch.zeros_like(X)

    XX[:, 0] = (X[:, 0] - center[0]) * np.cos(alpha) + (X[:, 1] - center[1]) * np.sin(alpha) + center[0]
    XX[:, 1] = - (X[:, 0] - center[0]) * np.sin(alpha) + (X[:, 1] - center[1]) * np.cos(alpha) + center[1]

    return XX

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
def flatten(l):
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def seq_collate(data):
    obs_traj_list, pred_traj_list, obs_traj_rel_list, pred_traj_rel_list, scene_img_list, features_list, prob_mask_list, wall_list, features_tiny_list = zip(
        *data)

    _len = [len(seq) for seq in obs_traj_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]


    obs_traj = torch.cat(obs_traj_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_traj_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_traj_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_traj_rel_list, dim=0).permute(1, 0, 2)

    scene_img_list = tuple(flatten(list(scene_img_list)))
    try:
        global_patch = torch.cat(features_list, dim=0)
        local_patch = torch.cat(features_tiny_list, dim=0)
        prob_mask = torch.cat(prob_mask_list, dim=0)
    except:
        global_patch = torch.empty(1)
        local_patch = torch.empty(1)
        prob_mask = torch.empty(1)

    return {"in_xy": obs_traj,
            "gt_xy": pred_traj,
            "in_dxdy": obs_traj_rel,
            "gt_dxdy": pred_traj_rel,
            "size": torch.LongTensor([obs_traj.size(1)]),
            "scene_img": scene_img_list,
            "global_patch": global_patch,
            "prob_mask": prob_mask,
            "occupancy": wall_list,
            "local_patch": local_patch,
            "seq_start_end": seq_start_end
            }


logger = logging.getLogger(__name__)
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""


    def __init__(self,
                 save=False,
                 load_p=True,
                 dataset_name="stanford_synthetic",
                 phase="test",
                 scene_batching = False,
                 obs_len=8,
                 pred_len=12,
                 time_step=0.4,
                 skip=1,
                 data_augmentation=0,
                 scale_img=True,
                 cnn=True,
                 max_num=None,
                 load_semantic_map=False,
                 logger=logger,
                 special_scene=None,
                 scaling_global=1,
                 scaling_local=0.25,
                 grid_size_in_global=32,
                 grid_size_out_global=16,
                 grid_size_local=8,
                 debug=False,
                 **kwargs
                 ):
        super().__init__()
        self.__dict__.update(locals())

        if phase!= "train":
            self.data_augmentation = 0
            logger.info("Data Augmentation only for 'test' data")


        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - delim: Delimiter in the dataset files
        """


        self.prepare()

    def get_save_path(self):
        path = ""
        for name, value in self.save_dict.items():
            if type(value) in [int, float, str]:
                path += "{}_{}_".format(name, value)
        path += ".p"
        self.save_path = path

    def print_wall(self, wall_points):
        """
        :param wall_points: list of wall elements
        :return: plots wall in image
        """
        for wall in wall_points:
            for index in range(1, len(wall) + 2):
                ind = [index % len(wall), (index + 1) % len(wall)]

                pos = wall[ind, :] / self.img_scaling
                plt.plot(pos[:, 0], pos[:, 1], color='r')

    def load_image(self, _path, scene, ):
        img = Image.open(_path)
        if "stanford" in self.dataset_name:

            ratio = self.homography.loc[((self.homography["File"] == "{}.jpg".format(scene)) & (
                        self.homography["Version"] == "A")), "Ratio"].iloc[0]

            scale_factor = ratio / self.img_scaling

            old_width = img.size[0]
            old_height = img.size[1]

            new_width = int(round(old_width * scale_factor))
            new_height = int(round(old_height * scale_factor))

            scaled_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        elif "gofp" in self.dataset_name:

            ratio = self.homography[scene]

            scale_factor = ratio / self.img_scaling

            old_width = img.size[0]
            old_height = img.size[1]

            new_width = int(round(old_width * scale_factor))
            new_height = int(round(old_height * scale_factor))

            scaled_img = img.resize((new_width, new_height), Image.ANTIALIAS)

        else:

            scaled_img = img

            scale_factor = 1
            ratio = 1.

        width = scaled_img.size[0]
        height = scaled_img.size[1]

        scale_factor_global = self.img_scaling / self.scaling_global
        global_width = int(round(width * scale_factor_global))
        global_height = int(round(height * scale_factor_global))
        global_image = scaled_img.resize((global_width, global_height), Image.ANTIALIAS)

        scale_factor_local = self.img_scaling / self.scaling_local
        local_width = int(round(width * scale_factor_local))
        local_height = int(round(height * scale_factor_local))
        local_image = scaled_img.resize((local_width, local_height), Image.ANTIALIAS)

        self.images.update({scene: {"ratio": ratio, "scale_factor": scale_factor, "scaled_image": scaled_img,
                                    "global_image": global_image, "local_image": local_image}})
        if self.wall_available:
            # load walls in meter
            walls, wall_points, _, _, _ = image_json(scene, self.data_dir, scaling=ratio)
            self.wall_points_dict[scene] = [np.stack(wall_i) for wall_i in wall_points]

    def get_ratio(self, scene):
        return self.images[scene]["ratio"]

    def get_stats(self):

        self.logger.info("Number of trajectories: {}".format(self.num_seq))

        max_dist_obs, _ = torch.abs(self.obs_traj - self.obs_traj[:, -1].unsqueeze(1)).view(-1, 2).max(0)
        max_dist_pred, _ = torch.abs(self.pred_traj - self.obs_traj[:, -1].unsqueeze(1)).view(-1, 2).max(0)

        range_goal = self.grid_size_in_global * self.scaling_global

        self.logger.info(
            "Max Dist Obs: {}, Max Dist Pred: {}, Grid Size: {} ".format(max_dist_obs, max_dist_pred, range_goal))

        if (range_goal < max_dist_obs).any():
            self.logger.warning("Image Patch does not fit all Observations")
        if (range_goal < max_dist_pred).any():
            self.logger.warning("Image Patch does not fit all Predictions")
        max_dx_obs, _ = torch.abs(self.obs_traj_rel).view(-1, 2).max(0)
        max_dx_pred, _ = torch.abs(self.pred_traj_rel).view(-1, 2).max(0)

        range_goal_local = self.grid_size_local * self.scaling_local

        if (range_goal_local < max_dx_obs).any():
            self.logger.warning("Goal Local Image Patch does not fit all Observations")
        if (range_goal_local < max_dx_pred).any():
            self.logger.warning("Goal Local Image Patch does not fit all Predictions")

        self.logger.info(
            "Max dx Obs: {}, Max dx Pred: {}, Grid Size: {}".format(max_dx_obs, max_dx_pred, range_goal_local))

    def scale2meters(self):
        self.obs_traj *= self.img_scaling
        self.pred_traj *= self.img_scaling
        self.obs_traj_rel *= self.img_scaling
        self.pred_traj_rel *= self.img_scaling
        self.trajectory *= self.img_scaling
        self.format = "meter"

    def load_file(self, _path, delim="tab"):
        if delim == 'tab':
            delim = "\t"
        elif delim == 'space':
            delim = ' '

        df = pd.read_csv(_path, header=None, delimiter=delim, dtype = None)
        df.columns = self.data_columns

        if "label" and "lost" in df:

            data_settings = {"label": "Pedestrian",
                             "lost": 0}

            for name, item in data_settings.items():
                df = df[df[name] == item]

        if self.dataset_name in ["stanford", "gofp"]:

            df = df[df["frame"] % int(round(self.framerate * self.time_step)) == 0]
            df["frame"] /= int(round(self.framerate * self.time_step))

        columns_experiment = ['frame', 'ID', 'x', 'y']
        df = df[columns_experiment]


        return np.asarray(df.values)

    def __len__(self):
        
        return self.num_seq
    """ ########## VISUALIZE ########## """
    def plot(self, index, modes=["in", "gt"], image_type="scaled", final_mask=False):

        out = self.get_scene(index)

        if "seq_start_end" in out:
            (index, end) = out["seq_start_end"]

        image = out["scene_img"][0]

        if image_type == "orig":
            img_label = "img"
            img = image[img_label]
            scale = 1
        elif image_type == "scaled":
            img_label = "scaled_image"
            img = image[img_label]
            if self.format == "meter":
                scale = 1. / self.img_scaling
            else:
                scale = 1
        elif image_type == "global":
            img_label = "global_image"
            img = image[img_label]
            scale = 1. / self.scaling_global

        elif image_type == "local":
            img_label = "local_image"
            img = image[img_label]
            scale = 1. / self.scaling_local
        elif image_type == "patch":
            img = re_im(out["global_patch"][0, :3].permute(1, 2, 0))
            scale = 1. / self.scaling_global

        else:
            assert False, "'{}' not valid <image_type>".format(image_type)

        center = out["in_xy"][-1, 0] * scale




        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)

        if final_mask:

            rel_scaling = (2 * self.grid_size_in_global + 1) / (2 * self.grid_size_out_global + 1)

            mask = out["prob_mask"][0, 0]

            final_pos_out = torch.cat(torch.where(mask == 1)).float()

            final_pos = torch.zeros_like(final_pos_out)
            final_pos[0] = final_pos_out[1]
            final_pos[1] = final_pos_out[0]
            center_pixel_global = out["in_xy"][-1, 0] * scale
            if not image_type == "patch":

                final_pos_pixel_centered = (final_pos - self.grid_size_out_global) * self.scaling_global * scale * rel_scaling

                final_pos_pixel = final_pos_pixel_centered + center_pixel_global



            elif image_type == "patch":
                final_pos_pixel = final_pos * rel_scaling
                mask = mask.numpy()
                mask = (mask * 255).astype(np.uint8)

                mask = Image.fromarray(mask, mode="L")
                mask = mask.resize((int(2 * self.grid_size_in_global + 1), int(2 * self.grid_size_in_global + 1)))
                ax.imshow(mask, alpha=0.5)

            plt.plot(final_pos_pixel[0], final_pos_pixel[1], "x")
            error = abs(final_pos_pixel / scale - out["gt_xy"][-1])
            error_bound = rel_scaling * self.scaling_global
            print("Error real and approximation: {} Threshold: {} ".format(error, error_bound))
            # assert (error < error_bound).all(), "Error above error bound"

        for m in modes:
            if m == "gt":
                marker = '-'
            else:
                marker = '-'

            traj = out["{}_xy".format(m)][:, 0] * scale
            traj = traj.cpu().numpy()

            if image_type == "patch":
                traj = traj + (self.grid_size_in_global) - center.cpu().numpy()

            ax.plot((traj[:, 0]).astype(int), (traj[:, 1]).astype(int), linestyle=marker, linewidth=int(3))


        plt.show()

        def trim_axs(axs, N):
            """little helper to massage the axs list to have correct length..."""
            axs = axs.flat
            for ax in axs[N:]:
                ax.remove()
            return axs[:N]

    def gen_global_patches(self, scene_image, trajectory, prediction, image_type="global_image"):
        if self.format == "meter":
            scale = 1. / self.scaling_global

        else:
            scale = 1

        rel_scaling = (2 * self.grid_size_in_global + 1) / (2 * self.grid_size_out_global + 1)

        img = scene_image[image_type]

        center_meter = trajectory[-1]  # center in meter

        end_dist_meters = prediction[-1] - center_meter
        end_point_pixel_global = scale * end_dist_meters

        center_pixel_global = center_meter * scale

        center_scaled = center_pixel_global.long()

        x_center, y_center = center_scaled

        cropped_img = TF.crop(img, int(y_center - self.grid_size_in_global), int(x_center - self.grid_size_in_global),
             int(2*  self.grid_size_in_global + 1), int(2* self.grid_size_in_global + 1))

        end_point = end_point_pixel_global / rel_scaling + self.grid_size_out_global

        x_end, y_end = np.clip(int(end_point[0]), 0, 2 * self.grid_size_out_global), np.clip(int(end_point[1]), 0,
                                                                                             2 * self.grid_size_out_global)

        prob_mask = torch.zeros((1, 1, self.grid_size_out_global * 2 + 1, self.grid_size_out_global * 2 + 1)).float()
        prob_mask[0, 0, y_end, x_end] = 1

        position = torch.zeros(1, self.grid_size_in_global * 2 + 1, self.grid_size_in_global * 2 + 1)
        position[0, self.grid_size_in_global, self.grid_size_in_global] = 1

        img = -1 + transforms.ToTensor()(cropped_img)  * 2.

        img = torch.cat((img, position), dim=0).unsqueeze(0)



        return img, prob_mask

    def gen_local_patches(self, scene_image, trajectory, prediction, image_type="local_image"):
        if self.format == "meter":
            scale = 1. / self.scaling_local

        else:
            scale = 1

        img = scene_image[image_type]

        center_meter = trajectory  # center in meter

        end_dist_meters = prediction - center_meter
        end_point_pixel_global = scale * end_dist_meters

        center_pixel_global = center_meter * scale

        center_scaled = center_pixel_global.long()

        x_center, y_center = center_scaled

        cropped_img = TF.crop(img,  int(y_center - self.grid_size_local),int(x_center - self.grid_size_local),
             int(2 *  self.grid_size_local + 1), int(2* self.grid_size_local + 1))

        end_point = end_point_pixel_global + self.grid_size_local

        x_end, y_end = np.clip(int(end_point[0]), 0, 2 * self.grid_size_local), np.clip(int(end_point[1]), 0,
                                                                                        2 * self.grid_size_local)

        prob_mask = torch.zeros((1, self.grid_size_local * 2 + 1, self.grid_size_local * 2 + 1, 1))
        prob_mask[0, y_end, x_end, 0] = 1.

        position = torch.zeros((1, self.grid_size_local * 2 + 1, self.grid_size_local * 2 + 1))
        position[0, self.grid_size_local, self.grid_size_local] = 1.


        img = -1 + transforms.ToTensor()(cropped_img) * 2.
        img = torch.cat((img.float(), position), dim=0).unsqueeze(0)

        return img

    def load_dset(self):

        pickle_path = os.path.join(self.data_dir, self.save_path)
        if os.path.isfile(pickle_path) and self.load_p:

            data = torch.load(pickle_path, map_location='cpu')

            self.__dict__.update(data)

            self.logger.info("data loaded to {}".format(pickle_path))
        else:
            self.load()

    def save_dset(self):
        pickle_path = os.path.join(self.data_dir, self.save_path)
        if not os.path.isfile(pickle_path):

            data_save = {}

            os.path.join(self.data_dir, self.save_path)

            for name, value in self.__dict__.items():

                try:

                    data_save.update({name: value})
                    torch.save(data_save, pickle_path)


                except:
                    data_save.pop(name)

            self.logger.info("data saved to {}".format(pickle_path))
    def prepare(self):
        self.save_dict = copy.copy(self.__dict__)

        self.get_save_path()

        self.dataset = getattr(data.experiments, self.dataset_name)()

        self.__dict__.update(self.dataset.get_dataset_args())
        self.data_dir = self.dataset.get_file_path(self.phase)
        self.seq_len = self.obs_len + self.pred_len
        # do not use any timestep twice
        all_files = os.listdir(self.data_dir)
        self.all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        if self.debug:
            self.max_num = 10
        self.load_dset()


    def load(self):

        scene_nr = []
        self.scene_list = []
        self.images = {}
        self.image_list = []
        self.wall_points_dict = {}
        self.walls_list = []
        seq_list = []


        collect_data = True
        for path in [file for file in self.all_files if ".jpg" in file]:

            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]

            img_parts = scene.split("-")

            if self.load_semantic_map and img_parts[-1] == "op":

                scene = img_parts[-2]
                self.load_image(path, scene)

            elif not self.load_semantic_map and img_parts[-1] != "op":
                self.load_image(path, scene)

                continue

        if len(self.images) == 0:
            assert False, "No valid imges in folder"
        num_peds_in_seq = []
        for path in [file for file in self.all_files if ".txt" in file]:
            if not collect_data:
                break
            if self.special_scene and not self.special_scene in path:
                continue

            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]

            if data_type == "txt":



                scene = '_'.join(scene.split("_")[1:])

                self.logger.info("preparing %s" % scene)
                data = self.load_file(path, self.delim)


                frames = np.unique(data[:, 0]).tolist()
                if self.debug:
                    frames = frames[:30]
                frame_data = []
         
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])



                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len) / self.skip))

                for idx in range(0, num_sequences * self.skip , self.skip):
                    if self.max_num and (len(scene_nr) > self.max_num if self.max_num else True):
                        break
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    num_peds = 0
                    peds_scene = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                     ped_id, :]


                        if (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] !=1 ).any() or (len(curr_ped_seq) != self.seq_len):
                            continue

                        num_peds+= 1

                        peds_scene.append(curr_ped_seq[:, 2:])


                    if num_peds > 0:
                        num_peds_in_seq.append(num_peds)

                        seq_list.append( np.stack((peds_scene), axis = 0 ))
                        if not self.scene_batching:
                            self.scene_list.extend(num_peds*[scene])
                        else:
                            self.scene_list.append(scene)

                        if self.wall_available:
                            self.walls_list.append([item.copy() for item in self.wall_points_dict[scene]])
                        if not self.data_augmentation:


                            img = self.images[scene]["scaled_image"]
                            small_image = self.images[scene]["global_image"]
                            tiny_image = self.images[scene]["local_image"]

                            self.image_list.append(
                                {"ratio": self.images[scene]["ratio"], "scene": scene, "scaled_image": copy.copy(img),
                                 "global_image": copy.copy(small_image), "local_image": copy.copy(tiny_image)})

                        scene_nr.append(1)

        seq_list = np.concatenate(seq_list, axis=0)


        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()

        if self.scene_batching:

            self.seq_start_end = [
                    (start, end)
                    for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        else:
            self.seq_start_end = [(i, i+1) for i in range(sum(num_peds_in_seq))]


        seq_list_rel = seq_list[:, 1:] - seq_list[:, :-1]

        self.trajectory = seq_list


        self.num_seq = len(self.seq_start_end)
        # Convert numpy -> Torch Tensor
        self.scene_nr = torch.LongTensor(np.cumsum(scene_nr))


     
        self.obs_traj = torch.from_numpy(
            seq_list[:, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :self.obs_len - 1]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, self.obs_len - 1:]).type(torch.float)

        del seq_list
        del seq_list_rel

        if self.scale:
            self.scale_func()
        if self.norm2meters:
            self.scale2meters()

        if self.cnn and not self.data_augmentation:
            self.get_global_patches()
            self.get_local_patches()
        if self.save:
            self.save_dset()
        self.get_stats()


    def print_wall(self, wall_points):

        for wall in wall_points:
            for index in range(1, len(wall) + 2):
                ind = [index % len(wall), (index + 1) % len(wall)]

                pos = wall[ind, :] / self.img_scaling
                plt.plot(pos[:, 0], pos[:, 1], color='r')
    #TODO: FixLoading



    def get_global_patches(self):
        feature_list = []
        cropped_img_list = []
        prob_mask_list = []
        for idx, (start, end) in enumerate(self.seq_start_end):
            scene_image = self.image_list[idx]
            for index in np.arange(start, end):

                features, prob_mask = self.gen_global_patches(scene_image, self.obs_traj[index],
                                                                            self.pred_traj[index])
                feature_list.append(features)

                prob_mask_list.append(prob_mask)

        self.global_patches = torch.cat(feature_list)
        self.Prob_Mask = torch.cat(prob_mask_list)



    def get_local_patches(self):
        feature_list = []
        for idx, (start, end) in enumerate(self.seq_start_end):
            scene_image = self.image_list[idx]
            for index in np.arange(start, end):
                trajectories = torch.cat((self.obs_traj[index], self.pred_traj[index]), dim=0)
                feature_pred = []

                for time in range(self.pred_len):

                    feature_pred.append(self.gen_local_patches(scene_image, trajectories[self.obs_len - 1 + time],
                                                                trajectories[time + self.obs_len]))
                feature_pred = torch.cat(feature_pred)
                feature_list.append(feature_pred)
        self.local_patches = torch.stack(feature_list)

    def get_scene(self, index):

        in_xy, gt_xy, in_dxdy, gt_dxdy, scene_img, features, prob_mask, walls, features_tiny = self.__getitem__(index)

        return {"in_xy":    in_xy.permute(1, 0, 2),
                "gt_xy" :   gt_xy.permute(1, 0, 2),
                "in_dxdy":  in_dxdy.permute(1, 0, 2),
                "gt_dxdy":  gt_dxdy.permute(1, 0, 2),
                "scene_img": scene_img,
                "global_patch": features,
                "prob_mask": prob_mask,
                "local_patch": features_tiny,
                "seq_start_end": [0, in_xy.size(0)]
                }

    def scale_func(self):

        for index in np.arange(self.num_seq):
            start, end = self.seq_start_end[index]
            scene = self.scene_list[index]
            ratio = self.images[scene]["scale_factor"]
            self.obs_traj[start:end] *= ratio
            self.pred_traj[start:end] *= ratio
            self.obs_traj_rel[start:end] *= ratio
            self.pred_traj_rel[start:end] *= ratio
            self.trajectory[start:end] *= ratio



    def data_aug_func(self, scene = None,
                            current_scene_image = None,
                            obs_traj = None,
                            pred_traj = None,
                            obs_traj_rel = None,
                            pred_traj_rel = None,
                            walls = None):

        # seed = torch.utils.data.get_worker_info().seed
        # print("Working seed {}".format( seed))
        # torch.manual_seed(seed)

        img = current_scene_image["scaled_image"]


        k = obs_traj.size(0)
        xy = torch.cat((obs_traj, pred_traj), dim= 1)


        if self.format == "pixel":
            scale2orig = 1 / current_scene_image["scale_factor"]
        elif self.format == "meter":
            scale2orig = self.img_scaling
        else:
            assert False, " Not valid format '{}': 'meters' or 'pixel'".format(self.format)
        alpha = torch.rand(1) * 2 * np.pi

        center = torch.tensor(img.size) / 2.
        corners = torch.tensor([[0, 0], [0, img.height], [img.width, img.height], [img.width, 0]])

        rand_num = torch.randint(0, 3, (1,))

        if rand_num != 0:
            if rand_num == 1:

                img = TF.hflip(img)
                xy[:,:,  0] = img.width * scale2orig - xy[:, :, 0]

                if self.wall_available:
                    for i in range(len(walls)):
                        walls[i][:, 0] = img.width * scale2orig - walls[i][:, 0]

            elif rand_num == 2:
                img = TF.vflip(img)
                xy[ :, :, 1] = img.height * scale2orig - xy[ :, :, 1]
                # transform wall
                if self.wall_available:
                    for i in range(len(walls)):
                        walls[i][:, 1] = img.height * scale2orig - walls[i][:, 1]


        img = TF.rotate(img,alpha / np.pi * 180 , expand = True)
        corners_trans = rotate(corners, center, alpha)
        offset = corners_trans.min(axis=0)[0]
        corners_trans -= offset


        if self.wall_available:
            wall_points = []
            for wall in walls.copy():
                wall_points.append(
                    (rotate( torch.from_numpy(wall), center * scale2orig, alpha) - offset * scale2orig).numpy())
            walls = wall_points

        else:
            walls = False

        xy = xy.reshape(k * self.seq_len, -1)
        xy = rotate(xy, center * scale2orig, alpha) - offset * scale2orig

        xy = xy.reshape(k, self.seq_len, -1)

        scaling_factor_global = self.img_scaling / self.scaling_global
        scaling_factor_local = self.img_scaling / self.scaling_local

        global_image = img.resize((int(round(img.width * scaling_factor_global)),
                                  int(round(img.height * scaling_factor_global))),
                                 Image.ANTIALIAS)
        local_image = img.resize((int(round(img.width * scaling_factor_local)),
                                 int(round(img.height * scaling_factor_local))),
                                Image.ANTIALIAS)

        scene_image = {"ratio": self.images[scene]["ratio"], "scene": scene, "scaled_image": copy.copy(img),
                                "global_image": copy.copy(global_image), "local_image": copy.copy(local_image)}


        dxdy = xy[:,  1:] - xy[:, :-1]
        obs_traj = xy[:, :self.obs_len]
        pred_traj = xy[:, self.obs_len:]

        feature_list = []
        cropped_img_list = []
        prob_mask_list = []



        for idx in np.arange(0,  k ):
            features_id,  prob_mask = self.gen_global_patches(scene_image, obs_traj[idx],pred_traj[idx])

            feature_list.append(features_id)
            prob_mask_list.append(prob_mask)

        global_patch = torch.cat(feature_list)
        prob_mask_tensor = torch.cat(prob_mask_list)
        feature_list = []
        for idx in np.arange(0, k):
            feature_pred = []
            for time in range(self.pred_len):

                feature_pred.append(self.gen_local_patches(scene_image, xy[idx, self.obs_len - 1 + time],
                                                            xy[idx, time + self.obs_len]))
            feature_pred = torch.cat(feature_pred)
            feature_list.append(feature_pred)

        local_patch = torch.stack(feature_list, dim = 0 )


        scene_image = (k) * [scene_image]



        return obs_traj,pred_traj, dxdy[:, :self.obs_len-1], dxdy[:, self.obs_len-1:], scene_image, global_patch, prob_mask_tensor, walls, local_patch




    def __getitem__(self, index):


        start, end = self.seq_start_end[index]
        obs_traj = self.obs_traj[start:end]
        pred_traj  = self.pred_traj[start:end]
        obs_traj_rel = self.obs_traj_rel[start:end]
        pred_traj_rel = self.pred_traj_rel[start:end]

        scene = self.scene_list[index]
        current_scene_image =  self.images[scene]

        if self.wall_available:

            walls = self.walls_list[index]

        else:
            walls = False
        
        if self.data_augmentation:
            obs_traj,pred_traj,obs_traj_rel, pred_traj_rel,  \
            scene_img, global_patch, prob_mask, walls, local_patch = self.data_aug_func(scene = scene,
                                                                                             current_scene_image = current_scene_image,
                                                                                              obs_traj = obs_traj,
                                                                                              pred_traj = pred_traj,
                                                                                              obs_traj_rel = obs_traj_rel,
                                                                                              pred_traj_rel = pred_traj_rel,
                                                                                              walls = walls)
        else:

            scene_img =  [self.image_list[index]]

            if self.wall_available:

                walls = self.walls_list[index]

            else:
                walls = False


            if self.cnn:
                global_patch = self.global_patches[start:end]
                local_patch = self.local_patches[start:end]
                prob_mask = self.Prob_Mask[start:end]


            else:
                global_patch = torch.empty(1)
                local_patch = torch.empty(1)
                prob_mask =  torch.empty(1)
        return [obs_traj,
                pred_traj,
                obs_traj_rel,
                pred_traj_rel,
                scene_img,
                global_patch,
                prob_mask,
                walls,
                local_patch
                ]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = TrajectoryDataset(dataset_name="stanford_synthetic", special_scene=None,  phase="test", obs_len=8,
                                pred_len=12, data_augmentation= 1, scaling_global=0.5, scaling_local=0.25, skip=20,
                                 load_semantic_map=False, cnn = True, max_num = 20, scene_batching = True,
                                grid_size_in_global=16, grid_size_out_global=16, grid_size_local=16, logger=logger)
    # print("Dataset {}: len(X) = {}".format("stanford", len(dataset.obs_traj)))
    # print(dataset.obs_traj.size() )
    # print(dataset.obs_traj[ 0])
    #
    # loader = DataLoader(
    #     dataset,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=seq_collate
    # )
    #
    # batch = next(iter(loader))
    # print(batch)
    # #batch = dataset.__getitem__(7)
    # """
    # ing = (1 + batch["features_tiny"][5, 0, 3])/2.
    # plt.imshow(ing)
    # plt.show()
    # plt.imshow(batch["scene_img"][0]["scaled_image"])
    # plt.show()
    # """
    # for i in np.arange(10, 12):
    #     dataset.plot(i, image_type="patch", final_mask=True)

    # import matplotlib.pyplot as plt

    # print(batch["features"][0].permute(1, 2, 0).size())
    # plt.imshow(((1 + batch["features"][0].permute(1, 2, 0)) / 2.))
    # plt.show()
    # print("test")
    # dataset.save_dset()


