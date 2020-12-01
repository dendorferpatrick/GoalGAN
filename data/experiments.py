from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

import cv2

root_path = Path(os.path.realpath(__file__)).parent.parent


from itertools import permutations


class Experiment:
    """The experiment objects store mainly paths to train and testfiles as well as homography matrices"""
    def __init__(self):
        super(Experiment, self).__init__( )

        self.data_path = ''
        self.video_file = ''
        self.trajectory_file = ''
        self.static_image_file = ''
        self.obstacle_image_file = ''
        self.test_dir = ''
        self.train_dir = ''
        self.val_dir = ''
        self.name = ""
        self.H = []
        self.homography = []



        self.scaling = 0.05
        self.get_name()
        self.data_path = root_path / 'datasets' /self.name
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'


    def get_name(self):
        self.name = self.__class__.__name__

    def set_phase(self, phase):
        self.phase

    def get_file_path(self, phase):
        if phase == "test":
            self.dataDir = self.test_dir

        elif phase == "train":
            self.dataDir = self.train_dir

        elif phase == "val":
            self.dataDir = self.val_dir
        else:
            raise AssertionError('"phase" must be either train, val or test.')
        return str(self.dataDir)
    def init_args_dict(self):

        self.args_dict = {}
        for item in self.args:
            self.args_dict[item] = getattr(self, item)

    def get_dataset_args(self):
        #self.init_args_dict()
        return self.args_dict

    def plot_image(self):
        img=plt.imread(self.static_image_file)
        plt.imshow(img)

    def load_data(self): 
        print(self.trajectory_file)
        self.world_data = np.loadtxt( self.trajectory_file)
        self.world_data = self.world_data[:,( 0, 1, 2, 4)]
        return self.world_data
    def world2pixel(self):
        self.world_data =np.concatenate(( self.world_data, np.ones((1, len(self.world_data))).T), axis = 1)
        pixel_data = self.world_data*0 
        # transform coordinates to pixel space
        for i, arr in enumerate(self.world_data[:,2:5]):
            pixel_data[i, 2:5]=np.dot(self.H_inv, arr)
        pixel_data[:,2:4]=pixel_data[:, 2:4]/pixel_data[:, 4][:, None]
        self.pixel_data = pixel_data


    def warp_obstacle(self):
        self.load_data()
        self.world2pixel()

        im_src = cv2.imread(str(self.obstacle_image_file))
        stat_img = cv2.imread(str( self.static_image_file))
        print(  im_src.shape[0])


        corners= np.array( [[im_src.shape[1], im_src.shape[0]],
                            [0, im_src.shape[0]],
                            [im_src.shape[1],0],
                            [0, 0]])

        corners = np.concatenate((corners, np.ones((1, len(corners))).T), axis=1)

        corners_real = corners * 0
        self.world_data[:, 2:4]-= np.min(self.world_data[:, 2:4], axis = 0)

        h, status = cv2.findHomography(self.pixel_data[:, (3, 2)] , self.world_data[:, (3, 2)]/self.scaling)
        self.world_shifted = self.world_data
        print(corners)
        for i, arr in enumerate(corners):
            corners_real[i] = np.dot(self.H, arr)

        corners_real[:, :2] = corners_real[:,:2] / corners_real[:, 2][:, None]

        corners_real-= np.min(corners_real, axis = 0)

        self.world_obst  = cv2.warpPerspective(im_src, h, ( int(np.max(corners_real[:,0])/self.scaling), int(np.max(corners_real[:,1])/self.scaling)), borderValue = (255, 255, 255))
        self.world_stat = cv2.warpPerspective(stat_img, h, ( int(np.max(corners_real[:,0])/self.scaling), int(np.max(corners_real[:,1])/self.scaling)), borderValue = (255, 255,255))

    def save_shift(self):
        time_sorted = sorted(np.unique(self.world_shifted[:, 0]))
        min_time = time_sorted[0]
        rel_min = time_sorted[1] - time_sorted[0]
        self.world_shifted[:, 0] = (self.world_shifted[:, 0] - min_time) / rel_min
        np.savetxt(os.path.join(self.data_path, "{}.txt".format(self.name)), self.world_shifted[:, :4], fmt="%i\t%i\t%1.2f\t%1.2f")
    def save_images(self):
        cv2.imwrite(os.path.join(self.data_path, "{}_op.jpg".format(self.name)), self.world_obst)
        cv2.imwrite(os.path.join(self.data_path, "{}.jpg".format(self.name)), self.world_stat)


    def plot_points(self):
        
        self.plot_image()
        plt.scatter(self.pixel_data[:, 3], self.pixel_data[:, 2])

        plt.show() 
        #plt.scatter(self.world_data[:, 3], self.world_data[:, 2])


class stanford_synthetic(Experiment):

    def __init__(self):
        super().__init__()


        self.args_dict =  {"norm2meters" : False,
                            "data_columns":  ['frame', 'ID', 'x', 'y'],
                            "delim": "tab",
                             "img_scaling": 0.05,
                             "wall_available": True,
                            "scale": False,
                            "homography": pd.read_csv(os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"),
                            "format": "meter",
                            "norm2meters": False,
                            "framerate": 30}




class BiWi(Experiment):
    """The experiment objects store mainly paths to train and testfiles as well as homography matrices"""
    def __init__(self):
        super().__init__()
        self.delim = 'tab'
        self.args_dict = { "norm2meters" : False,
                        "data_columns" : ['frame', 'ID', 'y', 'x'] ,
                        "delim" : "tab",
                        "wall_available" : False,
                        "scale" : False,
                        "img_scaling": 0.05,
                        "format" : "meter"}


class stanford(Experiment):

    def __init__(self):
        super().__init__()


        self.args_dict =  {"norm2meters": True,
                            "data_columns":  ['ID', 'xmin, left', 'ymin, left', 'xmax, right',
                            'ymax, right', 'frame', 'lost', 'occuluded', 'generated', 'label', 'x', 'y'],
                            "delim": "tab",
                           "homography" : pd.read_csv(os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"),
                           "wall_available": False,
                           "scale": True,
                           "img_scaling": 0.05,
                           "format" : "pixel",
                           "framerate": 30}



class gofp(Experiment):

    def __init__(self):
        super().__init__()


        self.args_dict =  {"norm2meters" : True,
                             "data_columns":  ['frame', 'ID', 'x', 'y','moment','old frame', 'old_ID', 'is_active'],
                            "delim": "tab",
                             "img_scaling": 0.05,
                             "wall_available": False,
                            "scale": True,
                            "homography":{'zara1': 0.03109532180986424,
                                          'eth': 0.06668566952360758,
                                          'hotel': 0.0225936169079401,
                                          '0000': 0.042200689823829046,
                                          '0400': 0.07905284109247492,
                                          '0401': 0.0598454105469989,
                                          '0500': 0.04631904070838066,
                                          'zara2': 0.03109532180986424},

                           # "homography": {'zara1': 0.01691242906217759, 'eth': 0.03896417367158226, 'hotel': 0.01320136124236592, '0000': 0.02736160163256867, '0400': 0.0473964760375766, '0401': 0.037807370094202, '0500': 0.03003181097645976, 'zara02': 0.01691242906217759},
                            "format": "pixel",
                           "framerate": 10

                           }


#### ETH and UCY datasets

class eth(BiWi):

    def __init__(self):
        super().__init__()


        self.video_file = self.data_path / 'seq_eth.avi'
        self.trajectory_file = self.data_path / 'eth_raw.txt'
        self.static_image_file = self.data_path / 'eth_static.jpg'
        self._image_file = self.data_path / 'eth_static.jpg'
        self.obstacle_image_file = self.data_path / 'annotations.jpg'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'


        self.H = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                           [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                           [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]])
        self.H_inv = inv(self.H)
        


class hotel(BiWi):

    def __init__(self):
        super().__init__()


        self.video_file = self.data_path / 'seq_hotel.avi'
        self.trajectory_file = self.data_path / 'hotel_raw.txt'
        self.static_image_file = self.data_path / 'hotel_static.jpg'
        self.obstacle_image_file = self.data_path / 'annotations.jpg'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'



        self.H = np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                           [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                           [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]])
        self.H_inv = inv(self.H)

class univ(BiWi):

    def __init__(self):
        super().__init__()


        self.video_file = self.data_path / 'students001.avi'
        self.trajectory_file = self.data_path / 'univ_raw.txt'
        self.static_image_file = self.data_path / 'univ_static.jpg'
        self.obstacle_image_file = self.data_path / 'annotations.jpg'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'


        self.H = np.array([[0.032529736503653,  -0.000730604859308 , -7.969749046103707],
                            [0.000883577230612,   0.026589331317173,  -8.754694531864281],
                            [0.001039809003515 ,  0.000025010101498 ,  1.007920696981254]])

        """
        self.H = np.array([[ 2.44207875e+01,  4.85423732e-01,  1.97314417e+02],
       [-9.03279503e+00,  3.71247434e+01,  2.51038281e+02],
       [-2.49692696e-02, -1.42197851e-03,  7.82355400e-01]])
         """
        #pts_img = np.array([[117, 476], [ 117, 562], [ 311, 562],[311, 476]])
        #pts_wrd = np.array([[10.2, 11], [12, 11 ],  [12, 8 ], [10.2,8 ]])
      
        #self.H, status = cv2.findHomography( pts_img, pts_wrd)
        self.H_inv = inv(self.H)
class zara1(BiWi):

    def __init__(self):
        super().__init__()

        self.video_file = self.data_path / 'crowds_zara01.avi'
        self.trajectory_file = self.data_path / 'zara1_raw.txt'
        self.static_image_file = self.data_path / 'zara_static.jpg'
        self.obstacle_image_file = self.data_path / 'annotations.jpg'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'

        """
        self.H = np.array([[-2.5956517e-02, -5.1572804e-18, 7.8388681e+00],
                           [-1.0953874e-03, 2.1664330e-02, -1.0032272e+01],
                           [1.9540125e-20, 4.2171410e-19, 1.0000000e+00]])
        """

        self.H = np.array([[-2.59600906e-02, -4.14338866e-07,  7.83994785e+00],
                            [-1.08705701e-03,  2.16676796e-02,  5.56418836e+00],
                            [ 6.05674393e-07, -8.00267888e-08,  1.00000000e+00]])


        """
        self.H= np.array([[2.16676796e-02, -1.08705701e-03, 5.56418836e+00],
               [-4.14338866e-07, -2.59600906e-02, 7.83994785e+00],
               [-8.00267888e-08, 6.05674393e-07, 1.00000000e+00]])
        """
        #pts_img = np.array([[117, 476], [117, 562], [311, 562], [311, 476]])
        #pts_wrd = np.array([[10.2, 11], [12, 11], [12, 8], [10.2, 8]])

        #self.H, status = cv2.findHomography(pts_img, pts_wrd)
        self.H_inv = inv(self.H)
       
class zara2(BiWi):

    def __init__(self):
        super().__init__()


        self.video_file = self.data_path / 'crowds_zara02.avi'
        self.trajectory_file = self.data_path / 'zara2_raw.txt'
        self.static_image_file = self.data_path / 'zara_static.jpg'
        self.obstacle_image_file = self.data_path / 'annotations.jpg'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'



        self.H = np.array([[-2.5956517e-02, -5.1572804e-18, 7.8388681e+00],
                           [-1.0953874e-03, 2.1664330e-02, -1.0032272e+01],
                           [1.9540125e-20, 4.2171410e-19, 1.0000000e+00]])


        #pts_img = np.array([[117, 476], [ 117, 562], [ 311, 562],[311, 476]])
        #pts_wrd = np.array([[10.2, 11], [12, 11 ],  [12, 8 ], [10.2,8 ]])
      
        #self.H, status = cv2.findHomography( pts_img, pts_wrd)
        self.H_inv = inv(self.H)
class Videodata:

    def __init__(self, experiment):
        self.homography = experiment.H
        self.video = cv2.VideoCapture(str(experiment.video_file))
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_file(self, _path, delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    def camcoordinates(self, xy):
        """Transform the meter coordinates with the homography matrix"""
        coords = xy.reshape(1, 1, -1)
        return cv2.perspectiveTransform(coords, np.linalg.inv(self.homography)).squeeze()[::-1]

    def getFrame(self, fid):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, fid)
        return self.video.read()[1]

    def staticImage(self):
        ret = True
        image = np.zeros((self.frame_height, self.frame_width, 3))
        while (ret):
            ret, img = self.video.read()
            if not ret:
                break
            image += img
        image /= self.frame_count
        image = image.astype('uint8')
        return image
