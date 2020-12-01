import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import torch.nn as nn
import itertools
import importlib
from attrdict import AttrDict
from argparse import Namespace
import os.path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import json
import random
import os



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def distanceP2W(point, wall):
    p0 = np.array([wall[0], wall[1]])
    p1 = np.array([wall[2], wall[3]])

    d = p1 - p0
    ymp0 = point - p0
    t = np.dot(d, ymp0) / np.dot(d, d)
    if t > 0.0 and t < 1.:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point)

    else:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point) * 0

    return dist, npw


def image_json(scene, json_path, scaling=1):
    json_path = os.path.join(json_path, "{}_seg.json".format(scene))

    wall_labels = ["lawn", "building", "car", "roundabout"]

    walls = []
    wall_points = []

    start_end_points = {}
    decisionZone = {}
    directionZone = {}


    nr_start_end = 0

    with open(json_path) as json_file:

        data = json.load(json_file)
        for p in data["shapes"]:
            label = p["label"]

            if label in wall_labels:

                points = np.array(p["points"]).astype(int)

                points = order_clockwise(points)
                for i in np.arange(len(points)):
                    j = (i + 1) % len(points)

                    p1 = points[i]
                    p2 = points[j]

                    concat = np.concatenate((p1, p2))
                    walls.append(scaling * concat)

                wall_points.append([p * scaling for p in points])
            elif "StartEndZone" in label:
                id = int(label.split("_")[-1])
                start_end_points[nr_start_end] = {"point": scaling * np.array(p["points"]),
                                                  "id": id}
                nr_start_end += 1
            elif "decisionZone" in label:
                id = int(label.split("_")[-1])
                decisionZone[id] = scaling * np.array(p["points"])

            elif "directionZone" in label:
                id = int(label.split("_")[-1])
                directionZone[id] = scaling * np.array(p["points"])

    return walls, wall_points, start_end_points, decisionZone, directionZone


# order points clockwise

def order_clockwise(point_array, orientation=np.array([1, 0])):
    center = np.mean(point_array, axis=0)
    directions = point_array - center

    angles = []
    for d in directions:
        t = np.arctan2(d[1], d[0])
        angles.append(t)
    point_array = [x for _, x in sorted(zip(angles, point_array))]

    return point_array


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            break

    return random_point


def get_batch_k( batch, k):
    new_batch = {}
    for name, data in batch.items():

        if name in ["global_patch", "prob_mask"]:

            new_batch[name] = data.repeat(k, 1, 1, 1).clone()
        elif name in ["local_patch"]:

            new_batch[name] = data.repeat(k, 1, 1, 1, 1).clone()
        elif name in ["scene_img", "occupancy", "walls"]:

            new_batch[name] = data * k
        elif name not in ["size", "scene_nr", "scene", "img", "cropped_img", "seq_start_end"]:
            new_batch[name] = data.repeat(1, k, 1).clone()


        else:
            new_batch[name] = data
    return new_batch



def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed



def re_im(img):
    """ Rescale images """
    img = (img + 1)/2.
    return img
