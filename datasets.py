import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import json
import cv2
import os
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self):
        self.base_path = '/home/nianyli/Desktop/code/thesis/experiment1'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            self.label_data = json.load(file)['data']
        
        # get the upper and lower bounds of the distance labels and normalize
        # between 0 and 1
        dist_min = 1000000
        dist_max = -1000000
        
        for data in self.label_data:
            if data['distance'] < dist_min:
                dist_min = data['distance']
            if data['distance'] > dist_max:
                dist_max = data['distance']
        
        for idx, data in enumerate(self.label_data):
            data['distance'] = (abs(data['distance'] - dist_min) / (dist_max-dist_min)) * 2 - 1
        
        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # max = -1
        # min = 1
        # for data in self.label_data:
        #     if data['distance'] > max:
        #         max = data['distance']
        #     if data['distance'] < min:
        #         min = data['distance']
                
        # print(min)
        # print(max)
            
    def __getitem__(self, idx):
        
        data = self.label_data[idx]
        
        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data['ground_truth_img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data['train_img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data['ground_truth_img_name']
        output['B_paths'] = data['train_img_name']
        output['distance_label'] = torch.unsqueeze(torch.tensor(data['distance']), 0)

        return output
    
    def __len__(self):
        return len(self.label_data)
    
class CustomDataset1(Dataset):
    """Provides img2img translation unconditioned on the distance label (for testing and debugging)"""

    def __init__(self):
        self.base_path = '/home/nianyli/Desktop/code/thesis/experiment1'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            self.label_data = json.load(file)['data']
        
        # get the upper and lower bounds of the distance labels and normalize
        # between 0 and 1
        dist_min = 1000000
        dist_max = -1000000
        
        for data in self.label_data:
            if data['distance'] < dist_min:
                dist_min = data['distance']
            if data['distance'] > dist_max:
                dist_max = data['distance']
        
        for idx, data in enumerate(self.label_data):
            data['distance'] = (abs(data['distance'] - dist_min) / (dist_max-dist_min)) * 2 - 1
        
        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # scrub the data for img2img translation not conditioned on distance label

        self.new_data = []

        for data in self.label_data:
            if data['distance'] == -1:
                self.new_data.append(data)

        # max = -1
        # min = 1
        # for data in self.label_data:
        #     if data['distance'] > max:
        #         max = data['distance']
        #     if data['distance'] < min:
        #         min = data['distance']
                
        # print(min)
        # print(max)
            
    def __getitem__(self, idx):
        
        data = self.new_data[idx]
        
        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data['ground_truth_img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data['train_img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data['ground_truth_img_name']
        output['B_paths'] = data['train_img_name']

        return output
    
    def __len__(self):
        return len(self.new_data)
    
class CustomDataset2(Dataset):
    """This dataset uses the experimental data and creates a dataset for all 
    multiple translations per training image. That way there is not only one
    ground truth img for each training img, but multiple ground truths based
    on the distance label. This will help ensure that the generator learns to
    generate an image based on the positional embedding as well."""

    def __init__(self):
        self.base_path = '/home/nianyli/Desktop/code/thesis/experiment1'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            self.label_data = json.load(file)['data']
        
        # get the upper and lower bounds of the distance labels and normalize
        # between 0 and 1
        dist_min = 1000000
        dist_max = -1000000
        
        for data in self.label_data:
            if data['distance'] < dist_min:
                dist_min = data['distance']
            if data['distance'] > dist_max:
                dist_max = data['distance']
        
        for idx, data in enumerate(self.label_data):
            data['distance'] = (abs(data['distance'] - dist_min) / (dist_max-dist_min)) * 2 - 1
        
        # group all of the training images by corresponding ground truth image
        grouped_images = dict()

        # sort the labelled data
        self.label_data = sorted(self.label_data, key= lambda x: x['ground_truth_img_name'])

        ground_truth_img_name = 'temp'

        idx = 0
        while idx < len(self.label_data):
            data = self.label_data[idx]
            cur_ground_truth_img_name = data['ground_truth_img_name']
            if ground_truth_img_name != cur_ground_truth_img_name:
                grouped_images[cur_ground_truth_img_name] = []
                ground_truth_img_name = cur_ground_truth_img_name
            new_data = {
                'img_name': data['train_img_name'],
                'original_distance_label': data['distance']
            }
            grouped_images[ground_truth_img_name].append(new_data)
            idx += 1

        # get more data points from the grouped images
        for ground_truth_name in grouped_images:
            img_group = grouped_images[ground_truth_name]
            l = 0
            while l < len(img_group) - 1:
                r = l+1
                train_img_relative_dist = img_group[l]['original_distance_label']
                ground_truth_img_relative_dist = img_group[r]['original_distance_label']

                while (abs(train_img_relative_dist-ground_truth_img_relative_dist) < 1):
                    new_data_1 = {
                        'ground_truth_img_name': img_group[r]['img_name'],
                        'train_img_name': img_group[l]['img_name'],
                        'distance': img_group[l]['original_distance_label']-img_group[r]['original_distance_label']
                    }
                    new_data_2 = {
                        'ground_truth_img_name': img_group[l]['img_name'],
                        'train_img_name': img_group[r]['img_name'],
                        'distance': img_group[r]['original_distance_label']-img_group[l]['original_distance_label']
                    }
                    self.label_data.append(new_data_1)
                    self.label_data.append(new_data_2)
            
                    r += 1

                    if r < len(img_group):
                        train_img_relative_dist = img_group[l]['original_distance_label']
                        ground_truth_img_relative_dist = img_group[r]['original_distance_label']
                    else:
                        break

                l += 1

        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # max = -1
        # min = 1
        # for data in self.label_data:
        #     if data['distance'] > max:
        #         max = data['distance']
        #     if data['distance'] < min:
        #         min = data['distance']
                
        # print(min)
        # print(max)
            
    def __getitem__(self, idx):
        
        data = self.label_data[idx]
        
        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data['ground_truth_img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data['train_img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data['ground_truth_img_name']
        output['B_paths'] = data['train_img_name']
        output['distance_label'] = torch.unsqueeze(torch.tensor(data['distance']), 0)

        return output
    
    def __len__(self):
        return len(self.label_data)
    
class CustomDataset2_Test(Dataset):
    """This dataset uses the experimental data and creates a dataset for all 
    multiple translations per training image. That way there is not only one
    ground truth img for each training img, but multiple ground truths based
    on the distance label. This will help ensure that the generator learns to
    generate an image based on the positional embedding as well."""

    def __init__(self):
        self.base_path = '/home/nianyli/Desktop/code/thesis/experiment3'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            label_data = json.load(file)['data']
        
        self.label_data = []

        # convert label data (quick fix)
        for data in label_data:
            new_data = dict()
            new_data['ground_truth_img_name'] = data['ground_truth_img_name']
            new_data['train_img_name'] = data['train_img_0_info']['img_name']
            new_data['distance'] = data['train_img_0_info']['location']['y']
            self.label_data.append(new_data)

        # get the upper and lower bounds of the distance labels and normalize
        # between 0 and 1
        dist_min = 1000000
        dist_max = -1000000
        
        for data in self.label_data:
            if data['distance'] < dist_min:
                dist_min = data['distance']
            if data['distance'] > dist_max:
                dist_max = data['distance']
        
        # have to hardcode this in
        dist_min = -5
        dist_max = 5

        for idx, data in enumerate(self.label_data):
            data['distance'] = (abs(data['distance'] - dist_min) / (dist_max-dist_min)) * 2 - 1
        
        # group all of the training images by corresponding ground truth image
        grouped_images = dict()

        # sort the labelled data
        self.label_data = sorted(self.label_data, key= lambda x: x['ground_truth_img_name'])

        ground_truth_img_name = 'temp'

        idx = 0
        while idx < len(self.label_data):
            data = self.label_data[idx]
            cur_ground_truth_img_name = data['ground_truth_img_name']
            if ground_truth_img_name != cur_ground_truth_img_name:
                grouped_images[cur_ground_truth_img_name] = []
                ground_truth_img_name = cur_ground_truth_img_name
            new_data = {
                'img_name': data['train_img_name'],
                'original_distance_label': data['distance']
            }
            grouped_images[ground_truth_img_name].append(new_data)
            idx += 1

        # get more data points from the grouped images
        for ground_truth_name in grouped_images:
            img_group = grouped_images[ground_truth_name]
            l = 0
            while l < len(img_group) - 1:
                r = l+1
                train_img_relative_dist = img_group[l]['original_distance_label']
                ground_truth_img_relative_dist = img_group[r]['original_distance_label']

                while (abs(train_img_relative_dist-ground_truth_img_relative_dist) < 1):
                    new_data_1 = {
                        'ground_truth_img_name': img_group[r]['img_name'],
                        'train_img_name': img_group[l]['img_name'],
                        'distance': img_group[l]['original_distance_label']-img_group[r]['original_distance_label']
                    }
                    new_data_2 = {
                        'ground_truth_img_name': img_group[l]['img_name'],
                        'train_img_name': img_group[r]['img_name'],
                        'distance': img_group[r]['original_distance_label']-img_group[l]['original_distance_label']
                    }
                    self.label_data.append(new_data_1)
                    self.label_data.append(new_data_2)
            
                    r += 1

                    if r < len(img_group):
                        train_img_relative_dist = img_group[l]['original_distance_label']
                        ground_truth_img_relative_dist = img_group[r]['original_distance_label']
                    else:
                        break

                l += 1

        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # max = -1
        # min = 1
        # for data in self.label_data:
        #     if data['distance'] > max:
        #         max = data['distance']
        #     if data['distance'] < min:
        #         min = data['distance']
                
        # print(min)
        # print(max)
            
    def __getitem__(self, idx):
        
        data = self.label_data[idx]
        
        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data['ground_truth_img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data['train_img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data['ground_truth_img_name']
        output['B_paths'] = data['train_img_name']
        output['distance_label'] = torch.unsqueeze(torch.tensor(data['distance']), 0)

        return output
    
    def __len__(self):
        return len(self.label_data)
    
class CustomDataset3(Dataset):
    """Dataset for 3 dimensional image translation (x, y, and z). This dataset
    creates groups of images and then outputs a random pair of images from that
    group of images during training."""

    def __init__(self):
        self.base_path = '/home/nianyli/Desktop/code/thesis/experiment2'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            label_data = json.load(file)['data']
        
        # get the upper and lower bounds of the distance labels and normalize
        # between -1 and 1
        dist_min = [1000000, 1000000, 1000000]
        dist_max = [-1000000, -1000000, -1000000]
        
        for data in label_data:
            x, y, z = data['location']

            dist_min = [
                min(dist_min[0], x),
                min(dist_min[1], y),
                min(dist_min[2], z)
            ]

            dist_max = [
                max(dist_max[0], x),
                max(dist_max[1], y),
                max(dist_max[2], z)
            ]
            

        for idx, data in enumerate(label_data):
            data['location'] = (abs(data['location'] - dist_min) / (dist_max-dist_min)) * 2 - 1
        
        # group all of the training images by corresponding ground truth image
        # each group of images will consist of a 
        self.grouped_images = list()

        # sort the labelled data
        self.label_data = sorted(self.label_data, key= lambda x: x['ground_truth_img_name'])

        ground_truth_img_name = 'temp'

        idx = 0
        while idx < len(label_data):
            data = label_data[idx]
            cur_ground_truth_img_name = data['ground_truth_img_name']
            if ground_truth_img_name != cur_ground_truth_img_name:
                ground_truth_img_data = {
                    'img_name': data['ground_truth_img_name'],
                    'original_distance_label': [0, 0, 0]
                }
                self.grouped_images.append([ground_truth_img_data])
                ground_truth_img_name = cur_ground_truth_img_name
            new_data = {
                'img_name': data['train_img_name'],
                'original_distance_label': data['location']
            }
            self.grouped_images[-1].append(new_data)
            idx += 1

        # we define the maximum, normalized translation distance in the x and z
        # directions as 2 (full translation across span of sensors) and the
        # maximum normalized translation distance in the y direction as 1. This
        # is because the sensor data is collected from both the right and left
        # sides of the car, but is only collected in the front (x axis) and 
        # above (z axis) the vehicle.
        max_translation_distance = [2, 1, 2]

        # determine the maximum distance 

        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
            
    def __getitem__(self, idx):

        # get group of images
        data = self.grouped_images[idx]
        
        # randomly select a group of images

        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data['ground_truth_img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data['train_img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data['ground_truth_img_name']
        output['B_paths'] = data['train_img_name']
        output['distance_label'] = torch.unsqueeze(torch.tensor(data['distance']), 0)

        return output
    
    def __len__(self):
        return len(self.label_data)
    
class CustomDataset4(Dataset):
    """Dataset contains groups of images from different timesteps in the CARLA
    simulation. Each image can be paired with any other image in the group to
    increase training diversity. The images all have different sensor locations.
    The variables that describe the sensor location are:
        x, y, z, z_angle (yaw)
    """

    def __init__(self):
        self.base_path = '/home/nianyli/Desktop/code/thesis/3D_trans_small'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            dataset_info = json.load(file)
        
        self.sensor_limits = dataset_info['sensor_limits']
        self.label_data = dataset_info['data']

        # normalize all location labels between -1 and 1
        for idx, data in enumerate(self.label_data):
            for key in data:
                for loc_key in data[key]['location']:
                    min_val, max_val = (self.sensor_limits[f'{loc_key}_limits'][0], 
                                        self.sensor_limits[f'{loc_key}_limits'][1])
                    data[key]['location'][loc_key] = (abs(data[key]['location'][loc_key] - min_val) / (max_val-min_val)) * 2 - 1
        
        self.group_keys = []
        for key in self.label_data[0]:
            self.group_keys.append(key)

        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
            
    def __getitem__(self, idx):

        # get group of images
        data = self.label_data[idx]
        
        keys = self.group_keys.copy()

        # randomly select two images from the group of images
        ground_truth_key = random.choice(keys)
        keys.remove(ground_truth_key)
        starting_img_key = random.choice(keys)

        ground_truth_location = data[ground_truth_key]['location']
        starting_img_location = data[starting_img_key]['location']

        # the relative location will be from the starting image relative to the
        # ground truth img in the form of [x, y, z, z_angle]
        relative_x = starting_img_location['x'] - ground_truth_location['x']
        relative_y = starting_img_location['y'] - ground_truth_location['y']
        relative_z = starting_img_location['z'] - ground_truth_location['z']
        relative_z_angle = starting_img_location['z_angle'] - ground_truth_location['z_angle']
        relative_location = [relative_x, relative_y, relative_z, relative_z_angle]

        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data[ground_truth_key]['img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data[starting_img_key]['img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data[ground_truth_key]['img_name']
        output['B_paths'] = data[starting_img_key]['img_name']
        output['location'] = torch.tensor(relative_location)

        return output
    
    def __len__(self):
        return len(self.label_data)
    
class CustomDataset5(Dataset):
    """Dataset contains groups of images from different timesteps in the CARLA
    simulation. Each image can be paired with any other image in the group to
    increase training diversity. The images all have different sensor locations.
    The variables that describe the sensor location are:
        x, y, z, z_angle (yaw). 
    
    *** This dataset defaults to translating between a random image and the 
    original ground truth image, but employs a randomization variable that will
    occasionally return a translation between two random images in the image
    group providing increased diversity to the data while maintaining the
    most common translation during inference time as the bulk of the training data. 
    """

    def __init__(self, randomizing_coefficient):

        self.randomizing_coefficient = randomizing_coefficient

        self.base_path = '/home/nianyli/Desktop/code/thesis/3D_trans_small'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'/*.json')[0]
        
        # get json data from label file path
        with open(self.label_file_path, 'r') as file:
            dataset_info = json.load(file)
        
        self.sensor_limits = dataset_info['sensor_limits']
        self.label_data = dataset_info['data']

        # normalize all location labels between -1 and 1
        for idx, data in enumerate(self.label_data):
            for key in data:
                for loc_key in data[key]['location']:
                    min_val, max_val = (self.sensor_limits[f'{loc_key}_limits'][0], 
                                        self.sensor_limits[f'{loc_key}_limits'][1])
                    data[key]['location'][loc_key] = (abs(data[key]['location'][loc_key] - min_val) / (max_val-min_val)) * 2 - 1
        
        self.group_keys = []
        for key in self.label_data[0]:
            self.group_keys.append(key)

        # define the image transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
            
    def __getitem__(self, idx):

        # get group of images
        data = self.label_data[idx]
        
        keys = self.group_keys.copy()
            
        if random.random() < self.randomizing_coefficient:
            # randomly select two images from the group of images
            ground_truth_key = random.choice(keys)
            keys.remove(ground_truth_key)
            starting_img_key = random.choice(keys)
        else:
            ground_truth_key = keys[-1]
            starting_img_key = random.choice(keys[:-1])

        ground_truth_location = data[ground_truth_key]['location']
        starting_img_location = data[starting_img_key]['location']

        # the relative location will be from the starting image relative to the
        # ground truth img in the form of [x, y, z, z_angle]
        relative_x = starting_img_location['x'] - ground_truth_location['x']
        relative_y = starting_img_location['y'] - ground_truth_location['y']
        relative_z = starting_img_location['z'] - ground_truth_location['z']
        relative_z_angle = starting_img_location['z_angle'] - ground_truth_location['z_angle']
        relative_location = [relative_x, relative_y, relative_z, relative_z_angle]

        # load the ground truth image (A)
        img_A_full_path = os.path.join(self.base_path, 
                                       data[ground_truth_key]['img_name'])
        img_A = cv2.imread(img_A_full_path)
    
        # load the alternate view image (B)
        img_B_full_path = os.path.join(self.base_path,
                                       data[starting_img_key]['img_name'])
        img_B = cv2.imread(img_B_full_path)
        
        # convert the images to a normalized tensor ([-1, 1] for each pixel rgb)
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        
        # get distance label in tensor form
        
        output = dict()
        
        output['A'] = img_A * 2 - 1
        output['B'] = img_B * 2 - 1
        output['A_paths'] = data[ground_truth_key]['img_name']
        output['B_paths'] = data[starting_img_key]['img_name']
        output['location'] = torch.tensor(relative_location)

        return output
    
    def __len__(self):
        return len(self.label_data)
    

if __name__=='__main__':
    dataset = CustomDataset4()

    dataset[23]
    print(len(dataset))