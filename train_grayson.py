"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
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

class CustomDataset(Dataset):
    def __init__(self):
        self.base_path = 'C:\\Users\\gbbyrd\Desktop\\thesis\\data\\experiment1'
        # img_paths = glob.glob(path+'\\*.png')
        self.label_file_path = glob.glob(self.base_path+'\\*.json')[0]
        
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

if __name__ == '__main__':
    
    opt = TrainOptions().parse()   # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    dataset = torch.utils.data.DataLoader(  # create custom dataset
            CustomDataset(),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
