

import argparse
import os
import util
import torch


class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        
        self.parser.add_argument('--name', default='dhcs4', type=str,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--dataroot', default='./dataset/', type=str,
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--n_domains', default=2, type=int, help='Number of domains to transfer among')

        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='none',
                                 help='scaling and cropping of images at load time [none|resize|resize_and_crop|crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')

        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')

        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        
        self.parser.add_argument('--netG_n_blocks', type=int, default=9,
                                 help='number of residual blocks to use for netG')  # 4 for small networks
        self.parser.add_argument('--netG_n_shared', type=int, default=0,
                                 help='number of blocks to use for netG shared center module')


        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='insert dropout for the generator')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

        

        self.isTrain = False

        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        self.parser.add_argument('--which_epoch', default=100, type=int, help='which epoch to load for inference?')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc (determines name of folder to load from)')

        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run (if serial_test not enabled)')
        self.parser.add_argument('--serial_test', default="True" ,action='store_true', help='read each image once from folders in sequential order')

        self.parser.add_argument('--autoencode', action='store_true', help='translate images back into its own domain')


        self.parser.add_argument('--show_matrix', action='store_true', help='visualize images in a matrix format as well')
        self.initialized = True



    def parse(self):
        if not self.initialized:
            
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.gpu_ids =[0]
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        return self.opt
        

