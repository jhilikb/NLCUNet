import numpy as np
import torch
from collections import OrderedDict
import util as util

import networks
import os




class GANModel():
    def name(self):
        return 'GANModel'

    def __init__(self, opt):
        super(GANModel, self)#.__init__(opt)

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

    def load_network(self, network, network_label, epoch):
        save_filename = '%d_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load(save_path)

    def set_input(self, input):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
        self.image_paths = input['path']

    def test(self):
        with torch.no_grad():
            self.visuals = [self.real_A]
            self.labels = ['real_%d' % self.DA]

            # cache encoding to not repeat it everytime
            encoded = self.netG.encode(self.real_A, self.DA)
            for d in range(self.n_domains):
                if d == self.DA and not self.opt.autoencode:
                    continue
                fake = self.netG.decode(encoded, d)
                self.visuals.append( fake )
                self.labels.append( 'fake_%d' % d )


    def get_image_paths(self):
        return self.image_paths


    #
    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))
