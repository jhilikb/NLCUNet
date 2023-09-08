import time
import os
from test_options import TestOptions
from data_loader import DataLoader
from GANloss import GANModel

import ntpath
import time
import util


opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1  

dataset = DataLoader(opt)
model = GANModel(opt)


im_dir = os.path.join(opt.results_dir, opt.name)
res_dir = im_dir
util.mkdirs(res_dir)
vis_buffer = []

# test
for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals(testing=True)
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)

    image_dir = im_dir
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]



    for label, image_numpy in visuals.items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(image_numpy, save_path)




