import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image
import os
import glob
import os.path
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass



def get_transform(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _ = self.load_image(DB, index_B)
            bundle.update( {'B': B_img, 'DB': DB} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'



class DataLoader():
    def name(self):
        return 'DataLoader'

    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
