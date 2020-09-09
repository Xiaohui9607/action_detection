# data loader class for training
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pickle
import dill

IMG_EXTENSIONS = ('.pkl',)

def make_dataset(path):
    if not os.path.exists(path):
        raise FileExistsError('some subfolders from dataset do not exists!')
    samples = []
    for sample in os.listdir(path):
        if not sample.startswith('.'):
            image = os.path.join(path, sample)
            samples.append(image)
    return samples


def pickle_loader(path):
    pickle_file = open(path, 'rb')
    samples = pickle.load(pickle_file)
    return samples


class my_Dataset(Dataset):
    def __init__(self, root, image_transform=None, loader=pickle_loader, device='cpu'):
        if not os.path.exists(root):
            raise FileExistsError('{0} does not exists!'.format(root))

        self.image_transform = lambda vision: torch.cat([image_transform(single_image).unsqueeze(0)
                                                         for single_image in vision], dim=0)

        self.samples = make_dataset(root)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.loader = loader
        self.device = device

    def __getitem__(self, index):
        imgs_with_labels = self.loader(self.samples[index])
        vision = imgs_with_labels['images']
        labels = imgs_with_labels['labels']
        if self.image_transform is not None:
            vision = self.image_transform(vision)

        # return vision.to(self.device), torch.from_numpy(np.array(list(labels))).to(self.device)
        return vision.to(self.device), torch.from_numpy(np.array(labels)).to(self.device)

    def __len__(self):
        return len(self.samples)

def build_dataloader(opt):

    def crop(im):
        height, width = im.shape[:-1]
        width = max(height, width)
        im = im[:width, :width,]
        return im

    image_transform = transforms.Compose([
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor()
    ])


    train_ds = my_Dataset(
        root=os.path.join(opt.data_dir+'/train'),
        image_transform=image_transform,
        loader=pickle_loader,
        device=opt.device
    )

    valid_ds = my_Dataset(
        root=os.path.join(opt.data_dir+'/test'),
        image_transform=image_transform,
        loader=pickle_loader,
        device=opt.device
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_dl, valid_dl



if __name__ == '__main__':

    # from options import Options
    # opt = Options().parse()
    # # samples = pickle_loader("../data/watch-n-bot/fridge_k1/out/data_01-05-11.pkl")
    #
    # tr, va = build_dataloader(opt)
    # dataloader = {'train': tr, 'valid': va}

    from options import Options
    import cv2
    opt = Options().parse()
    opt.batch_size = 5
    # opt.data_dir = '../data/watch-n-bot/fridge_k1/sequence'

    tr, vl = build_dataloader(opt)

    for index, (a,b) in enumerate(tr):
        # a: 5 x 10 x c x w x h
        # imgs = a[0].unbind(0)
        # imgs = list(map(lambda x: (x.permute([1, 2, 0]).numpy()*255).squeeze().astype(np.uint8), imgs))
        # for index, img in enumerate(imgs):
        #     cv2.imwrite('l_{}.png'.format(index), img)
        #     exit(1)

        print(index)
        print(a.shape)
        print(b.shape)
        print("***")



