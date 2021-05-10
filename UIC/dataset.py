import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

def listdir(folder, suffix):
    """ Output the path of files in the folder with specific suffix"""
    list_path = []
    for root, _, files in os.walk(folder, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                list_path.append(osp.join(root, f))
    return list_path

class DatasetGivenLabels(Dataset):
    """ A self-defined dataset with updated input labels"""
    def __init__(self, root, labels=None, transform=None, suffix='.jpg'):
        dir_list = []
        for dir_item in os.listdir(root):
            if osp.isdir(osp.join(root, dir_item)):
                dir_list.append(dir_item)
        dir_list.sort()

        imagedirs = []
        gt        = []
        for label_id, dir_item in enumerate(dir_list):
            sub_folder = osp.join(root, dir_item)
            imagedirs_item = listdir(sub_folder, suffix=suffix)
            imagedirs += imagedirs_item
            gt += [label_id for _ in range(len(imagedirs_item))]

        self.transform = transform
        self.imgs = imagedirs
        self.gt = gt
        self.labels = labels

    def __getitem__(self, index):
        with open(self.imgs[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img
        return img, int(self.labels[index]), index

    def __len__(self):
        return len(self.imgs) 