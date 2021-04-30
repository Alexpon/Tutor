import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from ipdb import set_trace


class MyDataset(Dataset):
    def __init__(self, filenames, labels):
        self.filenames  = filenames
        self.labels     = labels
        self.without_da = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
        self.with_da    = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(size=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        # Read imgs, Data Augmentation, To tensor...
        filename = os.path.join('data/dog_wolf_small', self.filenames[index])
        img_pil = Image.open(filename) # PIL Image
        img_tensor = self.without_da(img_pil)     
        label = self.labels[index]
        return img_tensor, label

    def __len__(self):
        return len(self.labels)
