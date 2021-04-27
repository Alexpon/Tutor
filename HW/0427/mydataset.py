from torch.utils.data import Dataset
from PIL import Image
from torchvision import tansforms

from ipdb import set_trace

class MyDataset(Dataset):
    def __init__(self, filenames, labels):
        self.filenames  = filenames
        self.labels     = labels
        self.without_da = transforms.ToTensor()
        self.with_da    = transforms.Compose([
                            transforms.CenterCrop(size=10),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        # 讀檔 / Data Augmentation / ...
        filename = self.filenames[index]
        pil_img = Image.open(filename) # PIL Image
        tensor_img = self.data_aug_fn(pil_img)     

        label = self.labels[index]
        
        return tensor_img, label

    def __len__(self):
        return len(self.labels)
