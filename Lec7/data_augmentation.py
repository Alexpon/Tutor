import os

from PIL import Image
from torchvision import transforms


def augment(img_list, transform_type):
    if transform_type=='None':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                #transforms.ToTensor()
        ])
    elif transform_type=='RandomHorizontalFlip':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.ToTensor()
        ])
    elif transform_type=='RandomVerticalFlip':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomVerticalFlip(p=0.5),
                #transforms.ToTensor()
        ])
    elif transform_type=='ColorJitter':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                #transforms.ToTensor()
        ])
    elif transform_type=='CenterCrop':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.CenterCrop(256)
                #transforms.ToTensor()
        ])

    elif transform_type=='RandomRotation':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomRotation(degrees=90)
        ])
    elif transform_type=='RandomCrop':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomCrop(128)
        ])
    elif transform_type=='RandomResizedCrop':
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomResizedCrop(256)
        ])

    new_img_list = []
    for img in img_list:
        new_img_list.append(transform(img))
    
    return new_img_list

img_path_list = ['Img-9033.jpg', 'n02095314_740.jpg', 'n02096437_1143.jpg', 'nature-3569122__480.jpg']
img_list = []

# load img
for img_path in img_path_list:
    filename = os.path.join('./data', img_path)
    img_list.append(Image.open(filename))

# do data augmentation
transform_type = 'RandomResizedCrop'
new_img_list = augment(img_list, transform_type)

# save img
for img, img_name in zip(new_img_list, img_path_list):
    img.save('./data/{}_{}'.format(transform_type, img_name))
