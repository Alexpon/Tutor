import os
import random
import numpy as np

from PIL import Image, PILLOW_VERSION
from torchvision import transforms
from torchvision.transforms.functional import _get_inverse_affine_matrix

from ipdb import set_trace

def get_params(degrees, translate, scale_ranges, shears, img_size):
    """Get parameters for affine transformation
    Returns:
        sequence: params to be passed to the affine transformation
    """
    angle = random.uniform(degrees[0], degrees[1])
    if translate is not None:
        max_dx = translate[0] * img_size[0]
        max_dy = translate[1] * img_size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if shears is not None:
        shear = random.uniform(shears[0], shears[1])
    else:
        shear = 0.0

    return angle, translations, scale, shear


# data list
img_path_list = ['Img-9033.jpg', 'n02095314_740.jpg', 'n02096437_1143.jpg', 'nature-3569122__480.jpg']


new_img_list = []

for img_path in img_path_list:
    # load PIL image
    filename = os.path.join('./data', img_path)
    pil_img  = Image.open(filename) # pillow type image
    
    # get parameters of data augmentations
    # sample parameters
    ret = get_params(degrees=[-180, 180], translate=[0.1, 0.1], scale_ranges=[0.8, 1.2], shears=[0.8, 1.2], img_size=pil_img.size)

    # you can assign your img size
    #output_size = pil_img.size
    output_size = (512, 512)
    
    # augment imgs
    center = (pil_img.size[0] * 0.5 + 0.5, pil_img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, *ret)
    aug_img = pil_img.transform(output_size, Image.AFFINE, matrix, resample=False)   
    new_img_list.append((aug_img, matrix)) #aug_img: PIL image
    # new_img_list.append((aug_img, np.array(aug_img), matrix))


# save img
for img_item, img_name in zip(new_img_list, img_path_list):
    img_item[0].save('./data/new_{}'.format(img_name))
