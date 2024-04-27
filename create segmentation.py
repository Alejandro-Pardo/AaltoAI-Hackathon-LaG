import numpy as np
import cv2
import numpy as np
import torch
from MakFormer import MaskFormer
from torchvision.transforms import Resize
# Create a resize transform



resize_transform = Resize((1920, 1080))
im1 = cv2.imread('IMG_1090.jpg')
im2 = cv2.imread('IMG_1091.jpg')

image_array1 = np.array(im1)
image1 = torch.tensor(image_array1)
image_array1 = resize_transform(image1).numpy()
image_array2 = np.array(im2)
image2 = torch.tensor(image_array2)
image_array2 = resize_transform(image2).numpy()
maskformer = MaskFormer(image_array1.shape,device="cpu")
for i in range(2):
    print(image1.shape)
    maskformer.gen_heat(image_array1)
    maskformer.gen_heat(image_array2)


im = np.array(maskformer.optical_flow)
im = np.repeat(im, 3, axis=2)
im = im.astype(np.uint8)
cv2.imwrite(f'output_flow.jpg', im)
