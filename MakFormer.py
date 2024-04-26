from transformers import DetrConfig, DetrModel, AutoImageProcessor, DetrForObjectDetection,MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class MaskFormer:
    def __init__(self,image_shape, device="cpu"):
        if torch.cuda.is_available():
            device = "cuda"
        self.image_shape = image_shape
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade").to(device)
        self.device = device
        self.heat_image = np.zeros((self.image_shape[0],self.image_shape[1]))
        self.old_image = None
        self.old_mask = None
        self.first = False
        self.p0 = None
    def people_mask(self,image):
        inputs = self.feature_extractor(images=torch.tensor(image), return_tensors="pt")
        inputs.to(self.device)
        outputs = self.model(**inputs)
        seg_mask = self.feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[(image.shape[0],image.shape[1])])[0]
        seg_mask = seg_mask == 12 
        return seg_mask.to("cpu").numpy()


    def create_optical_flow(self,image):
        if not self.first:
            self.first = True
            return
        else:
            old_gray = cv2.cvtColor(self.old_image, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = np.array(mag).reshape(self.image_shape[0], self.image_shape[1])

            return mag
    def gen_heat(self,image):
        people_mask = self.people_mask(image)
        optical_flow = self.create_optical_flow(image)
        
        self.heat_image *= 0.9
        self.heat_image[people_mask] += 17
        print(f"MAK: {self.heat_image.max()}")
        if optical_flow is not None:
            self.heat_image += optical_flow
        self.old_image = image
        self.old_mask = people_mask
        self.heat_image = np.clip(self.heat_image, a_min=None, a_max=255)
        return self.heat_image.astype(np.uint8)
        
