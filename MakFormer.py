from transformers import DetrConfig, DetrModel, AutoImageProcessor, DetrForObjectDetection,MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MaskFormer:
    def __init__(self,image_shape, device="cpu"):
        if torch.cuda.is_available():
            device = "cuda"
        self.image_shape = image_shape
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
        self.device = device
        self.heat_image = np.zeros((self.image_shape[0],self.image_shape[1]))
    def people_mask(self,image):
        inputs = self.feature_extractor(images=torch.tensor(image), return_tensors="pt")
        inputs.to(self.device)
        outputs = self.model(**inputs)
        seg_mask = self.feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[(image.shape[0],image.shape[1])])[0]

        seg_mask = seg_mask == 12 
        return seg_mask

    def gen_heat(self,image):
        people_mask = self.people_mask(image)
        self.heat_image *= 0.5
        self.heat_image[people_mask] += 100
        return self.heat_image
        
