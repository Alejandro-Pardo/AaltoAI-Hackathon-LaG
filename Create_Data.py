from file_camera import FileCamera
from MakFormer import MaskFormer
import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
cam = FileCamera("/root/lisan-ai-gaib/AaltoAI-Hackathon-LaG/IMG_1081.mp4",predefined_size=(1280, 720))
maskformer = MaskFormer(cam.get_shape(),device=device)
current_frame = cam.get_image_one_by_one()
counter = 0
while current_frame is not None:
    data = np.zeros((1280,720,2))
    current_frame = cam.get_image_one_by_one()
    print(current_frame.shape)
    heat_map = maskformer.gen_heat(current_frame)
    ang = maskformer.ang
    data[:,:,0] = heat_map.T
    if ang is not None:
        data[:,:,1] = ang.T
    np.save(f'/root/lisan-ai-gaib/AaltoAI-Hackathon-LaG/data/hack/data_{counter}.npy', data)
    counter += 1