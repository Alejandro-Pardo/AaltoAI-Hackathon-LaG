import torch
from Models import Encoder, Decoder
from Camera_DataLoader import HeatMapDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
encoder = Encoder(kernel_size=6, stride=2, padding=2, image_shape=(1280, 720))
decoder = Decoder(encoder.output_size, kernel_size=6, stride=2, padding=2)
enc_parameters = torch.load('/root/lisan-ai-gaib/AaltoAI-Hackathon-LaG/model/encoder_47.pth')
encoder.load_state_dict(enc_parameters)

dec_parameters = torch.load('/root/lisan-ai-gaib/AaltoAI-Hackathon-LaG/model/decoder_47.pth')
decoder.load_state_dict(dec_parameters)
resize_transform = Resize((256, 256))
device = "cuda"
batch_size = 10
encoder =encoder.to(device)
decoder = decoder.to(device)
dataset = HeatMapDataset(transform=resize_transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

encoder.eval()
decoder.eval()
counter = 0

with torch.no_grad():
    for i, data in enumerate(train_loader):
        inputs = data.to(device)
        outputs = decoder(encoder(inputs))
        for out in range(outputs.shape[0]):
            image = outputs[out].cpu().detach().numpy()[0,:,:]
            #image = image.reshape(image.shape[1],image.shape[2],image.shape[0])
            plt.figure()
            plt.imshow(image)
            plt.savefig(f"/root/lisan-ai-gaib/AaltoAI-Hackathon-LaG/data/images/img_{counter}.jpg")
            counter += 1









