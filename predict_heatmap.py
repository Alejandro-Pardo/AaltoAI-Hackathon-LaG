import torch
import numpy as np
from MakFormer import MaskFormer
from file_camera import FileCamera
from Models import Encoder, Decoder
from Camera_DataLoader import HeatMapDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize
from torchsummary import summary
import matplotlib.pyplot as plt
# Create a resize transform

resize_transform = Resize((256, 256))
device ="cuda"

encoder = Encoder(kernel_size=6, stride=2, padding=2, image_shape=(1280, 720))
decoder = Decoder(encoder.output_size, kernel_size=6, stride=2, padding=2)

encoder = encoder.to(device)
decoder = decoder.to(device)
#summary(encoder,(2,1280,720))
#summary(decoder,)
epochs = 500
dataset = HeatMapDataset("/Users/leonmagnus/Documents/AaltoAI-Hackathon-LaG/data",transform=resize_transform)
# Define the size of the test set
test_size = int(0.3 * len(dataset))  # 30% for testing
train_size = len(dataset) - test_size

# Split the dataset into a training set and a test set
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# Create data loaders
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
epoch_loss = np.zeros(epochs)
epoch_test_loss = np.zeros(epochs)
for epoch in range(epochs):
    avg_loss = 0
    test_loss = 0
    encoder.train()
    decoder.train()
    for i, data in enumerate(train_loader):
        inputs = data.to(device)
        outputs = decoder(encoder(inputs))
        loss = torch.nn.functional.mse_loss(outputs, inputs)
        avg_loss+= loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss[epoch] = avg_loss / len(train_loader)
    print(f"Epoch {epoch} Loss: {loss.item()}")
    encoder.eval()
    decoder.eval()
    for i, data in enumerate(test_loader):
        inputs = data.to(device)
        outputs = decoder(encoder(inputs))
        loss = torch.nn.functional.mse_loss(outputs, inputs)
        test_loss += loss.item()
    epoch_test_loss[epoch] = test_loss / len(test_loader)

    plt.figure()
    plt.plot(np.arange(epoch + 1), epoch_loss[:epoch + 1], label="Train Loss")
    plt.savefig("loss.png")
    plt.close()
    plt.figure()
    plt.plot(np.arange(epoch + 1), epoch_test_loss[:epoch + 1], label="Test Loss")
    plt.savefig("test_loss.png")
    
    

