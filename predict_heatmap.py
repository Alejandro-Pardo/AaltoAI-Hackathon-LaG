import torch
import numpy as np
from MakFormer import MaskFormer
from file_camera import FileCamera
from Models import Encoder, Decoder
from Camera_DataLoader import HeatMapDataset
from torch.utils.data import DataLoader, random_split




de
encoder = Encoder()
decoder = Decoder(encoder.out_shape)

epochs = 30
dataset = HeatMapDataset()

# Define the size of the test set
test_size = int(0.3 * len(dataset))  # 30% for testing
train_size = len(dataset) - test_size

# Split the dataset into a training set and a test set
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
for epoch in epochs:
    for i, data in enumerate(train_loader):
        inputs = data
        outputs = decoder(encoder(inputs))
        loss = torch.nn.functional.mse_loss(outputs, inputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} Loss: {loss.item()}")


