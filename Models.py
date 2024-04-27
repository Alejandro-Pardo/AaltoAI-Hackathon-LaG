import torch.nn as nn

def compute_output_size(input_size, kernel_size, padding, stride):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1



class Encoder(nn.Module):

    def __init__(self,image_shape=(1280,720),kernel_size=40,stride=5,padding=2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(2, 8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8,16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        output_shape = image_shape
        output_shape = (8,8)
        self.output_size = 8192

        self.Linear1 = nn.Linear(self.output_size, 1024)
        self.relu9 = nn.ReLU()
        self.Linear2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Linear1(x)
        x = self.relu9(x)
        x = self.Linear2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoder_output_size, kernel_size=40, stride=5, padding=2):
        super(Decoder, self).__init__()

        self.Linear1 = nn.Linear(512, 1024)
        self.relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(1024, encoder_output_size)
        self.relu2 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu5 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(16, 8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu6 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(8, 2, kernel_size=kernel_size, stride=stride, padding=padding)
    
        self.output_shape = (1280, 720)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.relu1(x)
        x = self.Linear2(x)
        x = self.relu2(x)
        x = x.reshape(x.shape[0], 128, 8, 8)
        x = self.conv1(x)
        x = self.relu3(x)
        x = self.conv2(x)
        x = self.relu4(x)
        x = self.conv3(x)
        x = self.relu5(x)
        x = self.conv4(x)
        x = self.relu6(x)
        x = self.conv5(x)
        print(x.shape)
        return x
