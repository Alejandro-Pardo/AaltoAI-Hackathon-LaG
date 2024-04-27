import torch.nn as nn





class Encoder(nn.Module):

    def __init__(self,image_shape=(1280,720)):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=20, stride=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64,128, kernel_size=20, stride=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=20, stride=5, padding=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=20, stride=5, padding=2)
        self.relu4 = nn.ReLU()
        stride = 1
        image_shape = (1280, 720)
        kernel_sizes = [20, 20, 20,20]
        paddings = [5, 5, 5,5]
        for i in range(4):
            output_shape = ((image_shape[0] - kernel_sizes[i] + 2*paddings[i]) // stride + 1,
                   (image_shape[1] - kernel_sizes[i] + 2*paddings[i]) // stride + 1)
        
        self.output_size = output_shape[0] * output_shape[1] * 512
        self.out_shape  = output_shape
        self.Linear1 = nn.Linear(self.output_size, 1024)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Linear1(x)

        return x


class Decoder(nn.Module):

    def __init__(self,output_shape):
        super(Decoder, self).__init__()

        self.Linear1 = nn.Linear(1024, output_shape[0] * output_shape[1] * 512)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=20, stride=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=20, stride=5, padding=2)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=20, stride=5, padding=2)
        self.relu4 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(64, 2, kernel_size=20, stride=5, padding=2)
    
        self.output_shape = output_shape
    def forward(self, x):

        x = self.Linear1(x)
        x = self.relu1(x)
        x = x.reshape(x.shape[0], 512, self.output_shape[0], self.output_shape[1])
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.relu4(x)
        x = self.conv4(x)

        return x
