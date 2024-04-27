import torch.nn as nn





class Encoder(nn.Module):

    def __init__(self,image_shape=(1280,720),kernel_size=40,stride=5,padding=2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(2, 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4,8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu7 = nn.ReLU()
        self.conv8  = nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu8 = nn.ReLU()
        stride = 1
        image_shape = (1280, 720)
        kernel_sizes = [20, 20, 20,20]
        paddings = [5, 5, 5,5]
        output_shape = image_shape
        for i in range(4):
            output_shape = ((output_shape[0] - kernel_sizes[i] + 2*paddings[i]) // stride + 1,
                   (output_shape[1] - kernel_sizes[i] + 2*paddings[i]) // stride + 1)
        print(output_shape)
        self.output_size = output_shape[0] * output_shape[1] * 512
        print(self.output_size)
        self.out_shape  = output_shape
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
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Linear1(x)
        x = self.relu9(x)
        x = self.Linear2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, output_shape, kernel_size=40, stride=5, padding=2):
        super(Decoder, self).__init__()

        self.Linear1 = nn.Linear(512, output_shape[0] * output_shape[1] * 512)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu5 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu6 = nn.ReLU()
        self.conv6 = nn.ConvTranspose2d(16, 8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu7 = nn.ReLU()
        self.conv7 = nn.ConvTranspose2d(8, 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu8 = nn.ReLU()
        self.conv8 = nn.ConvTranspose2d(4, 2, kernel_size=kernel_size, stride=stride, padding=padding)
    
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
        x = self.relu5(x)
        x = self.conv5(x)
        x = self.relu6(x)
        x = self.conv6(x)
        x = self.relu7(x)
        x = self.conv7(x)
        x = self.relu8(x)
        x = self.conv8(x)

        return x
