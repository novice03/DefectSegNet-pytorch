import torch
import torch.nn as nn

class DefectSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.act3 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(9, 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act4 = nn.ReLU()
        self.conv4 = nn.Conv2d(25, 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act5 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.act6 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(41, 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act7 = nn.ReLU()
        self.conv6 = nn.Conv2d(73, 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act8 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.act9 = nn.ReLU()
        
        self.conv7 = nn.Conv2d(105, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act10 = nn.ReLU()
        self.conv8 = nn.Conv2d(169, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act11 = nn.ReLU()
        self.tconv1 = nn.ConvTranspose2d(233, 32, kernel_size = (2, 2), stride = (2, 2))
        self.act12 = nn.ReLU()
        
        self.conv9 = nn.Conv2d(137, 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act13 = nn.ReLU()
        self.conv10 = nn.Conv2d(169, 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act14 = nn.ReLU()
        self.tconv2 = nn.ConvTranspose2d(201, 16, kernel_size = (2, 2), stride = (2, 2))
        self.act15 = nn.ReLU()
        
        self.conv11 = nn.Conv2d(57, 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act16 = nn.ReLU()
        self.conv12 = nn.Conv2d(73, 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act17 = nn.ReLU()
        self.tconv3 = nn.ConvTranspose2d(89, 8, kernel_size = (2, 2), stride = (2, 2))
        self.act18 = nn.ReLU()
        
        self.conv13 = nn.Conv2d(17, 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act19 = nn.ReLU()
        self.conv14 = nn.Conv2d(21, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act20 = nn.Sigmoid()       
    
    def forward(self, x):
        c1 = self.act1(self.conv1(x))
        c2 = self.act2(self.conv2(torch.cat([x, c1], dim = 1)))
        p1 = self.act3(self.pool1(torch.cat([x, c1, c2], dim = 1)))
        
        c3 = self.act4(self.conv3(p1))
        c4 = self.act5(self.conv4(torch.cat([p1, c3], dim = 1)))
        p2 = self.act6(self.pool2(torch.cat([p1, c3, c4], dim = 1)))
        
        c5 = self.act7(self.conv5(p2))
        c6 = self.act8(self.conv6(torch.cat([p2, c5], dim = 1)))
        p3 = self.act9(self.pool3(torch.cat([p2, c5, c6], dim = 1)))
        
        c7 = self.act10(self.conv7(p3))
        c8 = self.act11(self.conv8(torch.cat([p3, c7], dim = 1)))
        t1 = self.act12(self.tconv1(torch.cat([p3, c7, c8], dim = 1)))
        
        c9 = self.act13(self.conv9(torch.cat([p2, c5, c6, t1], dim = 1)))
        c10 = self.act14(self.conv10(torch.cat([p2, c5, c6, t1, c9], dim = 1)))
        t2 = self.act15(self.tconv2(torch.cat([p2, c5, c6, t1, c9, c10], dim = 1)))
        
        c11 = self.act16(self.conv11(torch.cat([p1, c3, c4, t2], dim = 1)))
        c12 = self.act17(self.conv12(torch.cat([p1, c3, c4, t2, c11], dim = 1)))
        t3 = self.act18(self.tconv3(torch.cat([p1, c3, c4, t2, c11, c12], dim = 1)))
        
        c13 = self.act19(self.conv13(torch.cat([x, c1, c2, t3], dim = 1)))
        c14 = self.act20(self.conv14(torch.cat([x, c1, c2, t3, c13], dim = 1)))
    
        return c14
