import torch
import torch.nn as nn

class DenseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, transpose = None):
        super().__init__()
        self.block_in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = 2 * out_channels + in_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.act2 = nn.ReLU()
        
        if transpose:
            self.tconv = nn.ConvTranspose2d(self.block_out_channels, int(out_channels / 2), kernel_size = (2, 2), stride = (2, 2))
            self.bn3 = nn.BatchNorm2d(self.tconv.out_channels)
        else:
            self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
            self.bn3 = nn.BatchNorm2d(self.block_out_channels)
                    
        self.act3 = nn.ReLU()
    
    def forward(self, x, concat_channels = None):
        if concat_channels is not None:
            c1 = self.act1(self.bn1(self.conv1(torch.cat([x, * concat_channels], dim = 1))))
            c2 = self.act2(self.bn2(self.conv2(torch.cat([x, * concat_channels, c1], dim = 1))))
            t1 = self.act3(self.bn3(self.tconv(torch.cat([x, * concat_channels, c1, c2], dim = 1))))
            
            return c1, c2, t1
            
        else:
            c1 = self.act1(self.bn1(self.conv1(x)))
            c2 = self.act2(self.bn2(self.conv2(torch.cat([x, c1], dim = 1))))
            p1 = self.act3(self.bn3(self.pool1(torch.cat([x, c1, c2], dim = 1))))
                    
            return c1, c2, p1
   

class DefectSegNet(nn.Module):
    def __init__(self, in_channels = 1):
        super().__init__()
        
        self.convblock1 = DenseConvBlock(in_channels = in_channels, out_channels = 4)
        self.convblock2 = DenseConvBlock(in_channels = self.convblock1.block_out_channels, out_channels = 16)
        self.convblock3 = DenseConvBlock(in_channels = self.convblock2.block_out_channels, out_channels = 32)
        
        self.tconvblock1 = DenseConvBlock(in_channels = self.convblock3.block_out_channels, out_channels = 64, transpose = True)
        self.tconvblock2 = DenseConvBlock(in_channels = int(self.convblock2.block_out_channels + 
                                                            (2 * self.convblock3.out_channels) + (self.tconvblock1.out_channels / 2)), out_channels = 32, transpose = True)
        self.tconvblock3 = DenseConvBlock(in_channels = int(self.convblock1.block_out_channels + 
                                                            (2 * self.convblock2.out_channels) + (self.tconvblock2.out_channels / 2)), out_channels = 16, transpose = True)
        
        self.conv13 = nn.Conv2d(int(self.convblock1.block_in_channels + 
                                   (2 * self.convblock1.out_channels) + (self.tconvblock3.out_channels / 2)), 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act19 = nn.ReLU()
        self.conv14 = nn.Conv2d(int(self.conv13.in_channels + self.conv13.out_channels), 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.act20 = nn.Sigmoid() 
        
    def forward(self, x):
        c1, c2, p1 = self.convblock1(x)
        c3, c4, p2 = self.convblock2(p1)
        c5, c6, p3 = self.convblock3(p2)
        
        c7, c8, t1 = self.tconvblock1(p3, concat_channels = [])
        c9, c10, t2 = self.tconvblock2(t1, concat_channels = [p2, c5, c6])
        c11, c12, t3 = self.tconvblock3(t2, concat_channels = [p1, c3, c4])
        
        c13 = self.act19(self.conv13(torch.cat([x, c1, c2, t3], dim = 1)))
        c14 = self.act20(self.conv14(torch.cat([x, c1, c2, t3, c13], dim = 1)))
        
        return c14
