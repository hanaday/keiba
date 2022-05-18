import torch
from torch import nn
from torch.nn import functional as F


#確認用
#from torchsummary import summary
#device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えるならGPUで、そうでないならCPUで計算する


class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self, in_size:int, hidden_size:int, out_size:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        #x = self.conv1(x)
        #x = self.batchnorm1(x)
        #x = F.ReLU(x)

        #x = self.conv2(x)
        #x = self.batchnorm2(x)
        #x = F.ReLU(x)
        return x
   
    """
    Combine output with the original input
    """
    def forward(self, x): 
        return x + self.convblock(x) # skip connection


class TResNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, NUM_ResBlock):
        super(TResNet, self).__init__()
        
        self.l1 = nn.Sequential()
        n_in = 1
        n_out = 8
        for i in range(7):
            stride = 1
            if i%2 == 0:
                stride = 2
                if i == 6:
                    stride = (3, 2)
            if i != 0:
                self.l1.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.l1.add_module('Conv_%d' %i, nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1))
            self.l1.add_module('Mish_%d' %i, nn.Mish())
            for j in range(NUM_ResBlock):
                self.l1.add_module('ResBlock_%d_%d' %(i, j), ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out * 2


        self.l2 = nn.Sequential()
        n_in = 1
        n_out = 8
        for i in range(7):
            stride = 1
            if i%2 == 0:
                stride = 2
                if i == 6:
                    stride = (2, 3)
            if i != 0:
                self.l2.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.l2.add_module('Conv_%d' %i, nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1))
            self.l2.add_module('Mish_%d' %i, nn.Mish())
            for j in range(NUM_ResBlock):
                self.l2.add_module('ResBlock_%d_%d' %(i, j), ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out * 2
            

        self.l3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)


        self.l4 = torch.nn.Sequential()
        n_in = 512
        n_out = 256
        for i in range(7):
            stride = 1
            if i == 6:
                stride = (1, 3)
            self.l4.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.l4.add_module('ConvT_%d' %i, nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=stride, padding=0))
            self.l4.add_module('Mish_%d' %i, nn.Mish())
            for j in range(NUM_ResBlock):
                self.l4.add_module('ResBlock_%d_%d' %(i, j), ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out // 2
            if n_in == 8:
                n_out = 1
        self.l4.add_module('Norm_last', nn.BatchNorm2d(1))
        self.l4.add_module('Conv_0', nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0))


        self.softmax = torch.nn.Softmax(dim=2)
 
    def forward(self, x, CulLoss=True):
        #print(x.shape)
        x_t = torch.transpose(x, 2, 3)
        #print(x_t.shape)
        x = self.l1(x)
        #print(x.shape)
        x_t = self.l2(x_t)
        #print(x_t.shape)
        x_t = torch.transpose(x_t, 2, 3)
        #print(x_t.shape)
        x = torch.concat((x, x_t), dim=1)
        #print(x.shape)
        x = self.l3(x)
        x = self.l4(x)
        #print(x.shape)

        if CulLoss == False and self.training == False:
            x = self.softmax(x)
 
        return x




class CSPBlock(nn.Module):
    def __init__(self, in_size:int, out_size:int, NUM_ResBlock):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_size), 
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1), 
            nn.Mish(), 
            )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_size), 
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), 
            nn.Mish(), 
            )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(out_size*2), 
            nn.Conv2d(out_size*2, out_size, kernel_size=3, stride=1, padding=1), 
            nn.Mish(), 
            )

        self.resblock = nn.Sequential()
        for i in range(NUM_ResBlock):
            self.resblock.add_module('ResBlock_%d' %i, ResBlock(out_size, out_size, out_size))

    def forward(self, x): 
        x = self.conv1(x)
        short = self.conv2(x)
        x = self.resblock(x)
        x = torch.concat((x, short), dim=1)
        x = self.conv3(x)
        return x



class K2Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, NUM_ResBlock):
        super(K2Net, self).__init__()
        
        self.l1 = nn.Sequential()
        n_in = 16
        n_out = n_in * 2
        self.l1.add_module('Conv_first', nn.Conv2d(1, n_in, kernel_size=3, stride=1, padding=1))
        for i in range(3):
            self.l1.add_module('CSPBlock_%d' %i, CSPBlock(n_in, n_out, NUM_ResBlock))
            n_in = n_out
            n_out = n_out * 2


        self.l2 = nn.Sequential()
        n_in = 16
        n_out = n_in * 2
        self.l2.add_module('Conv_first', nn.Conv2d(1, n_in, kernel_size=3, stride=1, padding=1))
        for i in range(3):
            self.l2.add_module('CSPBlock_%d' %i, CSPBlock(n_in, n_out, NUM_ResBlock))
            n_in = n_out
            n_out = n_out * 2


        self.l1_shuf = nn.Sequential()
        n_in = 16
        n_out = n_in * 2
        self.l1_shuf.add_module('Conv_first', nn.Conv2d(1, n_in, kernel_size=3, stride=1, padding=1))
        for i in range(3):
            self.l1_shuf.add_module('CSPBlock_%d' %i, CSPBlock(n_in, n_out, NUM_ResBlock))
            n_in = n_out
            n_out = n_out * 2


        self.l2_shuf = nn.Sequential()
        n_in = 16
        n_out = n_in * 2
        self.l2_shuf.add_module('Conv_first', nn.Conv2d(1, n_in, kernel_size=3, stride=1, padding=1))
        for i in range(3):
            self.l2_shuf.add_module('CSPBlock_%d' %i, CSPBlock(n_in, n_out, NUM_ResBlock))
            n_in = n_out
            n_out = n_out * 2
            

        self.l3_1 =  nn.Sequential(
            nn.BatchNorm2d(256), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.Mish(), 
            )

        self.l3_1_shuf = nn.Sequential(
            nn.BatchNorm2d(256), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.Mish(), 
            )

        self.l3_2 = nn.Sequential(
            nn.BatchNorm2d(512), 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.Mish(), 
            nn.BatchNorm2d(512), 
            nn.Conv2d(512, 1024, kernel_size=4, stride=(3, 2), padding=1), 
            nn.Mish(), 
            nn.BatchNorm2d(1024), 
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), 
            nn.Mish(), 
            )


        self.l4 = torch.nn.Sequential()
        n_in = 512
        n_out = 256
        for i in range(7):
            stride = 1
            if i == 6:
                stride = (1, 3)
            self.l4.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.l4.add_module('ConvT_%d' %i, nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=stride, padding=0))
            self.l4.add_module('Mish_%d' %i, nn.Mish())
            for j in range(NUM_ResBlock):
                self.l4.add_module('ResBlock_%d_%d' %(i, j), ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out // 2
            if n_in == 8:
                n_out = 1
        self.l4.add_module('Norm_last', nn.BatchNorm2d(1))
        self.l4.add_module('Conv_0', nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0))


        self.softmax = torch.nn.Softmax(dim=2)
 
    def forward(self, x, CulLoss=True):
        #print(x.shape)
        SEED = 42
        torch.manual_seed(SEED)
        idx = torch.randperm(9)
        x_shuf = x[:, :, :, idx]
        x_shuf_trans = torch.transpose(x_shuf, 2, 3)

        x_trans = torch.transpose(x, 2, 3)
        #print(x_t.shape)
        x = self.l1(x)
        #print(x.shape)
        x_trans = self.l2(x_trans)
        #print(x_t.shape)
        x_trans = torch.transpose(x_trans, 2, 3)
        #print(x_t.shape)
        x = torch.concat((x, x_trans), dim=1)
        #print(x.shape)
        x = self.l3_1(x)

        x_shuf = self.l1_shuf(x_shuf)
        x_shuf_trans = self.l2_shuf(x_shuf_trans)
        x_shuf_trans = torch.transpose(x_shuf_trans, 2, 3)
        x_shuf = torch.concat((x_shuf, x_shuf_trans), dim=1)
        x_shuf = self.l3_1_shuf(x)

        x = torch.concat((x, x_shuf), dim=1)
        x = self.l3_2(x)

        x = self.l4(x)
        #print(x.shape)

        if CulLoss == False and self.training == False:
            x = self.softmax(x)
 
        return x



class Origin(torch.nn.Module):
    def __init__(self, input_dim, output_dim, drop=0.2):
        super(Origin, self).__init__()
        
        self.main = nn.Sequential()
        n_in = 1
        n_out = 8
        for i in range(7):
            stride = 1
            if i%2 == 0:
                stride = 2
                if i == 6:
                    stride = (3, 2)
            if i != 0:
                self.main.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.main.add_module('Conv_%d' %i, nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1))
            self.main.add_module('Mish_%d' %i, nn.Mish())
            self.main.add_module('ResBlock_%d_0' %i, ResBlock(n_out, n_out, n_out))
            self.main.add_module('ResBlock_%d_1' %i, ResBlock(n_out, n_out, n_out))
            self.main.add_module('ResBlock_%d_2' %i, ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out * 2


        self.main2 = nn.Sequential()
        n_in = 1
        n_out = 8
        for i in range(7):
            stride = 1
            if i%2 == 0:
                stride = 2
                if i == 6:
                    stride = (2, 3)
            if i != 0:
                self.main2.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.main2.add_module('Conv_%d' %i, nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1))
            self.main2.add_module('Mish_%d' %i, nn.Mish())
            self.main2.add_module('ResBlock_%d_0' %i, ResBlock(n_out, n_out, n_out))
            self.main2.add_module('ResBlock_%d_1' %i, ResBlock(n_out, n_out, n_out))
            self.main2.add_module('ResBlock_%d_2' %i, ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out * 2
            
        self.l3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)

        self.main3 = torch.nn.Sequential()
        n_in = 512
        n_out = 256
        for i in range(7):
            stride = 1
            if i == 6:
                stride = (1, 3)
            self.main3.add_module('Norm_%d' %i, nn.BatchNorm2d(n_in))
            self.main3.add_module('ConvT_%d' %i, nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=stride, padding=0))
            self.main3.add_module('Mish_%d' %i, nn.Mish())
            self.main3.add_module('ResBlock_%d_0' %i, ResBlock(n_out, n_out, n_out))
            self.main3.add_module('ResBlock_%d_1' %i, ResBlock(n_out, n_out, n_out))
            self.main3.add_module('ResBlock_%d_2' %i, ResBlock(n_out, n_out, n_out))
            n_in = n_out
            n_out = n_out // 2
            if n_in == 8:
                n_out = 1
        self.main3.add_module('Norm_last', nn.BatchNorm2d(1))
        self.main3.add_module('Conv_0', nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0))

        self.l4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            #torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(256),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            #torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            #torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            #torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            #torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            #torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 1, kernel_size=2, stride=(1, 3), padding=0),
            torch.nn.Mish(),
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        )

        self.softmax = torch.nn.Softmax(dim=2)
 
    def forward(self, x, CulLoss=True):
        print(x.shape)
        x_t = torch.transpose(x, 2, 3)
        print(x_t.shape)
        x = self.main(x)
        print(x.shape)
        x_t = self.main2(x_t)
        print(x_t.shape)
        x_t = torch.transpose(x_t, 2, 3)
        print(x_t.shape)
        x = torch.concat((x, x_t), dim=1)
        print(x.shape)
        x = self.l3(x)
        #x = self.main3(x)
        x = self.l4(x)
        print(x.shape)
        put()

        if CulLoss == False and self.training == False:
            x = self.softmax(x)
 
        return x



# ロジスティック回帰モデル
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, drop=0.2):
        super(LogisticRegression, self).__init__()
        
        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            #torch.nn.LeakyReLU(0.1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
        )

        self.l2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
        )

        self.l2_l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
        )

        self.l2_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.3),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.3),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.3),
        )

        self.l3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=(3, 2), padding=1),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
        )

        self.l4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(256),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1, padding=0),
            torch.nn.Mish(),
            torch.nn.Dropout(drop),
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 1, kernel_size=2, stride=(1, 3), padding=0),
            torch.nn.Mish(),
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        )

        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, x, CulLoss=True):
        x_t = torch.transpose(x, 2, 3)
        #print(x_t.shape)
        x = self.l1(x)
        #print(x.shape)
        x_t_l = self.l2_l(x_t)
        #print(x_t_l.shape)
        x_t_r = self.l2_r(x_t)
        #print(x_t_r.shape)
        x_t = torch.concat((x_t_l, x_t_r), dim=1)
        #print(x_t.shape)
        x_t = torch.transpose(x_t, 2, 3)
        #print(x_t.shape)
        x = torch.concat((x, x_t), dim=1)
        #print(x.shape)
        #x = self.l2(x)
        x = self.l3(x)
        #print(x.shape)
        x = self.l4(x)
        #print(x.shape)
        if CulLoss == False and self.training == False:
            x = self.softmax(x)
        
        return x




# 重みの初期化を行う関数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        m.bias.data.fill_(0)

# ネットワークを可視化する
#summary(TResNet().to(device), (32, 1, 18, 9))