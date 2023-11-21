import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.relu(out)
        out = self.conv(out)
        out += residual
        out = self.relu(out)
        return out
    
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.grl = GradientReversalLayer()

        # 其他模型定义...
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)#测距特征是2x20的
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)#2x20——1x10
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(10, 2)

    def forward(self, x, alpha):
        x = self.grl.apply(x, alpha)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

'''  
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.domainclassifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),#测距特征是2x22的
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),#2x22——1x11
            nn.Flatten(),
            nn.Linear(11, 2)
        )

    def forward(self, x):
        x = self.domainclassifier(x)
        return x
'''

class DGM(nn.Module):
    def __init__(self, hidden_1=4, hidden_2=8):
        super(DGM, self).__init__()
        
        # Encoder(对信号x进行编码，对应于decoder)
        #感觉不应该有这个总的encoder，不然就是各做各的，x的重构对于环境和测距特征的提取参数没有作用

        #environment encoder
        self.encoder_environment = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0),#8x50——6x48
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0),#6x48——4x46
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0),#4x46——2x44
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))#2x44——2x22
        )

        #envir decoder
        self.decoder_envir = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=(4,1),padding=0, output_padding=0),#转换为8x25
            #ResidualBlock(1,1),
            #ResidualBlock(1,1),
            ResidualBlock(1,1)
        )

        #range error encoder
        self.encoder_range = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,11), stride=1, padding=0),#8x50——6x40
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(3,11), stride=1, padding=0),#6x40——4x30
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(3,11), stride=1, padding=0),#4x30——2x20
            nn.BatchNorm2d(1),
            nn.ReLU(),
            ResidualBlock(1,1)
            #ResidualBlock(1,1),
            #ResidualBlock(1,1)#2x20——2x20
        )
        
        #range decoder
        self.decoder_range = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(4,6), stride=(4,1), padding=0, output_padding=0),#2x20转换为8x25
            ResidualBlock(1,1)
            #ResidualBlock(1,1),
            #ResidualBlock(1,1)
        )
        
        #域分类器
        self.domain_classifier = DomainClassifier()

        # Environment classifier
        self.environment_classifier = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),#环境特征大小为2x22
            nn.BatchNorm2d(1),
            nn.ReLU(),
            #nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(1),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),#2x22——1x11
            nn.Flatten(),
            nn.Linear(11, 6)#应该是得到[batch_size,1,1,6]
        )
        
        # Range error regressor
        self.range_regressor = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),#测距特征大小为2x20
            nn.BatchNorm2d(1),
            nn.ReLU(),
            #nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(1),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),#2x20——1x10
            nn.Flatten(),
            nn.Linear(10, 2)
        )
        
    def forward(self, input_signal):
        # Encode input signal
        encoded_environment = self.encoder_environment(input_signal)
        encoded_range = self.encoder_range(input_signal)
        
        # Decode encoded signal
        decoded_envir = self.decoder_envir(encoded_environment)
        decoded_range = self.decoder_range(encoded_range)
        decoded_signal = torch.cat((decoded_envir, decoded_range), dim=-1)
        
        # Predict environment label
        environment_logits = self.environment_classifier(encoded_environment)
        environment_label = F.softmax(environment_logits, dim=-1)
        #environment_label = torch.argmax(softmax, dim=-1) + 1
        #environment_label = torch.unsqueeze(environment_label, dim=1)
        
        # Estimate distance 
        #range_error = self.range_regressor(encoded_range)
        distance = self.range_regressor(encoded_range)

        #domian classifier
        alpha = 1
        domain_out = self.domain_classifier(encoded_range, alpha)
        domain_out = F.softmax(domain_out, dim=-1)
        #domain_out = torch.argmax(temp, dim=-1)

        return decoded_signal, environment_label, distance, domain_out