import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms.functional as TF


# In project 5, you need to adjust the model architecture.
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        ################################ you need to modify the cnn model here ################################

        # after convolutoin, the feature map size = ((origin + padding*2 - kernel_size) / stride) + 1
        # input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # ((224+2*1-3)/1)+1=224  # output_shape=(64,224,224)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output_shape=(64,112,112) # (224)/2

        self.cnn2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # output_shape=(128,112,112)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # output_shape=(64,56,56)

        self.cnn3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # output_shape=(64,56,56)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # output_shape=(64,28,28)

        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(512, 12)
        # =================================================================================================== #

    def forward(self, x):

        ################################ you need to modify the cnn model here ################################
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.fc3(out)
        # =================================================================================================== #

        return out


# In project 5, you need to adjust the model architecture.
class MultiScaleResNet(nn.Module):
    def __init__(self, num_classes=12):
        super(MultiScaleResNet, self).__init__()

        # 初始化模型
        self.efficeientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.resnet2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet3 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 先取得每個模型的 in_features 再改成 Identity
        effnet_dim = self.efficeientnet.classifier[1].in_features
        resnet2_dim = self.resnet2.fc.in_features
        resnet3_dim = self.resnet3.fc.in_features

        self.efficeientnet.classifier[1] = nn.Identity()
        self.resnet2.fc = nn.Identity()
        self.resnet3.fc = nn.Identity()

        # 定義分類器
        self.classifier = nn.Linear(
            effnet_dim + resnet2_dim + resnet3_dim,
            num_classes,
        )

    def forward(self, x):
        # 三個不同尺寸輸入
        x1 = TF.resize(x, [384, 384])
        x2 = TF.resize(x, [224, 224])
        x3 = TF.resize(x, [112, 112])

        # 特徵提取
        f1 = self.efficeientnet(x1)
        f2 = self.resnet2(x2)
        f3 = self.resnet3(x3)

        # 特徵融合
        fused = torch.cat([f1, f2, f3], dim=1)
        out = self.classifier(fused)
        return out
