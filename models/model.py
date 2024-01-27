import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, resnet50
from models.UniAD import getUniAD


def get_My_efficientnetb3_fusion():
    model = efficientnet_b3(pretrained=True)
    features = model.children().__next__()
    modelList = list(features.children())
    stage1 = modelList[:2]
    stage2 = modelList[2]
    stage3 = modelList[3]
    stage4 = modelList[4:6]
    backbone = [stage1, stage2, stage3, stage4]
    return backbone, 240


def get_My_resnet50(pretrained=True):
    model = resnet50(pretrained=pretrained)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels


class eca_layer(nn.Module):
    def __init__(self, k_size=5):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Reconstruct(nn.Module):
    def __init__(self):
        super(Reconstruct, self).__init__()
        backbone, target_channel = get_My_efficientnetb3_fusion()
        self.stage1 = nn.Sequential(*backbone[0])
        self.target_channel = target_channel
        self.stage2 = nn.Sequential(*backbone[1])
        self.stage3 = nn.Sequential(*backbone[2])
        self.stage4 = nn.Sequential(*backbone[3])
        # print(self.stage1)
        for param in self.stage1.parameters():
            param.requires_grad = False
        for param in self.stage2.parameters():
            param.requires_grad = False
        for param in self.stage3.parameters():
            param.requires_grad = False
        for param in self.stage4.parameters():
            param.requires_grad = False

        self.ECA = eca_layer()

        self.linearPreject = nn.Conv2d(in_channels=self.target_channel, out_channels=256, kernel_size=1)

        self.Transformer = getUniAD()


    def FeatureRepresentation(self, x):
        feature1 = self.stage1(x)   # [B, C1, H1, H1]
        feature2 = self.stage2(feature1)    # [B, C2, H2, H2]
        feature3 = self.stage3(feature2)    # [B, C3, H3, H3]
        feature4 = self.stage4(feature3)    # [B, C4, H4, H4]

        scale1to4 = feature4.shape[-1] / feature1.shape[-1]
        scale2to4 = feature4.shape[-1] / feature2.shape[-1]
        scale3to4 = feature4.shape[-1] / feature3.shape[-1]

        feature1 = F.interpolate(feature1, scale_factor=scale1to4, mode="bilinear") # H1 -> H4
        feature2 = F.interpolate(feature2, scale_factor=scale2to4, mode="bilinear") # H2 -> H4
        feature3 = F.interpolate(feature3, scale_factor=scale3to4, mode="bilinear") # H3 -> H4

        feature_ori = torch.cat((feature1, feature2, feature3, feature4), dim=1) # [B, (C1+C2+C3+C4), H4, W4]

        return feature_ori

    def forward(self, image):
        featureOri = self.FeatureRepresentation(image) # [B, (C1+C2+C3+C4), H4, W4]
        featureEcf = self.ECA(featureOri) # [B, (C1+C2+C3+C4), H4, W4]

        featureEcf_std = 20*(torch.norm(featureEcf.detach()) / featureEcf.shape[1])
        noise = torch.normal(mean=0, std=featureEcf_std, size=featureEcf.shape).cuda()

        encoder_input = self.linearPreject((featureEcf + noise)) # [B, 256, H4, W4]
        decoder_output = self.Transformer(encoder_input)

        return featureOri, featureEcf, decoder_output

    def infer(self, image):
        self.Transformer.transformer.neighbor_mask = None
        featureOri = self.FeatureRepresentation(image)  # [B, (C1+C2+C3+C4), H4, W4]
        featureEcf = self.ECA(featureOri)  # [B, (C1+C2+C3+C4), H4, W4]

        encoder_input = self.linearPreject(featureEcf)
        encoder_output = self.Transformer.getEncoderOutput(encoder_input)

        return encoder_output



class MyRes50(nn.Module):
    def __init__(self, backbone, out_channels) -> None:
        super(MyRes50, self).__init__()
        self.stage1 = nn.Sequential(*backbone[0:5])
        self.ECA1 = eca_layer()
        self.out_channels = out_channels
        self.stage2 = backbone[5]
        self.ECA2 = eca_layer()
        self.stage3 = backbone[6]
        self.ECA3 = eca_layer()
        self.stage4 = backbone[7]
        self.ECA4 = eca_layer()

        # for param in self.stage1.parameters():
        #     param.requires_grad = False
        # for param in self.stage2.parameters():
        #     param.requires_grad = False
        # for param in self.stage3.parameters():
        #     param.requires_grad = False
        # for param in self.stage4.parameters():
        #     param.requires_grad = False

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(out_channels + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(512, 230)

    def forward(self, image, gender):
        x = self.ECA1(self.stage1(image))
        x = self.ECA2(self.stage2(x))
        x = self.ECA3(self.stage3(x))
        x = self.ECA4(self.stage4(x))

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)

        gender_encode = self.gender_bn(self.gender_encoder(gender))
        gender_encode = F.relu(gender_encode)

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return self.classifier(x)


if __name__ == '__main__':
    data = torch.ones((2, 3, 512, 512)).cuda()
    model = Reconstruct().cuda()
    print(model)
    featureOri, featureEcf, decoder_output = model(data)
    print(f"featureOri.shape: {featureOri.shape}")
    print(f"featureEcf.shape: {featureEcf.shape}")
    print(f"decoder_output.shape: {decoder_output.shape}")
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad==True])

    print("Number of training parameter: %.2fM" % (total / 1e6))
