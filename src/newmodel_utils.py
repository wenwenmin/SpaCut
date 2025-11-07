import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

# Initialisation
import torch
import torch.nn as nn
from torch.nn import init


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):   
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        y = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels= 2, out_channels= 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        
        return x * self.sigmoid(y)



class ForegroundPredictNet(nn.Module):
    def __init__(self, n_channels=313):
        super(ForegroundPredictNet, self).__init__()

        
        self.conv1x1_1 = nn.Conv2d(n_channels, 256, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(128, 64, kernel_size=1)

        self.dsc1 = DepthwiseSeparableConv(64, 128)
        self.dsc2 = DepthwiseSeparableConv(128, 256)
        self.dsc3 = DepthwiseSeparableConv(256, 128)
        self.dsc4 = DepthwiseSeparableConv(128, 64)
        
        self.ca1 = ChannelAttention(128)
        self.ca2 = ChannelAttention(256)
        self.ca3 = ChannelAttention(128)
        self.ca4 = ChannelAttention(64)
        

        self.fusion_attn = ChannelAttention(64+128+256+128+64)


        self.out_conv1 = nn.Conv2d((64+128+256+128+64), 256, kernel_size=1)
        self.out_conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.out_conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.out_conv4 = nn.Conv2d(64, 2, kernel_size=1)
        
    def forward(self, inputs):

        x = self.conv1x1_1(inputs)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)
        

        h1 = self.dsc1(h0)
        h1 = self.ca1(h1)
        
        h2 = self.dsc2(h1)
        h2 = self.ca2(h2)
        
        h3 = self.dsc3(h2)
        h3 = self.ca3(h3)
        
        h4 = self.dsc4(h3)
        h4 = self.ca4(h4)
        
        x = torch.cat((h0, h1, h2, h3, h4), 1)
        x = self.fusion_attn(x)
        
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = self.out_conv3(x)
        x = self.out_conv4(x)
        
        return x
        
    def get_feature(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)

        
        h1 = self.dsc1(h0)
        h1 = self.ca1(h1)
        
        h2 = self.dsc2(h1)
        h2 = self.ca2(h2)
        
        h3 = self.dsc3(h2)
        h3 = self.ca3(h3)
        
        h4 = self.dsc4(h3)
        h4 = self.ca4(h4)
        
        x = torch.cat((h0, h1, h2, h3, h4), 1)
        x = self.fusion_attn(x)
        
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = self.out_conv3(x)
        return x

    def get_spot_feature(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)

        return h0

    def get_conv_spot_feature(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)

        h1 = self.dsc1(h0)
        h1 = self.ca1(h1)
        
        h2 = self.dsc2(h1)
        h2 = self.ca2(h2)
        
        h3 = self.dsc3(h2)
        h3 = self.ca3(h3)
        
        h4 = self.dsc4(h3)
        h4 = self.ca4(h4)
        return h4
    
class CellPredictNet(nn.Module):

    def __init__(self, cond_channels=64):
        super(CellPredictNet, self).__init__()
        self.flim = nn.Sequential(
            nn.Conv2d(cond_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.spatial_attn = SpatialAttention(kernel_size=5)
        self.conv_refine = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.predict = nn.Conv2d(8, 2, kernel_size=1)
        
    def forward(self, nuclei_soft_mask, cond):

        f_mul = self.flim(cond)

        x = nuclei_soft_mask * f_mul

        x_combined = torch.cat([x, nuclei_soft_mask], dim=1)

        x_refined = self.spatial_attn(self.conv_refine(x_combined))
        
        output = self.predict(x_refined)
        
        return output



# Other
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def get_softmask(cell_nuclei_mask, tau=5):
    contours, _ = cv2.findContours((cell_nuclei_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    I = np.zeros((cell_nuclei_mask.shape[0], cell_nuclei_mask.shape[1]))
    for i in range(cell_nuclei_mask.shape[0]):
        for j in range(cell_nuclei_mask.shape[1]):
            # Why j, i?
            I[i, j] = sigmoid(cv2.pointPolygonTest(contours[0], [j,i], True) / tau)
    return I

def get_seg_mask(sample_seg, sample_n):
    """
    Generate the segmentation mask with unique cell IDs
    """
    sample_n = np.squeeze(sample_n)

    # Background prob is average probability of all cells EXCEPT FOR NUCLEI
    sample_probs = softmax(sample_seg, axis=1)
    bgd_probs = np.expand_dims(np.mean(sample_probs[:, 0, :, :], axis=0), 0)
    fgd_probs = sample_probs[:, 1, :, :]
    probs = np.concatenate((bgd_probs, fgd_probs), axis=0)
    final_seg = np.argmax(probs, axis=0)

    # Map predictions to original cell IDs
    ids_orig = np.unique(sample_n)
    if ids_orig[0] != 0:
        ids_orig = np.insert(ids_orig, 0, 0)
    ids_pred = np.unique(final_seg)
    if ids_pred[0] != 0:
        ids_pred = np.insert(ids_pred, 0, 0)
    ids_orig = ids_orig[ids_pred]

    dictionary = dict(zip(ids_pred, ids_orig))
    dictionary[0] = 0

    final_seg_raw = np.vectorize(dictionary.get)(final_seg)

    # Add nuclei back in
    final_seg_orig = np.where(sample_n > 0, sample_n, final_seg_raw)

    return final_seg_orig, final_seg_raw


if __name__ == "__main__":
    # 测试 ImprovedForegroundPredictNet 和 CellPredictNet
    n_gene = 313
    patch_size = 48
    batch_size = 2

    gene_map = torch.randn(batch_size, n_gene, patch_size, patch_size)
    nuclei_soft_mask = torch.rand(batch_size, 1, patch_size, patch_size)


    fg_net = ForegroundPredictNet(n_channels=n_gene)
    fg_out = fg_net(gene_map)
    print("ImprovedForegroundPredictNet output shape:", fg_out.shape)

    fg_feat = fg_net.get_feature(gene_map)
    print("ImprovedForegroundPredictNet feature shape:", fg_feat.shape)

    cell_net = CellPredictNet(cond_channels=fg_feat.shape[1])
    cell_out = cell_net(nuclei_soft_mask, fg_feat)
    print("CellPredictNet output shape:", cell_out.shape)
