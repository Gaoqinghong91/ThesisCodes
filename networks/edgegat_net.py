"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

from http.client import ImproperConnectionState
from telnetlib import PRAGMA_HEARTBEAT
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATv2Conv
import numpy as np
import scipy.sparse as s
from layers import *
import os
import functools
from dgl.nn.pytorch import EGATConv

def connected_adjacency(width,heigh, connect, patch_size=(1, 1)):
    """
    Creates an adjacency matrix from an image where nodes are considered adjacent 
    based on 4-connected or 8-connected pixel neighborhoods.

    :param image: 2 or 3 dim array
    :param connect: string, either '4' or '8'
    :param patch_size: tuple (n,m) used if the image will be decomposed into 
                   contiguous, non-overlapping patches of size n x m. The 
                   adjacency matrix will be formed from the smaller sized array
                   e.g. original image size = 256 x 256, patch_size=(8, 8), 
                   then the image under consideration is of size 32 x 32 and 
                   the adjacency matrix will be of size 
                   32**2 x 32**2 = 1024 x 1024
    :return: adjacency matrix as a sparse matrix (type=scipy.sparse.csr.csr_matrix)
    """

    r=width 
    c = heigh
    # print(r,c)
    r = int(r / patch_size[0])
    c = int(c / patch_size[1])

    if connect == '4':
        # constructed from 2 diagonals above the main diagonal
        # print(np.ones(c-1))
        # print(np.append(np.ones(c-1)))
        d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
        d2 = np.ones(c*(r-1))
        upper_diags = s.diags([d1, d2], [1, c])
        return upper_diags + upper_diags.T

    elif connect == '8':
        # constructed from 4 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
        d2 = np.append([0], d1[:c*(r-1)])
        d3 = np.ones(c*(r-1))
        d4 = d2[1:-1]
        upper_diags = s.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
        return upper_diags + upper_diags.T
    else:
        raise ValueError('Invalid parameter \'connect\'={connect}, must be "4" or "8".'
                     .format(connect=repr(connect)))
#######################################################
# adj = connected_adjacency(40,24, '4').toarray()#192
###########################################################
adj = connected_adjacency(40,32, '4').toarray()#256
src, dst = np.nonzero(adj)

def genG(src, dst,feat,final_x):
    g_batch=[]
    batch=feat.shape[0]
    final_x=final_x.reshape(batch,1280,3)
    grayMat = final_x.type(torch.FloatTensor)
    for i in range(batch):
        node_feat=feat[i,:,:].squeeze()
        b1=grayMat[i,src,:]
        b2=grayMat[i,dst,:]
        #edge_feat=torch.norm((b1 - b2), dim=1,keepdim=True).to("cuda:0")
        edge_feat=torch.abs(b1 - b2).to("cuda:0")
        g = dgl.graph((src, dst)).to("cuda:0")
        g.ndata["feat"]=node_feat
        g.edata["feat"]=edge_feat
        # g.to("cuda:0")
        g_batch.append(g)
    return g_batch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out

# ABN_module = InPlaceABN
# BatchNorm2d = functools.partial(ABN_module, activation='none')
affine_par = True
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)

class ResNet(nn.Module):
    """
    Basic ResNet101.
    """
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=1, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        self.features = []
        #x = (x - 0.45) / 0.225
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        self.features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.features.append(x)
        x = self.layer2(x)
        self.features.append(x)
        x = self.layer3(x)
        self.features.append(x)
        # x_dsn = self.dsn(x)
        x = self.layer4(x)
        self.features.append(x)

        return self.features


class GAT_edge(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 heads,
                 activation,pretrained):
        super(GAT_edge, self).__init__()
        self.num_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.resnet_model = ResNet(Bottleneck, [3, 4, 23, 3])
        self.num_ch_enc = np.array([128, 256, 512, 1024, 2048])
        if pretrained:
            pretrained_weights = torch.load('/home/walle/Documents/monodepth2-modify/splits/resnet101-imagenet.pth')
            model_dict = self.resnet_model.state_dict()
            self.resnet_model.load_state_dict({k: v for k, v in pretrained_weights.items() if k in model_dict})
        # EGATConv()
        # EGATConv(in_node_feats=128,
        #         in_edge_feats=3,
        #         out_node_feats=64,
        #         out_edge_feats=6,
        #         num_heads=3)
        self.gat_layers.append(EGATConv(in_node_feats=128,
                in_edge_feats=1,
                out_node_feats=64,
                out_edge_feats=1,
                num_heads=3))#32
        # hidden layers
        # for l in range(1, n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
        self.gat_layers.append(EGATConv(in_node_feats=64*3,
                in_edge_feats=1*3,
                out_node_feats=32,
                out_edge_feats=1,
                num_heads=3))
        # output projection
        self.gat_layers.append(EGATConv(in_node_feats=32*3,
                in_edge_feats=1*3,
                out_node_feats=32,
                out_edge_feats=1,
                num_heads=1))

        # self.fc = nn.Sequential(
        #     nn.Linear(3, in_feats), nn.Sigmoid())
        self.red = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.cls = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x,final_x):#
        features = self.resnet_model(x)
        feat=features[-1]
        feat=self.red(feat)
        # print(feat.shape)
        batch,channels,width,heigh= feat.shape
        feat=feat.view(batch,channels,width*heigh).permute(0,2,1)
        nfeature_list=[]
        efeature_list=[]
        b_g=genG(src, dst,feat,final_x)
        # print(b_g)
        for g in b_g:
            #addlinear_input_layers_s#############
            nfeature_list.append(g.ndata["feat"].float())
            efeature_list.append(g.edata["feat"].float())
        for i in range(self.num_layers):
            for j, g in enumerate(b_g):
                nfeature_list[j],efeature_list[j]=self.gat_layers[i](g, nfeature_list[j],efeature_list[j])
                efeature_list[j]=efeature_list[j].flatten(1)
                nfeature_list[j]=nfeature_list[j].flatten(1)
        for m, g in enumerate(b_g):
            # print(feature_list[m].shape) 
            nfeature_list[m],efeature_list[m]=self.gat_layers[-1](g,  nfeature_list[m],efeature_list[m])
            nfeature_list[m]=nfeature_list[m].mean(1)

        channels=nfeature_list[0].shape[1] 
        # print(channels)   
        feature_node=torch.stack(nfeature_list, dim=0)
        ###########192####################
        # features[-1] = feature_node.view(batch,channels,24,40)
        ###########192####################

        ###########256####################
        features[-1] = feature_node.view(batch,channels,32,40)
        ###########256####################
        features[-1]=self.cls(features[-1])
        
        return features
