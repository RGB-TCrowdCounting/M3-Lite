import torch
from torch import nn
import math
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from einops import rearrange, repeat
from module.BaseBlocks import BasicConv2d,BasicODConv2d,BasicDSConvConv2d
from utils.functions import PatchEmbeding
from utils.ODCONV import ODConv2d
from utils.functions import Attention_split
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from utils.functions import transform
from utils.DSConv   import DSConv
from torch.nn import functional as F
from backbone.mix_transformer import Attention
from torch.cuda.amp import custom_bwd, custom_fwd
import warnings
warnings.filterwarnings("ignore")





class IDEMHSF(nn.Module):
    def __init__(self, in_C, out_C):
        super(IDEMHSF, self).__init__()
        down_factor = in_C//out_C

        self.DWT = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.fuse_down_mul2 = BasicConv2d(2*in_C, out_C, 3, 1, 1)
        # self.res_main = DenseLayer4(in_C, in_C, down_factor=down_factor)
        # self.fuse_main = BasicConv2d(2in_C, 2in_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=1)
        self.fuse_main1 = BasicConv2d(out_C,out_C,kernel_size=1)
        self.fuse_main12 = BasicConv2d(4*out_C, out_C, kernel_size=1)
        self.fuse_main2 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main3 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.sfkong=nn.Softmax(dim=1)
        self.sf = nn.Softmax(dim=1)
        self.dconv=DeformConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.lska=LSKA(out_C,11)
        self.clska=CLSKA(out_C,11)
        # self.spanet=SPANet(out_C, inter_channels=None,sub_sample=True,adaptive=True,k=9)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.max_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)
        # self.spanet=SPANet(out_C, inter_channels=None,sub_sample=True,adaptive=True,k=9)
        self.sigmoid = nn.Sigmoid()
        self.FC = BasicDSConvConv2d(out_C, out_C, kernel_size=1)
        self.c2 = BasicDSConvConv2d(out_C // 2, out_C // 2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.c3 = BasicDSConvConv2d(out_C // 2, out_C // 2, kernel_size=3, stride=1, padding=3, dilation=3)
        self.PW = BasicDSConvConv2d(out_C, out_C // 2, kernel_size=1)
        # self.Graph = Graph_Attention_Union(out_C, out_C)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        rgb=self.fuse_main(rgb)
        depth = self.fuse_main(depth)
        rt=rgb - self.avg_pool(rgb)
        dt=depth-self.avg_pool(depth)
        dpoint=self.FC(dt)
        rpoint = self.FC(rt)
        dpoint=dpoint+rpoint
        rpoint=rpoint+dpoint
        wd=self.sigmoid(dpoint+self.max_pool(depth))
        wr = self.sigmoid(rpoint + self.max_pool(rgb))
        fuse=wd*rgb+wr*depth+rgb+depth
        chunks = torch.chunk(fuse, 2, dim=1)
        a=torch.cat([self.PW(fuse)*self.c2(chunks[0]),self.PW(fuse)*self.c3(chunks[1])],dim=1)*rgb+fuse##self.PW(rgbdepth)*
        feat=self.fuse_main3(a)
        return feat

class IDEMLFF(nn.Module):
    def __init__(self, in_C, out_C):
        super(IDEMLFF, self).__init__()
        down_factor = in_C//out_C

        self.DWT = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.fuse_down_mul2 = BasicConv2d(2*in_C, out_C, 3, 1, 1)
        # self.res_main = DenseLayer4(in_C, in_C, down_factor=down_factor)
        # self.fuse_main = BasicConv2d(2in_C, 2in_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=1)
        self.fuse_main1 = BasicConv2d(out_C,out_C,kernel_size=1)
        self.FC = BasicDSConvConv2d(out_C, out_C, kernel_size=1)
        # self.PW = BasicDSConvConv2d(out_C, out_C, kernel_size=1)
        self.fuse_main12 = BasicConv2d(2*out_C, out_C, kernel_size=1)
        self.fuse_main2 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main3 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.sfkong=nn.Softmax(dim=1)
        self.sf = nn.Softmax(dim=1)
        self.dconv=DeformConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.max_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)
        # self.spanet=SPANet(out_C, inter_channels=None,sub_sample=True,adaptive=True,k=9)
        self.sigmoid = nn.Sigmoid()
        self.c2=BasicDSConvConv2d(out_C//2, out_C//2, kernel_size=3, stride=1,padding=2,dilation=2)
        self.c3 = BasicDSConvConv2d(out_C // 2, out_C // 2, kernel_size=3, stride=1, padding=3,dilation=3)
        self.PW = BasicDSConvConv2d(out_C, out_C// 2, kernel_size=1)
        # self.c4 = BasicDSConvConv2d(out_C // 4, out_C // 4, kernel_size=3, stride=1,padding=4, dilation=4)
        # self.FCup = nn.Conv2d(out_C // 4, out_C, kernel_size=1)
        # self.Graph = Graph_Attention_Union(out_C, out_C)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        rgb1=self.fuse_main(rgb)
        depth1 = self.fuse_main(depth)
        rgbw=self.sf(self.FC(self.avg_pool(rgb1)+self.max_pool(rgb1)))
        depthw = self.sf(self.FC(self.avg_pool(depth1)+self.max_pool(depth1)))
        max_tensor= torch.max(rgbw, depthw)
        # max_tensor = torch.stack((rgbw, depthw))[max_index]
        rgbdep1=(max_tensor*rgb1)+(max_tensor*depth1)
        rgbdepth=rgb1+depth1+rgbdep1
        chunks = torch.chunk(rgbdepth, 2, dim=1)
        #a=torch.cat([self.c2(chunks[0]),self.c3(chunks[1])],dim=1)*rgb1+rgbdepth
        a=torch.cat([self.PW(rgbdepth)*self.c2(chunks[0]),self.PW(rgbdepth)*self.c3(chunks[1])],dim=1)*rgb1+rgbdepth##self.PW(rgbdepth)*
        feat=self.fuse_main3(a)
        # feat=rgb1+depth1
        return feat




class Resudiual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resudiual, self).__init__()
        self.conv = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.sigmoid(x1)
        out = x1*x
        return out

class CFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CFM, self).__init__()
        # self.conv = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.basicconv1 = BasicDSConvConv2d(in_planes=in_channel,out_planes=out_channel,kernel_size=1)
        self.basicconv4= BasicDSConvConv2d(in_planes=2*out_channel, out_planes=out_channel, kernel_size=1)
        self.basicconv2 = BasicDSConvConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.basicconv3 = BasicDSConvConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3,stride=1, padding=1)
        self.patch = PatchEmbeding(patch_size=1, in_channels=out_channel, embed_dim=out_channel, dropout=0.)
        self.ATTEN = Attention_split(dim=out_channel, heads=4, dim_head=16, dropout=0.)
        # self.basicconv1.to(self.device)
        project_out = not (4 == 1 and 16 == out_channel)
        inner_dim = 4 * 16
        self.scale = 16 ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_channel),
            # nn.Dropout(dropout=0.)
        ) if project_out else nn.Identity()
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            DSConv(32,32,1),
            nn.ReLU()
        )
        self.gam=Graph_Attention_Union(out_channel,out_channel)

    def forward(self, x,y):
        x1 = self.sigmoid(self.basicconv2(self.upsample1(self.basicconv1(x))+self.basicconv1(y)))
        x22=self.sigmoid(self.basicconv4(self.avg_pool(torch.cat([self.basicconv1(y),self.upsample1(self.basicconv1(x))],dim=1))))*self.basicconv4(torch.cat([self.basicconv1(y),self.upsample1(self.basicconv1(x))],dim=1))
        x2=x22*x1
        l=self.basicconv3(x2)

        Y=self.basicconv1(y)
        X=self.upsample1(self.basicconv1(x))
        Y_patch, Y_1, Y_2 = self.patch(Y)
        X_patch, X_1, X_2 = self.patch(X)
        X_q, X_k, X_v = self.ATTEN(X_patch)
        Y_q, Y_k, Y_v = self.ATTEN(Y_patch)
        dots_d = torch.matmul(X_q, X_k.transpose(-1, -2)) * self.scale
        attn_d = self.attend(dots_d)
        out_d = torch.matmul(attn_d, Y_v)
        out_d = rearrange(out_d, 'b h n d -> b n (h d)')
        d = self.to_out(out_d)
        feat=d
        B, n_patch, hidden = feat.size()
        h, w = int(X_1), int(X_2)
        xx2 = feat.permute(0, 2, 1)
        xx2 = xx2.contiguous().view(B, hidden, h, w)

        out = self.gam(l,xx2)
        return out


class Tdc3x3_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_1, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=1, padding=1)
        # self.conv2 = ODConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=1, padding=1)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        # x1 = x
        x2 = self.conv2(x)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_3, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        # x1 = x
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_5, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=4, padding=4)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        # x1 = x
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3,x4

class Tdc3x3_8(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_8, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=8, padding=8)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        # x1 = x
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3,x4

class BasicUpsample(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            # nn.Conv2d(32,32,kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)

class BasicUpsample_L(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample_L, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            # nn.Conv2d(128,32,kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)



class FDM(nn.Module):
    def __init__(self,):
        super(FDM, self).__init__()
        self.basicconv1 = BasicDSConvConv2d(in_planes=64,out_planes=32,kernel_size=1)
        self.basicconv6432 = BasicDSConvConv2d(in_planes=64, out_planes=32, kernel_size=1)
        self.basicconv2 = BasicDSConvConv2d(in_planes=32,out_planes=32,kernel_size=1)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            DSConv(32,32,1),
            nn.ReLU()
        )
        self.dnsample1 = nn.Sequential(
            nn.Upsample(scale_factor=0.5,mode='nearest'),
            DSConv(32,32,1),
            nn.ReLU()
        )
        self.basicconv3 = BasicDSConvConv2d(in_planes=32,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv4 = BasicDSConvConv2d(in_planes=64,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv11 = BasicDSConvConv2d(in_planes=32, out_planes=32, kernel_size=1)
        self.basicupsample16 = BasicUpsample(scale_factor=16)
        self.basicupsample8 = BasicUpsample(scale_factor=8)
        self.basicupsample4 = BasicUpsample(scale_factor=4)
        self.basicupsample2 = BasicUpsample(scale_factor=2)
        self.basicupsample1 = BasicUpsample(scale_factor=1)
        self.sg=nn.Sigmoid()

        self.reg_layer = nn.Sequential(
            DSConv(128,64,kernel_size=3,stride=2,padding=1),
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),#消融
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DSConv(64,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DSConv(32,16,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DSConv(16,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            )


    def forward(self,out_data_1,out_data_2,out_data_4,out_data_8):
        out_data_8 = self.basicconv1(out_data_8)

        out_data_4 = self.basicconv1(out_data_4)
        out_data_4 = out_data_4+self.upsample1(out_data_8)

        out_data_2 = out_data_2+self.upsample1(out_data_4)


        out_data_1 = self.upsample1(out_data_2)+self.basicconv6432(out_data_1)



        out_data_8 = self.basicupsample8(out_data_8)
        out_data_4 = self.basicupsample4(out_data_4)
        out_data_2 = self.basicupsample2(out_data_2)
        out_data_1 = self.basicupsample1(out_data_1)


        out_data = torch.cat([out_data_8, out_data_4, out_data_2, out_data_1], dim=1)

        out_data = self.reg_layer(out_data)


        return torch.abs(out_data)



class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output
