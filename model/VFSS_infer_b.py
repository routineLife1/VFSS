import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.core.Networks import build_network as VideoFlow
from model.MetricNet import MetricNet
from model.FeatureNet import FeatureNet
from model.FusionNet_b import GridNet
from model.softsplat import softsplat as warp

device = torch.device("cuda")


class Model:
    def __init__(self, mode='MOF'):
        if mode == 'BOF':  # 3
            from model.configs.sintel_submission import get_cfg
        elif mode == 'MOF':  # n
            from model.configs.multiframes_sintel_submission import get_cfg
            cfg = get_cfg()
            cfg.input_frames = 4
        self.mode = mode
        self.flownet = VideoFlow(cfg)
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
        self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path)))
        self.feat_ext.load_state_dict(torch.load('{}/feat.pkl'.format(path)))
        self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path)))

    def reuse(self, img0, img1, img2, img3, scale):
        feat11, feat12, feat13 = self.feat_ext(img1)
        feat21, feat22, feat23 = self.feat_ext(img2)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2, scale_factor=0.5, mode="bilinear", align_corners=False)
        img3 = F.interpolate(img3, scale_factor=0.5, mode="bilinear", align_corners=False)

        if scale != 1.0:
            if0 = F.interpolate(img0, scale_factor=scale, mode="bilinear", align_corners=False)
            if1 = F.interpolate(img1, scale_factor=scale, mode="bilinear", align_corners=False)
            if2 = F.interpolate(img2, scale_factor=scale, mode="bilinear", align_corners=False)
            if3 = F.interpolate(img3, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            if0 = img0
            if1 = img1
            if2 = img2
            if3 = img3

        frames = torch.stack((if0, if1, if2, if3)).permute(1, 0, 2, 3, 4)
        if self.mode == 'BOF':
            flow12 = self.flownet(frames[:, :-1, :, :, :], {})[0][0].unsqueeze(0)
            flow21 = self.flownet(frames[:, 1:, :, :, :], {})[0][1].unsqueeze(0)
        else:
            flow_x, _ = self.flownet(frames, {})
            flow12, flow21 = flow_x[0][0].unsqueeze(0), flow_x[0][3].unsqueeze(0)

        if scale != 1.0:
            flow12 = F.interpolate(flow12, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale
            flow21 = F.interpolate(flow21, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale

        metric0, metric1 = self.metricnet(img1, img2, flow12, flow21)

        return flow12, flow21, metric0, metric1, feat_ext0, feat_ext1

    def inference(self, img0, img1, img2, img3, reuse_things, timestep):  # 为了对齐输入留下img0, img3
        flow12, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][
            1], reuse_things[4][2]
        flow21, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][
            1], reuse_things[5][2]

        F1t = timestep * flow12
        F2t = (1 - timestep) * flow21

        Z1t = timestep * metric0
        Z2t = (1 - timestep) * metric1

        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        I1t = warp(img1, F1t, Z1t, strMode='soft')
        img2 = F.interpolate(img2, scale_factor=0.5, mode="bilinear", align_corners=False)
        I2t = warp(img2, F2t, Z2t, strMode='soft')

        feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

        out = self.fusionnet(torch.cat([img1, I1t, I2t, img2], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                             torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        return torch.clamp(out, 0, 1)
