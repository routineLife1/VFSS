import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
from model.core.Networks import build_network as VideoFlow
from model.MetricNet import MetricNet
from model.FeatureNet import FeatureNet
from model.FusionNet_b import GridNet
from model.softsplat import softsplat as warp
import torch.nn.functional as F
from model.loss import *

from model.lpips import LPIPS

device = torch.device("cuda")


class Model:
    def __init__(self, local_rank=-1, mode='MOF'):
        if mode == 'BOF':  # 3
            from model.configs.sintel_submission import get_cfg
        elif mode == 'MOF':  # n
            from model.configs.multiframes_sintel_submission import get_cfg
            cfg = get_cfg()
            cfg.input_frames = 4
        self.flownet = VideoFlow(cfg)
        self.mode = mode
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()
        self.device()
        # self.optimG = AdamW(self.fusionnet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.optimG = AdamW(itertools.chain(
            self.metricnet.parameters(),
            self.feat_ext.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-4)
        self.l1_loss = Charbonnier_L1().to(device)
        self.lpips = LPIPS(net='vgg').to(device)
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        self.flownet.eval()
        self.metricnet.train()
        self.feat_ext.train()
        self.fusionnet.train()

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

    def load_model(self, path, rank=-1):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if self.mode == 'BOF':
            self.flownet.load_state_dict(convert(torch.load('{}/BOF_sintel.pth'.format(path))))
        else:
            self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
        self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path)))
        self.feat_ext.load_state_dict(torch.load('{}/feat.pkl'.format(path)))
        self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path)))

    def save_model(self, path, epoch=0, step=0, rank=0):
        if epoch == 0:
            torch.save(self.flownet.state_dict(), f'{path}/flownet.pkl')
            torch.save(self.metricnet.state_dict(), f'{path}/metric.pkl')
            torch.save(self.feat_ext.state_dict(), f'{path}/feat.pkl')
            torch.save(self.fusionnet.state_dict(), f'{path}/fusionnet.pkl')
        else:
            torch.save(self.flownet.state_dict(), f'{path}/flownet-e{epoch}s{step}.pkl')
            torch.save(self.metricnet.state_dict(), f'{path}/metric-e{epoch}s{step}.pkl')
            torch.save(self.feat_ext.state_dict(), f'{path}/feat-e{epoch}s{step}.pkl')
            torch.save(self.fusionnet.state_dict(), f'{path}/fusionnet-e{epoch}s{step}.pkl')

    def inference(self, img0, img1, img2, img3, timestep, simple_color_aug):
        # img0, img1, img2, img3 = simple_color_aug.augment(img0), simple_color_aug.augment(img1), simple_color_aug.augment(img2), simple_color_aug.augment(img3)
        # get flow only
        with torch.no_grad():
            imgs_chunks = [torch.chunk(imgx, chunks=1, dim=0) for imgx in [img0, img1, img2, img3]]

            flow1_chunks = list()
            flow2_chunks = list()
            for s in range(1):
                frames = torch.stack(
                    (imgs_chunks[0][s], imgs_chunks[1][s], imgs_chunks[2][s], imgs_chunks[3][s])).permute(1, 0, 2, 3, 4)
                if self.mode == 'BOF':
                    flow_12 = self.flownet(frames[:, :-1, :, :, :], {})[:, 0, :, :, :]
                    flow_21 = self.flownet(frames[:, 1:, :, :, :], {})[:, 1, :, :, :]
                else:
                    flow_x, _ = self.flownet(frames, {})
                    flow_12, flow_21 = flow_x[:, 0, :, :, :], flow_x[:, 3, :, :, :]

                flow1_chunks.append(flow_12)
                flow2_chunks.append(flow_21)
            flow12 = torch.cat(flow1_chunks, dim=0)
            flow21 = torch.cat(flow2_chunks, dim=0)

            flow12 = F.interpolate(flow12, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
            flow21 = F.interpolate(flow21, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        img1, img2 = simple_color_aug.augment(img1), simple_color_aug.augment(img2)

        img1s = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        img2s = F.interpolate(img2, scale_factor=0.5, mode="bilinear", align_corners=False)

        with torch.autocast(device_type='cuda', dtype=torch.float16):  # only non-flow opponent applied autocast
            metric0, metric1 = self.metricnet(img1s, img2s, flow12, flow21)

            feat11, feat12, feat13 = self.feat_ext(img1)
            feat21, feat22, feat23 = self.feat_ext(img2)

            F1t = timestep * flow12
            F2t = (1 - timestep) * flow21

            Z1t = timestep * metric0
            Z2t = (1 - timestep) * metric1

            I1t = warp(img1s, F1t, Z1t, strMode='soft')
            I2t = warp(img2s, F2t, Z2t, strMode='soft')

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

            merged = self.fusionnet(torch.cat([img1s, I1t, I2t, img2s], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                                    torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

            merged = simple_color_aug.reverse_augment(merged)

        return flow12, flow21, metric0, metric1, merged

    def update(self, imgs, gt, learning_rate=0, training=True, timestep=0.5, step=0, spe=1136):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0, img1, img2, img3 = imgs[:, :3], imgs[:, 3:6], imgs[:, 6:9], imgs[:, 9:]

        if training:
            self.train()
        else:
            self.eval()

        accum_iter = 1  # no use accumulator (chunks)

        simple_color_aug = SimpleColorAugmentation(enable=True)

        flow12, flow21, metric0, metric1, merged = self.inference(img0, img1, img2, img3, timestep, simple_color_aug)
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            loss_l1 = self.l1_loss(merged - gt)

            loss_lpips = self.lpips.forward(torch.clamp(merged, 0, 1), gt).mean()

            # merged_chunks = torch.chunk(merged, chunks=2, dim=0)
            # gt_chunks = torch.chunk(gt, chunks=2, dim=0)
            # loss_lpips_chunks = list()
            # for s in range(2):
            #     lpips_loss = self.lpips.forward(torch.clamp(merged_chunks[s], 0, 1), gt_chunks[s])
            #     loss_lpips_chunks.append(lpips_loss)
            # loss_lpips = torch.cat(loss_lpips_chunks, dim=0).mean()

        if training:
            loss_G = (loss_l1 + loss_lpips) / accum_iter

            self.scaler.scale(loss_G).backward()
            # if ((step + 1) % accum_iter == 0) or ((step + 1) % spe == 0):
            self.scaler.step(self.optimG)
            self.scaler.update()
            self.optimG.zero_grad(set_to_none=True)

        return merged, torch.cat((flow12, flow21), 1), metric0, metric1, loss_l1, loss_lpips


class SimpleColorAugmentation:
    def __init__(self, enable=True) -> None:
        self.seed = random.uniform(0, 1)
        if self.seed < 0.167:
            self.swap = [2, 1, 0]  # swap 1,3
            self.reverse_swap = [2, 1, 0]
        elif 0.167 < self.seed < 0.333:
            self.swap = [2, 0, 1]
            self.reverse_swap = [1, 2, 0]
        elif 0.333 < self.seed < 0.5:
            self.swap = [1, 2, 0]
            self.reverse_swap = [2, 0, 1]
        elif 0.5 < self.seed < 0.667:
            self.swap = [1, 0, 2]
            self.reverse_swap = [1, 0, 2]
        elif 0.667 < self.seed < 0.833:
            self.swap = [0, 2, 1]
            self.reverse_swap = [0, 2, 1]
        else:
            self.swap = [0, 1, 2]
            self.reverse_swap = [0, 1, 2]
        if not enable:
            self.swap = [0, 1, 2]  # no swap
            self.reverse_swap = self.swap
        pass

    def augment(self, img):
        """
        param: img, torch tensor, CHW
        """
        img = img[:, self.swap, :, :]
        return img

    def reverse_augment(self, img):
        img = img[:, self.reverse_swap, :, :]
        return img
