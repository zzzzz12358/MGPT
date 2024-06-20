import random

import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.MHA = MultiHeadAttention(n_head=3, d_model=512, d_k=512, d_v=512)
        self.MLP = MultiLayerPerceptron(dim=512, mlp_dim=1024)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.iter = 3
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        # Dilate the mask
        kernel = np.ones((3, 3), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_, kernel, iterations=1)  # (256, 256)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots,
                                                               np.shape(supp_periphery_mask)[0],
                                                               np.shape(supp_periphery_mask)[1]))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots,
                                                           np.shape(supp_dilated_mask)[0],
                                                           np.shape(supp_dilated_mask)[1]))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])

        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        # Compute loss #
        periphery_loss = torch.zeros(1).to(self.device)
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        qry_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            # MGP Module
            fg_partition_prototypes = [[self.compute_multiple_grid_prototypes(
                 supp_fts[[epi], way, shot], supp_mask[[epi], way, shot],kernel_l = [20,18,16,14,12,10,6,8,4,2])
                for shot in range(self.n_shots)] for way in range(self.n_ways)]

            # calculate coarse query prototype
            supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]

            fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

            # Dilated region prototypes #
            supp_fts_dilated = [[self.getFeatures(supp_fts[[epi], way, shot], supp_dilated_mask[[epi], way, shot])
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_prototypes_dilated = self.getPrototype(supp_fts_dilated)

            # Segment periphery region with support images
            supp_pred_object = torch.stack([self.getPred(supp_fts[epi][way], fg_prototypes[way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1)   # N x Wa x H' x W'
            supp_pred_object = F.interpolate(supp_pred_object, size=img_size, mode='bilinear', align_corners=True)
            # supp_pred_object: (1, 1, 256, 256)

            supp_pred_dilated = torch.stack([self.getPred(supp_fts[epi][way], fg_prototypes_dilated[way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1)   # N x Wa x H' x W'
            supp_pred_dilated = F.interpolate(supp_pred_dilated, size=img_size, mode='bilinear', align_corners=True)
            # supp_pred_dilated: (1, 1, 256, 256)

            # Prediction of periphery region
            pred_periphery = supp_pred_dilated - supp_pred_object
            pred_periphery = torch.cat((1.0 - pred_periphery, pred_periphery), dim=1)
            # pred_periphery: (1, 2, 256, 256)  B x C x H x W
            label_periphery = torch.full_like(supp_periphery_mask[epi][0][0], 255, device=supp_periphery_mask.device)
            label_periphery[supp_periphery_mask[epi][0][0] == 1] = 1
            label_periphery[supp_periphery_mask[epi][0][0] == 0] = 0
            # label_periphery: (256, 256)  H x W

            # Compute periphery loss
            eps_ = torch.finfo(torch.float32).eps
            log_prob_ = torch.log(torch.clamp(pred_periphery, eps_, 1 - eps_))
            periphery_loss += self.criterion(log_prob_, label_periphery[None, ...].long()) / self.n_shots / self.n_ways

            # GQPG module
            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

            qry_prototype_coarse = self.compute_multiple_grid_prototypes(qry_fts[epi], qry_pred[epi],kernel_l=[20,16,12,8,4])

            # # The first MBATE block
            for i in range(self.iter):
                fg_partition_prototypes = [[self.MBATE(fg_partition_prototypes[way][shot][epi], qry_prototype_coarse[epi])
                                            for shot in range(self.n_shots)] for way in range(self.n_ways)]

                supp_proto = [[torch.mean(fg_partition_prototypes[way][shot], dim=1) + fg_prototypes[way] for shot in range(self.n_shots)]
                              for way in range(self.n_ways)]

                # GQPG module
                qry_pred_coarse = torch.stack(
                    [self.getPred(qry_fts[epi], supp_proto[way][epi], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)

                qry_prototype_coarse = self.compute_multiple_grid_prototypes(qry_fts[epi], qry_pred_coarse[epi],kernel_l=[20,16,12,8,4])

            # Get query predictions #

            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], supp_proto[way][epi], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

            # Combine predictions of different feature maps #
            qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

            outputs.append(preds)

            if train:
                align_loss_epi = self.alignLoss(supp_fts[epi], qry_fts[epi], preds, supp_mask[epi])
                align_loss += align_loss_epi
            if train:
                proto_mse_loss_epi = self.proto_mse(qry_fts[epi], preds, supp_mask[epi], fg_prototypes)
                mse_loss += proto_mse_loss_epi
            if train:
                qry_fts_ = [[self.getFeatures(qry_fts[epi], qry_mask)]]
                qry_prototypes = self.getPrototype(qry_fts_)
                qry_pred = self.getPred(qry_fts[epi], qry_prototypes[epi], self.thresh_pred[epi])

                qry_pred = F.interpolate(qry_pred[None, ...], size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - qry_pred, qry_pred), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                qry_loss += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, periphery_loss / supp_bs, align_loss / supp_bs, mse_loss / supp_bs, qry_loss / supp_bs

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def compute_multiple_grid_prototypes(self, sup_fts, sup_fg,kernel_l = [2,4,8]):
        """

        Parameters
        ----------
        fg_num: int
            Foreground partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        fg_proto: torch.Tensor
            [B, k, C], where k is the number of foreground proxies

        """

        B, C, h, w = sup_fts.shape  # B=1, C=512
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        # fg_mask = fg_mask.squeeze(0).bool()  # [B, h, w] --> bool
        batch_fg_protos = []

        for b in range(B):
            protos = []
            for kernel in kernel_l:
                padding = (kernel - (h % kernel))% kernel
                n_sup_fts = F.conv2d(F.pad(sup_fts * fg_mask,(0,padding,0,padding)),
                                     weight=torch.ones((512, 1, kernel, kernel), requires_grad=False).cuda(),
                                     stride=kernel,groups=512)
                y_sum = F.conv2d(F.pad(fg_mask,(0,padding,0,padding)),
                                 weight=torch.ones((1, 1, kernel, kernel),requires_grad=False).cuda(),
                                 stride=kernel)
                n_sup_fts = n_sup_fts / (y_sum + 1e-8)

                sup_nshot = sup_fts.shape[0]

                n_sup_fts = n_sup_fts.view(sup_nshot, C, -1).permute(0, 2, 1).unsqueeze(0)
                n_sup_fts = n_sup_fts.reshape(1, -1, C).unsqueeze(0)

                y_sum = y_sum.flatten(2)
                proto = n_sup_fts[y_sum > 0]  # npro,  # npro, nc

                protos.append(proto)

            protos = torch.cat(protos)
            glb_proto = torch.sum(sup_fts * fg_mask, dim=(-1, -2))/(fg_mask.sum(dim=(-1, -2)) + 1e-5)
            protos = torch.cat([protos, glb_proto])

            batch_fg_protos.append(protos)


        fg_proto = torch.stack(batch_fg_protos, dim=0) # [B, k, C]

        return fg_proto

    def MBATE(self, fg_prototypes, qry_prototype_coarse):

        L = fg_prototypes.shape[0]

        # S&W module
        A = torch.mm(fg_prototypes, qry_prototype_coarse.t())
        kc = ((A.min() + A.mean()) / 2).floor()

        if A is not None:
            S = torch.zeros(A.size(), dtype=torch.float).cuda()
            S[A < kc] = -10000.0

        A = torch.softmax((A + S), dim=0)
        A = torch.mm(A, qry_prototype_coarse)

        A = self.layer_norm(A + fg_prototypes)
        A = torch.cat([A, self.layer_norm(qry_prototype_coarse)])

        # rest Transformer operation
        T = self.MHA(A.unsqueeze(0), A.unsqueeze(0), A.unsqueeze(0))
        T = self.MLP(T)
        T = T[:,:L]

        return T

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)


                # Combine predictions of different feature maps
                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def proto_mse(self, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]

                fg_prototypes = self.getPrototype(qry_fts_)

                fg_prototypes_ = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0)
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes, dim=0), dim=0)

                # Combine prototypes from different scales
                # fg_prototypes = self.alpha * fg_prototypes[way]
                # fg_prototypes = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0) / torch.sum(self.alpha)
                # supp_prototypes_ = [self.alpha[n] * supp_prototypes[n][way] for n in range(len(supp_fts))]
                # supp_prototypes_ = torch.sum(torch.stack(supp_prototypes_, dim=0), dim=0) / torch.sum(self.alpha)

                # Compute the MSE loss

                loss_sim += self.criterion_MSE(fg_prototypes_, supp_prototypes_)

        return loss_sim







