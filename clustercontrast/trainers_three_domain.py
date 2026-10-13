# -*- coding: utf-8 -*-
"""Trainer for AG-ReID A_far/A_near/Ground Stage 2.

There is no new loss family here.  The trainer reuses the original independent
memory losses, BGM cross-memory losses, ALL-memory loss, and EMA model update.
Only ``loss_ema`` is excluded from optimization, matching the no-EMA-loss
ablation requested as the experimental base.
"""

from __future__ import absolute_import, print_function

import time

import torch

from .three_domain import (compose_three_domain_all_loss,
                           compose_three_domain_primary_loss)
from .trainers import ClusterContrastTrainer_PCLMP
from .utils.meters import AverageMeter


class ClusterContrastTrainer_PCLMP_ThreeDomain(ClusterContrastTrainer_PCLMP):
    def __init__(self, encoder, encoder_ema, memory=None, far_weight=0.70,
                 chain_cross_weight=0.25):
        super(ClusterContrastTrainer_PCLMP_ThreeDomain, self).__init__(
            encoder, encoder_ema, memory=memory)
        self.memory_far = memory
        self.memory_near = memory
        self.memory_ground = memory
        self.far_weight = float(far_weight)
        self.chain_cross_weight = float(chain_cross_weight)

    @staticmethod
    def _mapped_labels(labels, mapping):
        return torch.tensor(
            [mapping[int(label)] for label in labels.detach().cpu().tolist()],
            dtype=torch.long, device=labels.device)

    def train(self, epoch, data_loader_far, data_loader_near,
              data_loader_ground, data_loader_all_far,
              data_loader_all_near, data_loader_all_ground, optimizer,
              print_freq=10, train_iters=400, far_to_near=None,
              near_to_far=None, near_to_ground=None, ground_to_near=None):
        self.encoder.train()
        self.encoder_ema.train()

        required_maps = (far_to_near, near_to_far,
                         near_to_ground, ground_to_near)
        if any(mapping is None or not mapping for mapping in required_maps):
            raise ValueError("all four chain BGM mappings must be non-empty")

        print("============================================")
        print("EXPERIMENT: THREE-DOMAIN STAGE 2 (NO EMA LOSS)")
        print("Domains: A_far <-> A_near <-> Ground")
        print("A_far loss weight: {:.6f}".format(self.far_weight))
        print("A_near loss weight: {:.6f}".format(1.0 - self.far_weight))
        print("Chain cross weight: {:.6f}".format(self.chain_cross_weight))
        print("EMA loss included: False (monitor only)")
        print("EMA model update enabled: True")
        print("ALL memory optimization enabled: True")
        print("Optimizer steps per iteration: 2")
        print("============================================")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        primary_losses = AverageMeter()
        all_losses = AverageMeter()
        end = time.time()

        for iteration in range(train_iters):
            batch_far = data_loader_far.next()
            batch_near = data_loader_near.next()
            batch_ground = data_loader_ground.next()
            data_time.update(time.time() - end)

            inputs_far, labels_far, _ = self._parse_data_ir(batch_far)
            inputs_near, labels_near, _ = self._parse_data_ir(batch_near)
            (inputs_ground, inputs_ground_aug, labels_ground,
             _) = self._parse_data_rgb(batch_ground)

            far_batch_size = inputs_far.size(0)
            inputs_aerial = torch.cat((inputs_far, inputs_near), dim=0)
            labels_aerial = torch.cat((labels_far, labels_near), dim=0)
            inputs_ground = torch.cat(
                (inputs_ground, inputs_ground_aug), dim=0)
            labels_ground = torch.cat(
                (labels_ground, labels_ground), dim=0)

            (_, features_ground, features_aerial, labels_ground,
             labels_aerial, _, _) = self._forward(
                inputs_ground, inputs_aerial, label_1=labels_ground,
                label_2=labels_aerial, modal=0)
            features_far = features_aerial[:far_batch_size]
            features_near = features_aerial[far_batch_size:]

            loss_far = self.memory_far(features_far, labels_far)
            loss_near = self.memory_near(features_near, labels_near)
            loss_ground = self.memory_ground(
                features_ground, labels_ground)

            if epoch % 2 == 0:
                loss_far_near = self.memory_near(
                    features_far,
                    self._mapped_labels(labels_far, far_to_near))
                loss_near_ground = self.memory_ground(
                    features_near,
                    self._mapped_labels(labels_near, near_to_ground))
                chain_direction = "Far->Near, Near->Ground"
            else:
                loss_far_near = self.memory_far(
                    features_near,
                    self._mapped_labels(labels_near, near_to_far))
                loss_near_ground = self.memory_near(
                    features_ground,
                    self._mapped_labels(labels_ground, ground_to_near))
                chain_direction = "Near->Far, Ground->Near"

            # Keep the no-EMA-loss baseline's monitor for a directly comparable
            # log, but do not add it to the optimized objective.
            with torch.no_grad():
                (_, ema_ground, ema_aerial, ema_ground_labels,
                 _, _, _) = self._forward_ema(
                    inputs_ground, inputs_aerial, label_1=labels_ground,
                    label_2=labels_aerial, modal=0)
                ema_far = ema_aerial[:far_batch_size]
                ema_near = ema_aerial[far_batch_size:]
                loss_ema_far = self.memory_far(
                    ema_far, labels_far, model_name="encoder_ema")
                loss_ema_near = self.memory_near(
                    ema_near, labels_near, model_name="encoder_ema")
                loss_ema_ground = self.memory_ground(
                    ema_ground, ema_ground_labels,
                    model_name="encoder_ema")
                loss_ema_monitor = (
                    self.far_weight * loss_ema_far
                    + (1.0 - self.far_weight) * loss_ema_near
                    + loss_ema_ground)

            primary_loss = compose_three_domain_primary_loss(
                loss_far, loss_near, loss_ground,
                loss_far_near, loss_near_ground,
                self.far_weight, self.chain_cross_weight)
            optimizer.zero_grad()
            primary_loss.backward()
            optimizer.step()
            primary_losses.update(primary_loss.item())

            all_far, all_far_labels, _ = self._parse_data_ir(
                data_loader_all_far.next())
            all_near, all_near_labels, _ = self._parse_data_ir(
                data_loader_all_near.next())
            (all_ground, all_ground_aug, all_ground_labels,
             _) = self._parse_data_rgb(data_loader_all_ground.next())

            all_far_batch_size = all_far.size(0)
            all_aerial = torch.cat((all_far, all_near), dim=0)
            all_aerial_labels = torch.cat(
                (all_far_labels, all_near_labels), dim=0)
            all_ground = torch.cat(
                (all_ground, all_ground_aug), dim=0)
            all_ground_labels = torch.cat(
                (all_ground_labels, all_ground_labels), dim=0)

            (_, all_ground_features, all_aerial_features,
             all_ground_labels, all_aerial_labels, _, _) = self._forward(
                all_ground, all_aerial, label_1=all_ground_labels,
                label_2=all_aerial_labels, modal=0)
            all_far_features = all_aerial_features[:all_far_batch_size]
            all_near_features = all_aerial_features[all_far_batch_size:]

            loss_all_far = self.memory_all(
                all_far_features, all_far_labels)
            loss_all_near = self.memory_all(
                all_near_features, all_near_labels)
            loss_all_ground = self.memory_all(
                all_ground_features, all_ground_labels)
            all_loss = compose_three_domain_all_loss(
                loss_all_far, loss_all_near, loss_all_ground,
                self.far_weight)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            all_losses.update(all_loss.item())

            self._update_ema_variables(
                self.encoder, self.encoder_ema, 0.999)

            batch_time.update(time.time() - end)
            end = time.time()
            if (iteration + 1) % print_freq == 0:
                print(
                    "Epoch: [{}][{}/{}]\t"
                    "Time {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\t"
                    "Loss1 {:.3f} ({:.3f})\t"
                    "Far {:.3f}\tNear {:.3f}\tGround {:.3f}\t"
                    "Cross F-N {:.3f}\tCross N-G {:.3f}\t"
                    "Direction {}\tEMA monitor EXCLUDED {:.3f}\t"
                    "Loss2 ALL {:.3f} ({:.3f})\t"
                    "ALL Far {:.3f}\tALL Near {:.3f}\tALL Ground {:.3f}".format(
                        epoch, iteration + 1, train_iters,
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        primary_losses.val, primary_losses.avg,
                        loss_far, loss_near, loss_ground,
                        loss_far_near, loss_near_ground,
                        chain_direction, loss_ema_monitor,
                        all_losses.val, all_losses.avg,
                        loss_all_far, loss_all_near, loss_all_ground))
