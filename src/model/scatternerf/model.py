# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from random import random
from typing import *

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import src.model.scatternerf.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel


@gin.configurable()
class NeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 2,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, x, condition):

        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density_fog = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )
        
        raw_density = raw_density_fog[...,0]
        if self.num_density_channels == 2 : 
            raw_is_fog = raw_density_fog[...,1]

        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        if self.num_density_channels == 2 : 
            return raw_rgb, raw_density, raw_is_fog
        else:
            return raw_rgb, raw_density


@gin.configurable()
class ScatterNeRF(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False,

    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(ScatterNeRF, self).__init__()


        self.softmax = nn.Softmax(dim=0)
        self.rgb_activation = nn.Sigmoid()
        self.fog_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view, num_density_channels = 1 )
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view, num_density_channels = 1 )
        self.fog_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view, 
                                                        netdepth = 4,
                                                        netwidth =128,
                                                        netwidth_condition = 64,
                                                        skip_layer = 2,
                                                        num_density_channels = 1)


    def forward(self, rays, randomized, white_bkgd, near, far):

        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                mlp = self.coarse_mlp

            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(
                    bins=t_mids,
                    weights=weights_defog[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp

            samples_enc = helper.pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )
            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)

            raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc)
            raw_rgb_fog, raw_sigma_fog = self.fog_mlp(samples_enc, viewdirs_enc)

            if self.noise_std > 0 and randomized:
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std

            rgb = self.rgb_activation(raw_rgb)
            rgb_fog = self.rgb_activation(raw_rgb_fog)
            sigma = self.sigma_activation(raw_sigma)[...,None]
            sigma_fog= self.sigma_activation(raw_sigma_fog)[...,None]



            eps = 1e-6
            sigma_fogged = sigma + sigma_fog
            w1 = (sigma+eps)/(sigma_fogged+eps) #torch.concat( [(eps+torch.exp(sigma))/(eps+torch.exp(sigma) + torch.exp(sigma_fog))]*3 , dim = -1 )
            w2 = (sigma_fog+eps)/(sigma_fogged+eps) #torch.concat( [(eps+torch.exp(sigma_fog))/(eps+torch.exp(sigma) + torch.exp(sigma_fog))]*3, dim = -1 )

            rgb_fogged = rgb*w1 + rgb_fog*w2

            comp_rgb, acc, weights, alpha, delta_t = helper.volumetric_rendering(
                rgb_fogged,
                sigma + sigma_fog,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            sigma_defog = sigma
            assert( list(sigma_defog.shape) == list(sigma.shape) )
            comp_rgb_defog, _, weights_defog, alpha_defog, delta_t_defog = helper.volumetric_rendering(
                rgb,
                sigma_defog,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            depth = (weights * t_vals).sum(dim=-1)
            depth_defog = (weights_defog * t_vals).sum(dim=-1)

            alpha_detach = helper.get_alpha(
                sigma.detach() + sigma_fog ,
                t_vals,
                rays["rays_d"]
                )
            alpha_only_fog = helper.get_alpha(
                sigma_fog ,
                t_vals,
                rays["rays_d"]
                )                

            assert(list(alpha_only_fog.shape) == list(alpha.shape) )
            alpha_prod = alpha_only_fog * alpha

            ret.append((comp_rgb, acc, alpha, depth, {"rgb_points":rgb_fogged, "sigma_points":sigma_fogged, "t_vals":t_vals, 
                                                    "comp_rgb_defog":comp_rgb_defog, "depth_defog":depth_defog, 
                                                    "weights_defog":weights_defog, "depth":depth, 
                                                    "weights":weights, "alpha_defog":alpha_defog, "rgb_fog":rgb_fog, "sigma_defog":sigma_defog,
                                                    "alpha_detach":alpha_detach, "alpha_prod":alpha_prod } ))

        return ret


@gin.configurable()
class LitScatterNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitScatterNeRF, self).__init__()
        self.model = ScatterNeRF()

    def setup(self, stage: Optional[str] = None) -> None:
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def get_entropy_ray(self, alpha) : 
        sum_alpha = torch.sum(alpha, dim = 1, keepdim=True)
        prob = alpha / (sum_alpha + 1e-3)
        entropy = -1 * torch.sum(prob *torch.log(prob+1e-5), dim =1, keepdim=True)
        mask = (sum_alpha > 0.05).float()
        entropy_masked = entropy * mask 
        assert(list(entropy.shape) == list(mask.shape) )
        avg_entropy_masked = torch.sum(entropy_masked) / (torch.sum(mask) +1e-5)
        if False :
            print("sum_alpha",sum_alpha.shape)
            print("mask",mask)
            print("sum_alpha",sum_alpha)
            print("entropy", entropy.shape)
            print("entropy_masked", entropy_masked)
            print(avg_entropy_masked)
        return avg_entropy_masked, prob

    def depth_kl_loss(self, weights, t_vals, depth, delta_t, sigmasquare=0.1):
        i = torch.log(weights+0.001) * torch.exp(-torch.square(t_vals - depth)/(2*sigmasquare) ) * delta_t
        kl_loss = -1*torch.mean(i) / 10**7
        #print("kl_loss",kl_loss)
        return kl_loss 

    def training_step(self, batch, batch_idx):

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]

        # #Get entropy loss
        alpha_coarse = rendered_results[0][2]
        alpha_fine = rendered_results[1][2]
        flattened_alpha_fine = torch.flatten(alpha_fine)[None,...]
        flattened_alpha_coarse = torch.flatten(alpha_coarse)[None,...]
        alpha_coarse_detach = rendered_results[0][4]["alpha_detach"]
        alpha_fine_detach = rendered_results[1][4]["alpha_detach"]
        
        avg_entropy_masked_fine, prob_fine = self.get_entropy_ray(alpha_fine_detach)
        avg_entropy_masked_coarse, prob_coarse = self.get_entropy_ray(alpha_coarse_detach)
        tot_entropy = avg_entropy_masked_fine + avg_entropy_masked_coarse
        # #Get entropy defog loss
        weights_defog_coarse = rendered_results[0][4]["weights_defog"]
        weights_defog_fine = rendered_results[1][4]["weights_defog"]
        avg_entropy_masked_defog_fine, prob_fine = self.get_entropy_ray(weights_defog_fine)
        avg_entropy_masked_defog_coarse, prob_coarse = self.get_entropy_ray(weights_defog_coarse)
        tot_entropy_defog = avg_entropy_masked_defog_fine + avg_entropy_masked_defog_coarse


        # Depth sup loss
        sigma_points_coarse = rendered_results[0][4]["sigma_points"][:,:,0]
        sigma_points_fine = rendered_results[1][4]["sigma_points"][:,:,0]
        t_vals_coarse = rendered_results[0][4]["t_vals"]
        t_vals_fine = rendered_results[1][4]["t_vals"]
        rgb_points_coarse = rendered_results[0][4]["rgb_points"]
        rgb_points_fine = rendered_results[1][4]["rgb_points"]          



        #print("is_fog_coarse",is_fog_coarse.shape) #>>[1366, 129]
        #print("sigma_points_coarse", sigma_points_coarse.shape) #>>[1366, 129]


        assert( rendered_results[1][4]["depth"][...,None].shape == batch["target_depth"].shape)        
        assert(list(torch.square( batch["target_depth"]   - rendered_results[0][4]["depth_defog"][...,None]).shape) ==  list((batch["target_depth"]>0).shape ) )
        assert(batch["unc_depth"][...,None].shape == batch["target_depth"].shape )
        is_ok_depth = 1- batch["unc_depth"][...,None]
        loss_depth = torch.sum( torch.square( batch["target_depth"]   - rendered_results[0][4]["depth_defog"][...,None]) *(batch["target_depth"]>0)*is_ok_depth  ) / (torch.sum(batch["target_depth"]>0) +1e-5)
        loss_depth += torch.sum(torch.square( batch["target_depth"]  - rendered_results[1][4]["depth_defog"][...,None] )*(batch["target_depth"]>0)*is_ok_depth  ) / (torch.sum(batch["target_depth"]>0) +1e-5)
        loss_depth *= 0.1

        defogged_img_coarse = rendered_results[0][4]["comp_rgb_defog"] 
        defogged_img_fine = rendered_results[1][4]["comp_rgb_defog"] 
        rgb_fog_coarse = rendered_results[0][4]["rgb_fog"] 
        rgb_fog_fine = rendered_results[1][4]["rgb_fog"] 

        is_ok_color = torch.stack( [torch.sum(batch["target_defogged"], dim = -1) > 0.1 ]*3, dim = 1)
        is_CLEAR =    torch.stack( [torch.sum(batch["target"], dim = -1) < 0.01 ]*3, dim = 1)
        unc3 = torch.stack( [batch["uncertainity"] ]*3, dim = 1)
        assert(list(is_ok_color.shape) == list(unc3.shape) )


        COLOR_FOG_coarse = torch.stack( [ batch["target_airlight"] ]*rgb_fog_coarse.shape[1],  dim = 1) 
        COLOR_FOG_fine = torch.stack( [ batch["target_airlight"] ]*rgb_fog_fine.shape[1],  dim = 1) 
        is_ok_depth_coarse = torch.concat( [batch["target_depth"]>0 ]*rgb_fog_coarse.shape[1], dim =1 )
        is_ok_depth_fine = torch.concat( [batch["target_depth"]>0 ]*rgb_fog_fine.shape[1], dim =1 )


        assert(list(rgb_fog_coarse.shape) == list(COLOR_FOG_coarse.shape))
        assert(list(rgb_fog_fine.shape) == list(COLOR_FOG_fine.shape))
        assert(list(torch.sum(torch.square(rgb_fog_coarse - COLOR_FOG_coarse), dim=-1).shape) == list(is_ok_depth_coarse.shape) )
        loss_fog_color = torch.sum( torch.sum(torch.square(rgb_fog_coarse - COLOR_FOG_coarse), dim=-1) *is_ok_depth_coarse  ) / (torch.sum(is_ok_depth_coarse) +1e-5)
        loss_fog_color += torch.sum( torch.sum(torch.square(rgb_fog_fine - COLOR_FOG_fine), dim=-1) *is_ok_depth_fine  ) / (torch.sum(is_ok_depth_fine) +1e-5)
        loss_fog_color *= 0.5


        target = batch["target"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)

        loss = 2.5*loss1 + 2.5*loss0 + loss_depth*0.1  +0.5*(0.0001*tot_entropy_defog - 0.00001*tot_entropy) + 0.5*loss_fog_color

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)
        
        self.log("loss_img", 2.5*loss1 + 2.5*loss0, on_step=True, prog_bar=True, logger=True)
        self.log("loss_depth", loss_depth*0.1 , on_step=True, prog_bar=True, logger=True)
        self.log("tot_entropy_defog", 0.5*0.0001*tot_entropy_defog, on_step=True, prog_bar=True, logger=True)
        self.log("tot_entropy", 0.5*0.00001*tot_entropy, on_step=True, prog_bar=True, logger=True)
        self.log("loss_fog_color", 0.5*loss_fog_color, on_step=True, prog_bar=True, logger=True)
        
        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True )
        self.log("train/loss", loss, on_step=True)

        return loss

    def render_rays(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far
        )
        rgb_fine = rendered_results[1][0]
        target = batch["target"]
        target_depth = batch["target_depth"]
        ret["target"] = target
        ret["target_depth"] = target_depth
        ret["rgb"] = rgb_fine
        ret["depth"] = rendered_results[1][3]
        ret["comp_rgb_defog"] = rendered_results[1][4]["comp_rgb_defog"]
        ret["depth_defog"] = rendered_results[1][4]["depth_defog"]
        ret["target_defogged"] = batch["target_defogged"]

        #debug
        if False:
            ret["rgb_points"] = rendered_results[1][4]["rgb_points"]
            ret["sigma_points"] = rendered_results[1][4]["sigma_points"]
            ret["sigma_defog"] = rendered_results[1][4]["sigma_defog"]
            ret["t_vals"] = rendered_results[1][4]["t_vals"]
            ret["weights"] = rendered_results[1][4]["weights"]
            ret["weights_defog"] = rendered_results[1][4]["weights_defog"]
            ret["alpha"] = rendered_results[1][2]
            ret["alpha_defog"] = rendered_results[1][4]["alpha_defog"]

        # np.save("depth", rendered_results[1][4]["depth"].detach().cpu().numpy() )
        # np.save("rgb_points", rendered_results[1][4]["rgb_points"].detach().cpu().numpy() )
        # np.save("sigma_points", rendered_results[1][4]["sigma_points"].detach().cpu().numpy() )
        # np.save("t_vals", rendered_results[1][4]["t_vals"].detach().cpu().numpy() )    
        # np.save("target_depth", target_depth.detach().cpu().numpy() )    
        #asdasd

        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        comp_rgb_defog = self.alter_gather_cat(outputs, "comp_rgb_defog", all_image_sizes)
        target_defogged = self.alter_gather_cat(outputs, "target_defogged", all_image_sizes)
        

        depths = self.alter_gather_cat(outputs, "depth", all_image_sizes, channels_number=1)

        defog_depths = self.alter_gather_cat(outputs, "depth_defog", all_image_sizes, channels_number=1)
        # target_depth = self.alter_gather_cat(outputs, "target_depth", all_image_sizes, channels_number=1)
        psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)

        #debug:
        if False:
            rgb_points = self.alter_gather_cat_raw(outputs, "rgb_points", all_image_sizes)
            sigma_points = self.alter_gather_cat_raw(outputs, "sigma_points", all_image_sizes)
            t_vals =      self.alter_gather_cat_raw(outputs, "t_vals", all_image_sizes )
            sigma_defog = self.alter_gather_cat_raw(outputs, "sigma_defog", all_image_sizes )
            weights = self.alter_gather_cat_raw(outputs, "weights", all_image_sizes )
            weights_defog = self.alter_gather_cat_raw(outputs, "weights_defog", all_image_sizes )
            alpha = self.alter_gather_cat_raw(outputs, "alpha", all_image_sizes )
            alpha_defog = self.alter_gather_cat_raw(outputs, "alpha_defog", all_image_sizes )

            # # print(rgb_points)
            np.save("rgb_points", rgb_points[0].detach().cpu().numpy() )
            # print(len(rgb_points))
            np.save("sigma_points", sigma_points[0].detach().cpu().numpy() )
            np.save("sigma_defog", sigma_defog[0].detach().cpu().numpy() )
            np.save("weights", weights[0].detach().cpu().numpy() )
            np.save("weights_defog", weights_defog[0].detach().cpu().numpy() )
            np.save("alpha_defog", alpha_defog[0].detach().cpu().numpy() )
            np.save("alpha", alpha[0].detach().cpu().numpy() )
            

            np.save("t_vals", t_vals[0].detach().cpu().numpy() )
            asdasd
        

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)
            #store_image.store_image(image_dir, targets, gt=True)
            store_image.store_image(image_dir, comp_rgb_defog, appendix="gt")
            #store_image.store_image(image_dir, target_defogged, appendix="defog_target" )
            store_image.store_depth(image_dir, depths)
            #store_image.store_depth(image_dir, target_depth, appendix="_gt")
            store_image.store_depth(image_dir, defog_depths, appendix="_defog")

            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips
