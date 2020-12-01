import pytorch_lightning as pl
from argparse import Namespace
from utils.utils import get_batch_k
import numpy as np
from omegaconf import DictConfig
import torch.nn as nn
from utils.losses import l2_loss, GANLoss, cal_ade, cal_fde, crashIntoWall
from .get_model import get_model
import torch
from data import TrajectoryDataset, seq_collate
from torch.utils.data import DataLoader
from utils.radam import RAdam
from utils.utils import re_im

from utils.visualize import visualize_traj_probabilities
import os
import itertools
import psutil




class TrajPredictor(pl.LightningModule):
	"""
	PyTorch Lightning Module for training GOAL GAN
	Hyperparamters of of training are set in 'config/training/training.yaml'
	and explained in 'HYPERPARAMETERS.md'
	"""
	def __init__(self, hparams: DictConfig = None, args: Namespace = None, loss_fns = None,):
		super().__init__()

		self.args = args
		self.hparams = hparams
		self.generator, self.discriminator = get_model(self.hparams)
		print(self.generator)
		print(self.discriminator)
		# init loss functions
		self.loss_fns = loss_fns if loss_fns else {'L2': l2_loss, #L2 loss
												   'ADV': GANLoss(hparams.gan_mode), # adversarial Loss
												   'G': l2_loss, # goal achievement loss
												   'GCE': nn.CrossEntropyLoss() }# Goal Cross Entropy loss
		# init loss weights

		self.loss_weights =  {'L2': hparams.w_L2,
							   'ADV': hparams.w_ADV,  # adversarial Loss
							   'G': hparams.w_G,  # goal achievement loss
							   'GCE': hparams.w_GCE  } # Goal Cross Entropy loss

		self.current_batch_idx = -1
		self.plot_val = True


		if self.hparams.batch_size_scheduler:
			self.batch_size = self.hparams.batch_size_scheduler
		else:
			self.batch_size = cfg.batch_size

	"""########## DATE PREPARATION ##########"""
	def setupData(self):
		# called only on 1 GPU
		self.train_dset = TrajectoryDataset( phase = "train",  **self.hparams)
		self.val_dset = TrajectoryDataset( mode = "val",  scene_batching = True, **self.hparams)



	def train_dataloader(self):
		# REQUIRED

		return DataLoader(
			self.train_dset,
			batch_size=self.batch_size,
			shuffle= True,
			num_workers=self.hparams.num_workers,
			collate_fn=seq_collate
		)


	def val_dataloader(self):
		# OPTIONAL
		return DataLoader(
            self.val_dset,
            batch_size=self.hparams.batch_size,
            shuffle= True,
            num_workers=self.hparams.num_workers,
            collate_fn=seq_collate
        )

	def setupTestData(self):


		self.test_dset = TrajectoryDataset(mode="test",  scene_batching = True, **self.hparams)

	def test_dataloader(self):
		self.setupTestData()

		return DataLoader(

			self.test_dset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			collate_fn=seq_collate
			)

	"""########## TRAINING ##########"""

	def training_step(self, batch, batch_idx, optimizer_idx):

		# gives a single float value
		self.batch_idx = batch_idx
		self.generator.gen()
		self.logger.experiment.add_scalar('train/CPU Usage', psutil.cpu_percent(), self.global_step)

		if self.device.type != 'cpu':
			self.logger.experiment.add_scalar('train/GPU Usage', torch.cuda.get_device_properties(self.device).total_memory,self.global_step )
		self.plot_val = True
		if self.gsteps and optimizer_idx == 0 and self.current_batch_idx != batch_idx:

			output =  self.generator_step(batch)

			self.output = output
			return output
		elif self.dsteps and optimizer_idx == 1 and self.current_batch_idx != batch_idx:

			output =  self.discriminator_step(batch)

			self.output = output
			return output
		else:
			return self.output


	def test(self, batch):

		self.generator.test()
		with torch.no_grad():
			out = self.generator( batch )
		return out

	def forward(self, batch):
		return self.generator(batch)


	def generator_step(self, batch):

		"""Generator optimization step.
		Args:
		    batch: Batch from the data loader.

		Returns:
		    discriminator loss on fake
		    norm loss on trajectory
		    kl loss
		"""

		# init loss and loss dict
		tqdm_dict = {}
		total_loss = 0.
		ade_sum, fde_sum = [], []
		ade_sum_pixel, fde_sum_pixel = [], []
		# get k times batch
		batch = get_batch_k(batch, self.hparams.best_k)

		batch_size = batch["size"].item()

		generator_out = self.generator(batch)




		if self.hparams.absolute:
			l2 = self.loss_fns["L2"](
				batch["gt_xy"],
				generator_out["out_xy"],
				mode='average',
				type="mse")
		else:
			l2 = self.loss_fns["L2"](
				batch["gt_dxdy"],
				generator_out["out_dxdy"],
				mode='raw',
				type="mse")

		ade_error = cal_ade(
			batch["gt_xy"], generator_out["out_xy"], mode='raw'
		)

		fde_error = cal_fde(
			batch["gt_xy"], generator_out["out_xy"], mode='raw'
		)

		ade_error = ade_error.view(self.hparams.best_k, batch_size)

		fde_error = fde_error.view(self.hparams.best_k, batch_size)

		# get pixel ratios
		ratios = []
		for img in batch["scene_img"]:
			ratios.append(torch.tensor(img["ratio"]))
		ratios = torch.stack(ratios).to(self.device)

		for idx, (start, end) in enumerate(batch["seq_start_end"]):
			ade_error_sum = torch.sum(ade_error[:, start:end], dim=1)
			fde_error_sum = torch.sum(fde_error[:, start:end], dim=1)

			ade_sum_scene, id_scene = ade_error_sum.min(dim=0, keepdims=True)
			fde_sum_scene, _ = fde_error_sum.min(dim=0, keepdims=True)

			ade_sum.append(ade_sum_scene / (self.hparams.pred_len * (end - start)))
			fde_sum.append(fde_sum_scene / (end - start))

			ade_sum_pixel.append(ade_sum_scene / (self.hparams.pred_len * (end - start) * ratios[idx]))
			fde_sum_pixel.append(fde_sum_scene / (ratios[idx] * (end - start)))

		tqdm_dict["ADE_train"] = torch.mean(torch.stack(ade_sum))
		tqdm_dict["FDE_train"] = torch.mean(torch.stack(fde_sum))


		tqdm_dict["ADE_pixel_train"] = torch.mean(torch.stack(ade_sum_pixel))
		tqdm_dict["FDE_pixel_train"] = torch.mean(torch.stack(fde_sum_pixel))
		# count trajectories crashing into the 'wall'
		if any(batch["occupancy"]):
			wall_crashes = crashIntoWall(generator_out["out_xy"].detach().cpu(), batch["occupancy"])
		else:
			wall_crashes = [0]
		tqdm_dict["feasibility_train"] = torch.tensor(1 - np.mean(wall_crashes))
		l2 = l2.view(self.hparams.best_k, -1)

		loss_l2, _ = l2.min(dim=0, keepdim=True)
		loss_l2 = torch.mean(loss_l2)

		loss_l2 = self.loss_weights["L2"]*loss_l2
		tqdm_dict["L2_train"] = loss_l2
		total_loss+=loss_l2
		if self.generator.global_vis_type == "goal":
			target_reshaped = batch["prob_mask"][:batch_size].view(batch_size, -1)
			output_reshaped = generator_out["y_scores"][:batch_size].view(batch_size, -1)

			_, targets = target_reshaped.max(dim=1)

			loss_gce = self.loss_weights["GCE"] * self.loss_fns["GCE"](output_reshaped, targets)

			total_loss+=loss_gce
			tqdm_dict["GCE_train"] = loss_gce


			final_end = torch.sum(generator_out["out_dxdy"], dim=0, keepdim=True)
			final_end_gt = torch.sum(batch["gt_dxdy"], dim=0, keepdim=True)

			final_pos = generator_out["final_pos"]

			goal_error = self.loss_fns["G"](final_pos.detach(), final_end_gt)
			goal_error = goal_error.view(self.hparams.best_k, -1)
			_, id_min = goal_error.min(dim=0, keepdim=False)
			# id_min*=torch.range(0, len(id_min))*10

			final_pos = final_pos.view(self.hparams.best_k, batch["size"], -1)
			final_end = final_end.view(self.hparams.best_k, batch["size"], -1)

			final_pos = torch.cat([final_pos[id_min[k], k].unsqueeze(0) for k in range(final_pos.size(1))]).unsqueeze(0)
			final_end = torch.cat([final_end[id_min[k], k].unsqueeze(0) for k in range(final_end.size(1))]).unsqueeze(0)

			loss_G  = self.loss_weights["G"] * torch.mean( self.loss_fns["G"](final_pos.detach(), final_end, mode='raw'))

			total_loss+=loss_G
			tqdm_dict["G_train"] = loss_G


		traj_fake = generator_out["out_xy"][:, :batch_size]
		traj_fake_rel = generator_out["out_dxdy"][:, :batch_size]

		if self.generator.rm_vis_type == "attention":
			image_patches = generator_out["image_patches"][:, :batch_size]
		else:
			image_patches = None

		fake_scores = self.discriminator(in_xy=batch["in_xy"][:, :batch_size],
										  in_dxdy=batch["in_dxdy"][:, :batch_size],
										  out_xy=traj_fake,
										  out_dxdy=traj_fake_rel,
										  images_patches=image_patches)

		loss_adv = self.loss_weights["G"] * self.loss_fns["ADV"](fake_scores, True).clamp(min=0)

		total_loss+=loss_adv
		tqdm_dict["ADV_train"] = loss_adv

		tqdm_dict["G_loss"] = total_loss
		for key, loss in tqdm_dict.items():
			self.logger.experiment.add_scalar('train/{}'.format(key), loss, self.global_step)
		
		return {"loss": total_loss}


	def discriminator_step(self, batch):

		"""Discriminator optimization step.

		Args:
			batch: Batch from the data loader.

		Returns:
			discriminator loss on fake
			discriminator loss on real
		"""
		# init loss and loss dict
		tqdm_dict = {}
		total_loss = 0.

		self.generator.gen()
		self.discriminator.grad(True)

		with torch.no_grad():

			out = self.generator(batch)

		traj_fake = out["out_xy"]
		traj_fake_rel = out["out_dxdy"]


		if self.generator.rm_vis_type == "attention":
			image_patches = out["image_patches"]
		else:
			image_patches = None

		dynamic_fake = self.discriminator(in_xy=batch["in_xy"],
										 in_dxdy=batch["in_dxdy"],
										 out_xy=traj_fake,
										 out_dxdy=traj_fake_rel,
										 images_patches=image_patches)

		if self.generator.rm_vis_type == "attention":
			image_patches = batch["local_patch"].permute(1, 0, 2, 3, 4)
		else:
			image_patches = None

		dynamic_real = self.discriminator(in_xy=batch["in_xy"],
										 in_dxdy=batch["in_dxdy"],
										 out_xy=batch["gt_xy"],
										 out_dxdy=batch["gt_dxdy"],
										 images_patches=image_patches)


		disc_loss_real_dynamic = self.loss_fns["ADV"](dynamic_real, True).clamp(min=0)
		disc_loss_fake_dynamic = self.loss_fns["ADV"](dynamic_fake, False).clamp(min=0)

		disc_loss = disc_loss_real_dynamic + disc_loss_fake_dynamic

		tqdm_dict = { "D_train" : disc_loss,
					  "D_real_train" : disc_loss_real_dynamic,
					  "D_fake_train" : disc_loss_fake_dynamic}


		for key, loss in tqdm_dict.items():
			self.logger.experiment.add_scalar('train/{}'.format(key), loss, self.global_step)
		return {
			'loss': disc_loss
		}


	"""########## VISUALIZATION ##########"""
	def visualize_results(self, batch, out):

		background_image = batch["scene_img"][0]["scaled_image"].copy()

		inp = batch["in_xy"]
		gt = batch["gt_xy"]
		pred = out["out_xy"]
		pred = pred.view(pred.size(0), self.hparams.best_k_val, -1, pred.size(-1))

		y = out["y_map"]
		y_softmax = out["y_softmax"]



		image = visualize_traj_probabilities(
			input_trajectory=inp.cpu()[:, 0],
			gt_trajectory=None,
			prediction_trajectories=pred.cpu()[:,:, 0],
			background_image=background_image,
			img_scaling=self.val_dset.img_scaling,
			scaling_global=self.val_dset.scaling_global,
			grid_size=20,
			y_softmax=y_softmax,
			y=y,
			global_patch=re_im(batch["global_patch"][0]).cpu().numpy(),
			probability_mask=batch["prob_mask"][0][0].cpu().numpy(),
			grid_size_in_global=self.val_dset.grid_size_in_global

			)



		self.logger.experiment.add_image(f'Trajectories', image, self.current_epoch)



	"""########## EVAL HELPERS ##########"""


	def eval_step(self, batch, best_k=10):
		ade_sum, fde_sum = [], []
		ade_sum_pixel, fde_sum_pixel = [], []

		# get pixel ratios
		ratios = []
		for img in batch["scene_img"]:
			ratios.append(torch.tensor(img["ratio"]))
		ratios = torch.stack(ratios).to(self.device)

		batch = get_batch_k(batch, best_k)
		batch_size = batch["size"]

		out = self.test(batch)

		if self.plot_val:
			self.plot_val = False
			self.visualize_results(batch, out)


		# FDE and ADE metrics
		ade_error = cal_ade(
			batch["gt_xy"], out["out_xy"], mode='raw'
		)

		fde_error = cal_fde(
			batch["gt_xy"], out["out_xy"], mode='raw'
		)

		ade_error = ade_error.view(best_k, batch_size)

		fde_error = fde_error.view(best_k, batch_size)

		for idx, (start, end) in enumerate(batch["seq_start_end"]):
			ade_error_sum = torch.sum(ade_error[:, start:end], dim=1)
			fde_error_sum = torch.sum(fde_error[:, start:end], dim=1)

			ade_sum_scene, id_scene = ade_error_sum.min(dim=0, keepdims=True)
			fde_sum_scene, _ = fde_error_sum.min(dim=0, keepdims=True)

			ade_sum.append(ade_sum_scene / (self.hparams.pred_len * (end - start)))
			fde_sum.append(fde_sum_scene / (end - start))

			ade_sum_pixel.append(ade_sum_scene / (self.hparams.pred_len * (end - start) * ratios[idx]))
			fde_sum_pixel.append(fde_sum_scene / (ratios[idx] * (end - start)))


		# compute Mode Caughts metrics
		fde_min, _ = fde_error.min(dim=0)
		modes_caught = (fde_min < self.hparams.mode_dist_threshold).float()

		if any(batch["occupancy"]):

			wall_crashes = crashIntoWall(out["out_xy"].cpu(), batch["occupancy"])
		else:
			wall_crashes = [0]
		return {"ade": ade_sum, "fde": fde_sum, "ade_pixel": ade_sum_pixel, "fde_pixel": fde_sum_pixel,
				"wall_crashes": wall_crashes, "modes_caught": modes_caught}

	def collect_losses(self, outputs, mode = "val", plot = True):

		ade  = torch.stack(list(itertools.chain(*[x['ade'] for x in outputs]))).mean()
		fde = torch.stack(list(itertools.chain(*[x['fde'] for x in outputs]))).mean()
		ade_pixel = torch.stack(list(itertools.chain(*[x['ade_pixel'] for x in outputs]))).mean()
		fde_pixel = torch.stack(list(itertools.chain(*[x['fde_pixel'] for x in outputs]))).mean()
		feasibility = 1 -  np.mean(list(itertools.chain(*[x["wall_crashes"] for x in outputs])))
		mc_metric = torch.stack(list(itertools.chain(*[x["modes_caught"] for x in outputs]))).mean()


		loss = ((fde+ade)/2.)
		logs = {'{}_loss'.format(mode): loss, 'ade_{}'.format(mode): ade.item(),
				"fde_{}".format(mode) : fde.item(),	"ade_pixel_{}".format(mode) : ade_pixel.item(),
				"fde_pixel_{}".format(mode) : fde_pixel.item(), "feasibility_{}".format(mode) : feasibility,
				"mc_{}".format(mode): mc_metric}
		# plot val
		if plot:
			for key, loss in logs.items():
				self.logger.experiment.add_scalar('{}/{}'.format(mode, key), loss, self.current_epoch)


		return {'{}_loss'.format(mode): loss, 'progress_bar': logs}

	"""########## VALIDATION ##########"""
	def validation_step(self, batch, batch_idx):
		return self.eval_step(batch, self.hparams.best_k_val)

	def validation_epoch_end(self, outputs):
		return self.collect_losses(outputs, mode = "val")

	"""########## TESTING ##########"""
	def test_step(self, batch, batch_idx):
		output = self.eval_step(batch, self.hparams.best_k_test)
		return output

	def test_epoch_end(self, outputs):
		results = self.collect_losses(outputs, mode="test")

		torch.save( results["progress_bar"], os.path.join(self.logger.log_dir, "results.pt"))

		print(results)
		return results


	"""########## OPTIMIZATION ##########"""
	def backward(self, trainer, loss, optimizer, optimizer_idx ):
		# condition that backward is not called when nth is passed
		if self.current_batch_idx != self.batch_idx and ( ((optimizer_idx == 0 ) and self.gsteps) or ((optimizer_idx == 1) and self.dsteps)):

			loss.backward()

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs):
		# Step using d_loss or g_loss
		# update generator opt every 2 steps
		if self.gsteps and optimizer_idx == 0 and self.current_batch_idx != batch_idx:
			self.gsteps -= 1
			if not self.gsteps:
				if self.discriminator:
					self.dsteps = self.hparams.d_steps
				else:
					self.gsteps = self.hparams.g_steps
			self.current_batch_idx = batch_idx
			optimizer.step()
			optimizer.zero_grad()


		# update discriminator opt every 4 steps
		if self.dsteps and optimizer_idx == 1 and self.current_batch_idx != batch_idx:
			self.dsteps -= 1
			if not self.dsteps:
				self.gsteps = self.hparams.g_steps
			self.current_batch_idx = batch_idx
			optimizer.step()
			optimizer.zero_grad()


	def configure_optimizers(self):
			opt_g = RAdam(self.generator.parameters(), lr = self.hparams.lr_gen)
			opt_d = RAdam(self.discriminator.parameters(), lr = self.hparams.lr_dis)

			schedulers = []
			if self.hparams.lr_scheduler_G:
				lr_scheduler_G = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler_G)(opt_g)
				schedulers.append(lr_scheduler_G)
			else:
				schedulers.append(None)

			if self.hparams.lr_scheduler_D:
				lr_scheduler_D = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler_D)(opt_d)
				schedulers.append(lr_scheduler_D)
			else:
				schedulers.append(None)

			self.gsteps = self.hparams.g_steps
			self.dsteps = 0
			return [opt_g, opt_d], schedulers
