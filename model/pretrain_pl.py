import pytorch_lightning as pl
from omegaconf import DictConfig
import torch.nn as nn
from utils.losses import l2_loss, GANLoss
import torch
from data import seq_collate
from torch.utils.data import DataLoader
from utils.utils import re_im
from pytorch_lightning import Trainer
from utils.visualize import visualize_probabilities
# load optimizer
from utils.radam import RAdam


def pretrain_func(generator, train_dset, val_dset, cfg, logger = None):
	pretrain = Pretrain(generator, train_dset, val_dset, cfg)

	StopCriterium = StopPretrain(cfg.pretrain)
	callbacks = [StopCriterium]
	if cfg.pretraining.batch_size_scheduler:
		from utils.batchsizescheduler import BatchSizeScheduler
		callbacks.append(BatchSizeScheduler(bs = cfg.pretraining.batch_size_scheduler,
											max_bs = cfg.pretraining.batch_size ))


	preTrainer = Trainer(logger,
						 callbacks = callbacks,
						 checkpoint_callback=False,
						 num_sanity_val_steps=0,
						 progress_bar_refresh_rate=10,
						 **cfg.trainer)

	preTrainer.fit(pretrain)



class StopPretrain(pl.callbacks.base.Callback):
	def __init__(self, min_loss):
		self.min_loss = min_loss  
		self.monitor_metric = "val_loss"
	def on_validation_epoch_end(self, trainer, pl_module):

		if trainer.callback_metrics[self.monitor_metric] < self.min_loss:
			trainer.should_stop = True



class Pretrain(pl.LightningModule):
	def __init__(self, generator, train_dset, val_dset, cfg: DictConfig = None, loss_fns = None):
		super().__init__()


		self.cfg = cfg
		self.generator = generator

		self.generator.gen()
		# init loss functions
		self.loss_fns = loss_fns if loss_fns else {'L2': l2_loss,  # L2 loss
												   'ADV': GANLoss(cfg.gan_mode),  # adversarial Loss
												   'G': l2_loss,  # goal achievement loss
												   'GCE': nn.CrossEntropyLoss()}  # Goal Cross Entropy loss
		# init loss weights

		self.loss_weights = {'L2': cfg.w_L2,
							 'ADV': cfg.w_ADV,  # adversarial Loss
							 'G': cfg.w_G,  # goal achievement loss
							 'GCE': cfg.w_GCE}  # Goal Cross Entropy loss
		self.train_dset = train_dset
		self.val_dset = val_dset

		self.plot_val = True

		if self.cfg.pretraining.batch_size_scheduler:
			self.batch_size = self.cfg.pretraining.batch_size_scheduler
		else: self.batch_size = self.cfg.batch_size
	def train_dataloader(self):
		# REQUIRED
		return DataLoader(
			self.train_dset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.cfg.num_workers,
			collate_fn=seq_collate
		)

	def val_dataloader(self):
		# OPTIONAL

		return DataLoader(
			self.val_dset,
			batch_size=self.cfg.pretraining.batch_size,
			shuffle=True,
			num_workers=self.cfg.num_workers,
			collate_fn=seq_collate
		)

	def training_step(self, batch, batch_idx):
		# gives a single float value
		# init loss and loss dict
		tqdm_dict = {}
		total_loss = 0.



		batch_size = batch["size"].item()

		generator_out = self.generator(batch)


	
		target_reshaped = batch["prob_mask"].view(batch_size, -1)
		output_reshaped = generator_out["y_scores"].view(batch_size, -1)

		_, targets = target_reshaped.max(dim=1)
		
		loss_gce = self.loss_fns["GCE"](output_reshaped, targets)

		total_loss += loss_gce
		tqdm_dict["GCE_pretrain"] = loss_gce
	
		for key, loss in tqdm_dict.items():
			self.logger.experiment.add_scalar('pre/{}'.format(key), loss, self.global_step)

		return {"loss": total_loss}

	def validation_step(self, batch, batch_idx):

		self.generator.test()

		# init loss and loss dict
		tqdm_dict = {}
		total_loss = 0.

		batch_size = batch["size"].item()

		generator_out = self.generator(batch)

		if self.plot_val:
			self.plot_val = False
			self.visualize_results(batch, generator_out)

		target_reshaped = batch["prob_mask"][:batch_size].view(batch_size, -1)
		output_reshaped = generator_out["y_scores"][:batch_size].view(batch_size, -1)

		_, targets = target_reshaped.max(dim=1)

		loss_gce = self.loss_weights["GCE"] * self.loss_fns["GCE"](output_reshaped, targets)

		total_loss += loss_gce
		tqdm_dict["GCE_pretrain"] = loss_gce


		# include early stopping when loss below threshold


		return {"loss": total_loss}


	def visualize_results(self, batch, out):

		y = out["y_map"]
		y_softmax = out["y_softmax"]


		image = visualize_probabilities(y_softmax = y_softmax,
										y = y,
										global_patch = re_im(batch["global_patch"][0]).cpu().numpy(),
										probability_mask = batch["prob_mask"][0][0].cpu().numpy(),
										grid_size_in_global = self.val_dset.grid_size_in_global
										)
		self.logger.experiment.add_image(f'Map', image, self.current_epoch)


	def validation_epoch_end(self, outputs):

		GCE_loss = torch.stack([x['loss'] for x in outputs]).mean()

		self.logger.experiment.add_scalar('pre/GCE_val', GCE_loss, self.current_epoch)
		self.plot_val = True
	
		return {'val_loss':  GCE_loss}

	def configure_optimizers(self):
		opt_g = RAdam(self.generator.parameters(), lr=self.cfg.lr_pretrain)

		
		if self.cfg.lr_scheduler_pretrain:
			lr_scheduler_pretrain = [getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler_pretrain)(opt_g)]

		else:
			lr_scheduler_pretrain = []

		return [opt_g], lr_scheduler_pretrain
