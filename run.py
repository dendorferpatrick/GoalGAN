from model.main_pl import TrajPredictor
from model.pretrain_pl import pretrain_func
import hydra
from pytorch_lightning import Trainer, seed_everything
import os
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import torch

logger = logging.getLogger(__name__)
@hydra.main(config_path="config", config_name="config")
def run(cfg):
	"""
	Main function training Goal-GAN
	"""
	torch.multiprocessing.set_sharing_strategy('file_system')
	seed_everything(cfg.random_seed)


	"""
	Results and model are saved with Tensorboard
	"""
	tb_logger = TensorBoardLogger(".", name = "{}".format(cfg.dataset_name))

	if cfg.make_checkpoint:
		checkpoint_location = os.path.join(tb_logger.log_dir,"checkpoints")
		logger.info("Checkpoint Location: {}".format(checkpoint_location))
		checkpoint_callback = ModelCheckpoint(
			filepath=os.path.join(tb_logger.log_dir,"checkpoints", 'checkpoint'),
			save_top_k=1,
			verbose=True,
			monitor='val_loss',
			mode='min'
		)

	# init model
	model = TrajPredictor(cfg)
	# load data
	model.setupData()

	"""
	Before training the entire model you can choose to pretrain the Goal-Module only. 
	This is helpful to get reasonable goal estimates before  training the entire pipeline. 
	Otherwise, the network may converge to an undesired local minimum."""
	# pretrain if wished
	if cfg.pretrain:
		pretrain_func(generator=model.generator,
					  train_dset=model.train_dset,
					  val_dset=model.val_dset,
					  cfg=cfg,
					  logger=tb_logger)


	if cfg.batch_size_scheduler:
		# init batchsize_scheduler
		from utils.batchsizescheduler import BatchSizeScheduler
		callbacks = [BatchSizeScheduler(	bs = cfg.batch_size_scheduler,
											max_bs = cfg.batch_size )]



	else:
		callbacks = False

	trainer = Trainer(logger=tb_logger,
					  callbacks=callbacks,
					  checkpoint_callback=checkpoint_callback,
					  num_sanity_val_steps=0,
					  progress_bar_refresh_rate=10,
					  **cfg.trainer)
	trainer.fit(model)

	logger.info("Loading best model")
	trainer.test(ckpt_path='best')

if __name__ == "__main__":
	run()
