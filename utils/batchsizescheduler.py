import pytorch_lightning as pl
import numpy as np

class BatchSizeScheduler(pl.callbacks.base.Callback):
    """
    Implmentation of a BatchSize Scheduler following the paper
    'Don't Decay the Learning Rate, Increase the Batch Size'
    (https://arxiv.org/abs/1711.00489)
    The scheduler increases the batchsize if the validation loss does not decrease.
    """
    def __init__(self, bs = 4 , factor=2, patience=3, max_bs=64, mode = "min", monitor_val = "val_loss"):
        """
        :param bs: initial batch size
        :param factor: factor by which current batch size is increased
        :param patience: duration in which loss does not have to decrease
        :param max_bs: maximum batch size
        :param mode: considering 'min' or 'max' for 'monitor_val'
        :param monitor_val: considered loss for scheduler
        """

        self.factor = factor
        self.patience = patience
        self.max_bs = max_bs
        self.current_count = patience*1.
        self.cur_metric = False
        self.monitor_metric = monitor_val
        self.cur_bs = bs
        if mode not in ["min", "max"]:
            assert False, "Variable for mode '{}' not valid".format(mode)
        self.mode = mode
        if max_bs > bs:
            self.active = True
        else: self.active = False

    def on_validation_end(self, trainer, pl_module):

        self.cur_bs = int(np.minimum(self.cur_bs * self.factor, self.max_bs))

        # set new batch_size
        pl_module.batch_size = self.cur_bs
        trainer.reset_train_dataloader(pl_module)

        if not self.cur_metric:
            self.cur_metric = trainer.callback_metrics[self.monitor_metric]

        if self.active:
            if self.mode == "min":
                if trainer.callback_metrics[self.monitor_metric]  < self.cur_metric:
                    self.cur_metric =trainer.callback_metrics[self.monitor_metric]
                    self.current_count = self.patience*1

                else:
                    self.current_count-=1


            else:
                if trainer.callback_metrics[self.monitor_metric]  > self.cur_metric:
                    self.cur_metric = trainer.callback_metrics[self.monitor_metric]
                    self.current_count = self.patience*1

                else:
                    self.current_count -= 1

            if self.current_count == 0:
                self.cur_bs = int(np.minimum(self.cur_bs*self.factor, self.max_bs))

                # set new batch_size
                pl_module.batch_size = self.cur_bs
                trainer.reset_train_dataloader(pl_module)
                print("SET BS TO {}".format(self.cur_bs))
                self.current_count = self.patience*1
                if self.cur_bs >=self.max_bs:
                    self.active = False
