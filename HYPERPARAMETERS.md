# Hyperparameters

For training Goal-GAN, you can set a number of hyperparameters effecting model architecture, training, and testing in [config](config). 
### Hyperparameters for model in ```config/model/model.yaml```
<!--- START:configs:model --->
 |Argument | Documentation |
| --- | --- |
| dropout | Dropout probability in generator (float) | 
| batch_norm | Using Batch Norm in generator (bool) | 
| input_dim | Trajectory input dimensionality; usually 2d (int) | 
| encoder_h_dim_g | Dimensionality of LSTM hidden state of generator encoder (int) | 
| decoder_h_dim_g | Dimensionality of LSTM hidden state of generator decoder (int) | 
| mlp_dim | Dimensionality of MLPs in generator | 
| embedding_dim | Spation coordinate embedding dim (int) | 
| temperature_global | Gumbel Temperatur of global goal module (float) | 
| force_hard_global | Forward pass style of Gumbel Softmmax Trick (bool) | 
| num_layers_cnn_global | Number of cnn layers in goal module (int) | 
| channels_cnn_global | Channels of first conv in goal module (int) | 
| batch_norm_cnn_global | Batch norm in goal module (bool) | 
| dropout_cnn_global | Dropout in cnn bocks in goal module (float) | 
| non_lin_cnn_global | Nonlinear activation goal module ('relu', 'sigmoid','None') | 
| skip_connection_global | Using skip connection for goal module CNN (bool) | 
| final_embedding_dim_global | Dimension of goal estimate embedding (int) | 
| global_vis_type | Type in goal module ('goal', 'None') | 
| rm_vis_type | Visual type in routing module ('attention') | 
| num_layers_cnn_rm | Number of cnn layers in routing module (int) | 
| batch_norm_cnn_rm | Batch norm in goal module (bool) | 
| dropout_cnn_rm | Dropout in cnn bocks in goal module (float) | 
| non_lin_cnn_rm | Nonlinear activation goal module ('relu', 'sigmoid','None') | 
| noise_attention_dim_rm | Dimension of noise vector in routing module (int) | 
| final_embedding_dim_rm | Dimension of feature embedding in routing module (int) | 
| skip_connection_rm | Using skip connection for routing module CNN (bool) | 
| channels_cnn_rm | Channels of first conv in routing module (int) | 
| gan_mode | Type of GAN objective. It currently supports ('vanilla', 'lsgan', 'wgan'). (str) | 
| encoder_h_dim_d | Dimensionality of LSTM hidden state of discriminator encoder (int) | 
| dropout_disc | Dropout probability in discriminator (float) | 
| dropout_cnn_disc | Dropout probability in discriminator (float) | 
| non_lin_cnn_disc | Nonlinear activation goal module ('relu', 'sigmoid','None') | 
| channels_cnn_disc | Channels of first conv in discriminator (int) | 
| num_layers_cnn_disc | Number of cnn layers in discriminator (int) | 
| batch_norm_disc | Using Batch Norm in the model (bool) | 
| visual_features_disc | Using visual features/ CNN in discriminator | 
<!--- END:configs:model --->
### Hyperparameters for experiment in  files ```config/experiment```
<!--- START:configs:experiment --->
 |Argument | Documentation |
| --- | --- |
| max_num | Maximal number of samples in dataset, 'False' if all samples, (int, bool) | 
| dataset_name | Available datasets are :'eth', 'hotel', 'zara1', 'zara2', 'univ', 'stanford_synthetic' , 'stanford' (str) | 
| obs_len | Observation time length (int); default:8 | 
| pred_len | Prediction time lenght (int); default:12 | 
| random_seed | Random seed for training run (int) | 
| scaling_global | Scaling (meter/pixel) of image patch of goal module (float) | 
| scaling_local | Scaling (meter/pixel) of image patch for routing module (float) | 
| img_scaling | Image scaling (meter/pixel)  of background image (float) | 
| grid_size_in_global | Input grid size of image patch global module (int)) | 
| grid_size_out_global | Output grid size of image patch global module (int) | 
| grid_size_local | Grid size of image patch routing module (int) | 
| load_semantic_map | Use semantic map as image input (bool) | 
| pretrain | Pretraining goal module. Training until loss value is reached (float) | 
<!--- END:configs:experiment --->
### Hyperparameters for training ```config/training/training.yaml```

<!--- START:configs:training --->
 |Argument | Documentation |
| --- | --- |
| num_workers | Number of workers in dataloader (int) | 
| make_checkpoint | Save checkpoints (bool) | 
| data_augmentation | Data Augmentation on data (flip and rotation), (bool) | 
| skip | Number of time steps skipped between two samples from same trajectory during loading (int) | 
| trainer.max_epochs | Number of epochs (int) | 
| trainer.gpus | List of GPUs (bool) or (list) | 
| trainer.fast_dev_run | Run fast development run | 
| batch_size | Training batch size (int) | 
| batch_size_scheduler | If True, use batch size scheduer (https://arxiv.org/abs/1711.00489) during training; set initial training batch size for scheduler (int) | 
| pretraining.batch_size | Pretraing batch size (int) | 
| pretraining.batch_size_scheduler | If True, starting pretraining batch size for scheduler | 
| lr_scheduler_G | Learning rate scheduler for generator function e.g. ReduceLROnPlateau | 
| lr_scheduler_D | Learning rate scheduler for discriminator function e.g. ReduceLROnPlateau | 
| lr_scheduler_pretrain | Learning rate scheduler for pretrain optimizer function e.g. ReduceLROnPlateau | 
| best_k | Number of training samples (int) | 
| best_k_val | Number of validation samples (int) | 
| absolute | Using error of absolute coordinates for L2 loss (bool) | 
| w_ADV | Weight of adversarial loss (float) | 
| w_L2 | Weight of L2 loss (float) | 
| w_G | Weight of goal loss (float) | 
| w_GCE | Weight goal cross entropy loss (float) | 
| lr_gen | Learning rate generator optimizer (float) | 
| lr_dis | Learning rate discriminator optimzier (float) | 
| lr_pretrain | learning rate pretrain optimizer (float) | 
| g_steps | Training steps generator (int) | 
| d_steps | Training steps discriminator (int) | 
<!--- END:configs:training --->
### Hyperparameters for evaluation ```config/eval/eval.yaml```
<!--- START:configs:evaluation --->
 |Argument | Documentation |
| --- | --- |
| mode_dist_threshold | Maximum distance [m] between GT and prediction final point for 'mode caught' metric (float) | 
| best_k_test | Number of test samples (int) | 
<!--- END:configs:evaluation --->

