import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


"""In this script are all modules required for the  generator and discriminator"""
### Helper Functions ###
def make_mlp(dim_list, activation_list, batch_norm=False, dropout=0):
    """
    Generates MLP network:

    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)

    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        activation = activation_list[index]
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))
        index += 1
    return nn.Sequential(*layers)

### Convolutional Blocks and U-NET CNN ###

class Conv_Blocks(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size=3, batch_norm=False, non_lin="tanh", dropout=0.,
                 first_block=False, last_block=False, skip_connection=False):
        super(Conv_Blocks, self).__init__()
        self.skip_connection = skip_connection
        self.last_block = last_block
        self.first_block = first_block
        self.Block = nn.Sequential()
        self.Block.add_module("Conv_1", nn.Conv2d(input_dim, output_dim, filter_size, 1, 1))
        if batch_norm:
            self.Block.add_module("BN_1", nn.BatchNorm2d(output_dim))
        if non_lin == "tanh":
            self.Block.add_module("NonLin_1", nn.Tanh())
        elif non_lin == "relu":
            self.Block.add_module("NonLin_1", nn.ReLU())
        elif non_lin == "leakyrelu":
            self.Block.add_module("NonLin_1", nn.LeakyReLU())
        else:
            assert False, "non_lin = {} not valid: 'tanh', 'relu', 'leakyrelu'".format(non_lin)


        self.Block.add_module("Pool", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False))
        if dropout > 0:
            self.Block.add_module("Drop", nn.Dropout2d(dropout))

    def forward(self, x, ):

        if self.skip_connection:
            if not self.first_block:

                x, skip_con_list = x

            else:
                skip_con_list = []

        x = self.Block(x)
        if self.skip_connection:
            if not self.last_block:
                skip_con_list.append(x)
            x = [x, skip_con_list]

        return x


class UpConv_Blocks(nn.Module):
    def __init__(self, input_dim, output_dim, filter=4, padding=1, first_block=False, last_block=False,
                 batch_norm=False, non_lin="relu", dropout=0, skip_connection=False):
        super(UpConv_Blocks, self).__init__()
        self.Block = nn.Sequential()
        self.skip_connection = skip_connection
        self.first_block = first_block
        self.last_block = last_block
        if self.skip_connection and not self.first_block:
            ouput_dim_conv = input_dim
            input_dim *= 2
        else:
            ouput_dim_conv = output_dim

        self.Block.add_module("UpConv", nn.ConvTranspose2d(input_dim, output_dim, filter, 2, padding))
        if not last_block:
            if batch_norm:
                self.Block.add_module("BN_up", nn.BatchNorm2d(output_dim))
            if non_lin == "tanh":
                self.Block.add_module("NonLin_up", nn.Tanh())
            elif non_lin == "relu":
                self.Block.add_module("NonLin_up", nn.ReLU())
            elif non_lin == "leakyrelu":
                self.Block.add_module("NonLin_up", nn.LeakyReLU())
            if dropout > 0:
                self.Block.add_module("Drop_up", nn.Dropout2d(dropout))

    def forward(self, x, ):
        if self.skip_connection:
            x, skip_con_list = x
            if not self.first_block:
                x = torch.cat((x, skip_con_list.pop(-1)), -3)
        x = self.Block(x)

        if self.skip_connection and not self.last_block:
            x = [x, skip_con_list]
        return x


class CNN(nn.Module):
    def __init__(self,
                 social_pooling=False,
                 channels_cnn=4,
                 mlp=32,
                 encoder_h_dim=16,
                 insert_trajectory=False,
                 need_decoder=False,
                 PhysFeature=False,
                 grid_size_in=32,
                 grid_size_out=32,
                 num_layers=3,
                 dropout=0.,
                 batch_norm=False,
                 non_lin_cnn="tanh",
                 in_channels=3,
                 skip_connection=False,
                 ):
        super(CNN, self).__init__()
        self.__dict__.update(locals())


        self.bottleneck_dim = int(grid_size_in / 2 ** (num_layers - 1)) ** 2

        num_layers_dec = int(num_layers + ((grid_size_out - grid_size_in) / grid_size_out))

        self.encoder = nn.Sequential()

        layer_out = channels_cnn
        self.encoder.add_module("ConvBlock_1", Conv_Blocks(in_channels, channels_cnn,
                                                           dropout=dropout,
                                                           batch_norm=batch_norm,
                                                           non_lin=self.non_lin_cnn,
                                                           first_block=True,
                                                           skip_connection=self.skip_connection

                                                           ))
        layer_in = layer_out
        for layer in np.arange(2, num_layers + 1):

            if layer != num_layers:
                layer_out = layer_in * 2
                last_block = False
            else:
                layer_out = layer_in
                last_block = True
            self.encoder.add_module("ConvBlock_%s" % layer,
                                    Conv_Blocks(layer_in, layer_out,
                                                dropout=dropout,
                                                batch_norm=batch_norm,
                                                non_lin=self.non_lin_cnn,
                                                skip_connection=self.skip_connection,
                                                last_block=last_block
                                                ))
            layer_in = layer_out

        self.bootleneck_channel = layer_out
        if self.need_decoder:

            self.decoder = nn.Sequential()
            layer_in = layer_out
            for layer in range(1, num_layers_dec + 1):
                first_block = False
                extra_d = 0
                layer_in = layer_out
                last_block = False
                filter = 4
                padding = 1
                if layer == 1:
                    if self.insert_trajectory:
                        extra_d = 1

                    first_block = True
                    layer_out = layer_in

                else:
                    layer_out = int(layer_in / 2.)

                if layer == num_layers_dec:
                    layer_out = 1
                    last_block = True
                    padding = 0
                    filter = 3

                self.decoder.add_module("UpConv_%s" % layer,
                                        UpConv_Blocks(int(layer_in + extra_d),
                                                      layer_out,
                                                      first_block=first_block,
                                                      filter=filter,
                                                      padding=padding,
                                                      dropout=dropout,
                                                      batch_norm=batch_norm,
                                                      non_lin=self.non_lin_cnn,
                                                      skip_connection=self.skip_connection,
                                                      last_block=last_block))

        if self.insert_trajectory:
            self.traj2cnn = make_mlp(
                dim_list=[encoder_h_dim, mlp, self.bottleneck_dim],
                activation_list=["tanh", "tanh"],
            )

        self.init_weights()

    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                m.bias.data.fill_(0.01)
            # if type(m) in [nn.ConvTranspose2d]:
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
            # m.bias.data.fill_(50)

        def init_xavier(m):
            if type(m) == [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        if self.non_lin_cnn in ['relu', 'leakyrelu']:
            self.apply(init_kaiming)
        elif self.non_lin_cnn == "tanh":
            self.apply(init_xavier)
        else:
            assert False, "non_lin not valid for initialisation"

    def forward(self, image, traj_h=torch.empty(1), pool_h=torch.empty(1)):
        output = {}

        enc = self.encoder(image)

        if self.PhysFeature:
            # enc_out = self.leakyrelu(self.encoder_out(enc))
            # enc_out = enc_out.permute(1, 0, 2, 3).view(1, enc_out.size(0), -1)
            output.update(Features=enc)

        if self.need_decoder:

            if self.skip_connection:
                batch, c, w, h = enc[0].size()
                in_decoder, skip_con_list = enc

            else:
                batch, c, w, h = enc.size()
                in_decoder = enc

            if self.insert_trajectory:

                traj_enc = self.traj2cnn(traj_h)

                traj_enc = traj_enc.view(batch, 1, w, h)
                in_decoder = torch.cat((traj_enc, in_decoder), 1)
            if self.social_pooling:

                social_enc = self.social_states(pool_h)

                social_enc = social_enc.view(batch, 1, w, h)
                in_decoder = torch.cat((social_enc, in_decoder), 1)
            if self.skip_connection: in_decoder = [in_decoder, skip_con_list]
            dec = self.decoder(in_decoder)
            output.update(PosMap=dec)

        return output


class MotionEncoder(nn.Module):
    """MotionEncoder extracts dynamic features of the past trajectory and consists of an encoding LSTM network"""

    def __init__(self,
        encoder_h_dim=64,
        input_dim=2,
        embedding_dim=16,
        dropout=0.0):
        """ Initialize MotionEncoder.
        Parameters.
            encoder_h_dim (int) - - dimensionality of hidden state
            input_dim (int) - - input dimensionality of spatial coordinates
            embedding_dim (int) - - dimensionality spatial embedding
            dropout (float) - - dropout in LSTM layer
        """
        super(MotionEncoder, self).__init__()
        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        if embedding_dim:
            self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
            self.encoder = nn.LSTM(embedding_dim, encoder_h_dim)
        else:
            self.encoder = nn.LSTM(input_dim, encoder_h_dim)

    def init_hidden(self, batch, obs_traj):

        return (
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj),
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj)
        )

    def forward(self, obs_traj, state_tuple=None):
        """ Calculates forward pass of MotionEncoder
            Parameters:
                obs_traj (tensor) - - Tensor of shape (obs_len, batch, 2)
                state_tuple (tuple of tensors) - - Tuple with hidden state (1, batch, encoder_h_dim) and cell state tensor (1, batch, encoder_h_dim)
            Returns:
                output (tensor) - - Output of LSTM netwok for all time steps (obs_len, batch, encoder_h_dim)
                final_h (tensor) - - Final hidden state of LSTM network (1, batch, encoder_h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        if not state_tuple:
            state_tuple = self.init_hidden(batch, obs_traj)
        if self.embedding_dim:
            obs_traj = self.spatial_embedding(obs_traj)

        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]
        return output, final_h



class VisualNetwork(nn.Module):
    """VisualNetwork is the parent class for the attention and goal networks generating the CNN"""

    def __init__(self,
                 decoder_h_dim=128,
                 dropout=0.0,
                 batch_norm=False,
                 mlp_dim=32,
                 img_scaling=0.25,
                 final_embedding_dim=4,
                 grid_size_in=16,
                 grid_size_out=16,
                 num_layers=1,
                 batch_norm_cnn=True,
                 non_lin_cnn="relu",
                 img_type="local_image",
                 skip_connection=False,
                 channels_cnn=4,
                 social_pooling=False,
                 **kwargs):

        super(VisualNetwork, self).__init__()
        self.__dict__.update(locals())

    def init_cnn(self):
        self.CNN = CNN(social_pooling=self.social_pooling,
                       channels_cnn=self.channels_cnn,
                       encoder_h_dim=self.decoder_h_dim,
                       mlp=self.mlp_dim,
                       insert_trajectory=True,
                       need_decoder=self.need_decoder,
                       PhysFeature=self.PhysFeature,
                       grid_size_in=self.grid_size_in,
                       grid_size_out=self.grid_size_out,
                       dropout=self.dropout,
                       batch_norm=self.batch_norm_cnn,
                       non_lin_cnn=self.non_lin_cnn,
                       num_layers=self.num_layers,
                       in_channels=4,
                       skip_connection=self.skip_connection
                       )

### Visual Attention ###

class AttentionNetwork(VisualNetwork):
    def __init__(self,
                 noise_attention_dim=8,
                 **kwargs
                 ):
        super(AttentionNetwork, self).__init__()
        VisualNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())
        self.PhysFeature = True
        self.skip_connection = False
        self.need_decoder = False

        self.init_cnn()
        self.final_embedding = self.CNN.bottleneck_dim + self.noise_attention_dim
        attention_dims = [self.CNN.bootleneck_channel, self.mlp_dim, 1]
        activation = ['leakyrelu', None]
        self.cnn_attention = make_mlp(
            attention_dims,
            activation_list=activation, )

    def get_noise(self, batch_size, type="gauss"):
        """
           Create noise vector:
           Parameters
           ----------
           batchsize : int, length of noise vector
           noise_type: str, 'uniform' or 'gaussian' noise

           Returns
           -------
           Random noise vector
           """

        if type == "gauss":
            return torch.randn((1, batch_size, self.noise_attention_dim))
        elif type == "uniform":

            rand_num = torch.rand((1, batch_size, self.noise_attention_dim))
            return rand_num
        else:
            raise ValueError('Unrecognized noise type "%s"' % noise_type)




class AttentionRoutingModule(AttentionNetwork):
    def __init__(self,
                 **kwargs):
        super(AttentionNetwork, self).__init__()
        AttentionNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())
        self.img_patch = Patch_gen(img_scaling=self.img_scaling,
                                   grid_size=self.grid_size_in,
                                   type_img=self.img_type)
        self.init_cnn()

    def forward(self, scene_img, last_pos, h, noise=torch.Tensor()):

        img_patch = self.img_patch.get_patch(scene_img, last_pos)
        visual_features = self.CNN(img_patch, h)["Features"].permute(0, 2, 3, 1)
        batch_size, hh, w, c = visual_features.size()
        visual_features = visual_features.view(batch_size, -1, c)
        attention_scores = self.cnn_attention(visual_features)
        attention_vec = attention_scores.softmax(dim=1).squeeze(2).unsqueeze(0)
        if self.noise_attention_dim > 0:
            if len(noise) == 0:
                noise = self.get_noise(batch_size)
            else:
                assert noise.size(-1) != self.noise_attention_dim, "dimension of noise {} not valid".format(
                    noise.size())

        x = torch.cat((attention_vec, noise.to(attention_vec)), dim=2)

        return x, attention_vec, img_patch, noise


class AttentionGlobal(AttentionNetwork):
    """Alternative Visual Attention to GoalModule"""
    def __init__(self, **kwargs):
        super(AttentionNetwork, self).__init__()
        AttentionNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())

        self.init_cnn()

    def forward(self, features, h, noise=torch.Tensor()):
        visual_features = self.CNN(features, h)["Features"].permute(0, 2, 3, 1)
        batch_size, hh, w, c = visual_features.size()
        visual_features = visual_features.view(batch_size, -1, c)
        attention_scores = self.cnn_attention(visual_features)
        attention_vec = attention_scores.softmax(dim=1).squeeze(2).unsqueeze(0)
        if self.noise_attention_dim > 0:
            if len(noise) == 0:
                noise = self.get_noise(batch_size)
            else:
                assert noise.size(-1) != self.noise_attention_dim, "dimension of noise {} not valid".format(
                    noise.size())

        x = torch.cat((attention_vec, noise.to(attention_vec)), dim=2)

        return x, attention_vec, noise

 ### GOAL Module ###
class GoalGlobal(VisualNetwork):

    def __init__(self,
                 temperature=1, # temperature of the gumbel sampling
                 force_hard=True, # mode of the gumbel sampling
                 **kwargs):

        super(GoalGlobal, self).__init__()
        VisualNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())

        self.PhysFeature = False
        self.need_decoder = True

        self.init_cnn()
        self.gumbelsampler = GumbelSampler(
            temp=self.temperature,
            grid_size_out=self.grid_size_out,
            force_hard=force_hard,
            scaling=self.img_scaling)
    def forward(self, features, h, pool_h=torch.empty(1)):
        cnn_out = self.CNN(features, h, pool_h)

        final_pos, final_pos_map_decoder, final_pos_map, y_softmax, y_scores = self.gumbelsampler(cnn_out)
        return final_pos, final_pos_map_decoder, final_pos_map, y_softmax, y_scores






class RoutingModule(nn.Module):
    """RoutingModule is part of TrajectoryGenerator and generates the prediction for each time step.
    The MotionDecoder consists of a LSTM network and a local goal network or attention network"""

    def __init__(
            self,
            seq_len=12,
            input_dim=2,
            decoder_h_dim=128,
            embedding_dim=64,
            dropout=0.0,
            batch_norm=False,
            mlp_dim=32,
            img_scaling_local=0.25,
            final_embedding_dim_rm=4,
            rm_vis_type="attention",
            grid_size_rm=8,
            dropout_cnn_rm=0.0,
            num_layers_rm=3,
            non_lin_cnn_rm="relu",
            force_hard_rm=True,
            temperature_rm=1,
            batch_norm_cnn_rm=False,
            noise_attention_dim_rm=True,
            skip_connection_rm=False,
            channels_cnn_rm=4,
            global_vis_type="goal"):
        """Initialise Motion Decoder network
                Parameters.
                    seq_len (int) - - Prediction length of trajectory
                    input_dim (int) - - input / output dimensionality of spatial coordinates
                    decoder_h_dim (int) - - hidden state dimenstion of decoder LSTM
                    embedding_dim (int) - - dimensionality spatial embedding
                    dropout (float) - - dropout
                    final_embedding_dim (int) - - embedding for final position estimate
                    mlp_dim (int) - - bottleneck dimensionality of mlp networks
                    PhysAtt (bool) - - depreciated. should not be used
                    device (torch.device) - - Choose device: cpu or gpu (cuda)
                    batch_norm (bool) - - if true, applies batch norm in mlp networks
                    img_scaling (float) - - ratio [m/px] between real and pixel space
                    grid_size (int) - - defines size of image path in goal / attention network (grid size is 2xgrid_size +1 )
                    decoder_type ("goal", "attention", none) - -
        """
        super(RoutingModule, self).__init__()

        self.__dict__.update(locals())
        if self.rm_vis_type:
            if self.rm_vis_type == "attention":
                self.rm_attention = AttentionRoutingModule(
                    channels_cnn=self.channels_cnn_rm,
                    decoder_h_dim=self.decoder_h_dim,
                    dropout=self.dropout_cnn_rm,
                    mlp_dim=self.mlp_dim,
                    img_scaling=self.img_scaling_local,
                    grid_size_in=self.grid_size_rm,
                    grid_size_out=self.grid_size_rm,
                    num_layers=self.num_layers_rm,
                    batch_norm_cnn=self.batch_norm_cnn_rm,
                    non_lin_cnn=self.non_lin_cnn_rm,
                    final_embedding_dim=final_embedding_dim_rm,
                    noise_attention_dim=self.noise_attention_dim_rm,
                    skip_connection=self.skip_connection_rm)
                self.final_embedding_dim_rm = self.rm_attention.final_embedding
            self.output_dim = self.decoder_h_dim + self.final_embedding_dim_rm

        elif not self.rm_vis_type:
            self.output_dim = self.decoder_h_dim

        else:
            assert False, "`{}` not valid for `decoder_type`: Choose `goal`, 'attention`, or none".format(decoder_type)

        self.final_output = make_mlp(
            [self.output_dim, self.mlp_dim, self.input_dim],
            activation_list=["relu", None],
            dropout=dropout,
            batch_norm=self.batch_norm)
        self.spatial_embedding = nn.Linear(self.input_dim, self.embedding_dim)

        if self.global_vis_type == "goal":
            self.input_dim_decoder = self.self.embedding_dim * 2 + 1

        else:
            self.input_dim_decoder = self.embedding_dim

        self.decoder = nn.LSTM(self.input_dim_decoder, self.decoder_h_dim)

    def forward(self, last_pos, rel_pos, state_tuple, dist_to_goal=0, scene_img=None):
        """ Calculates forward pass of MotionDecoder
            Parameters:
                obs_traj (tensor) - - Tensor of shape (obs_len, batch, 2)
                state_tuple (tuple of tensors) - - Tuple with hidden state (1, batch, encoder_h_dim) and cell state tensor (1, batch, encoder_h_dim)
            Returns:
                output (tensor) - - Output of LSTM netwok for all time steps (obs_len, batch, encoder_h_dim)
                final_h (tensor) - - Final hidden state of LSTM network (1, batch, encoder_h_dim)
        """

        batch_size = rel_pos.size(0)
        pred_traj_fake_rel = []
        pred_traj_fake = []
        softmax_list = []
        final_pos_list = []
        img_patch_list = []
        final_pos_map_decoder_list = []

        for t in range(self.seq_len):

            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(1, batch_size, self.embedding_dim)
            if self.global_vis_type != "none":
                distance_embeding = self.spatial_embedding(dist_to_goal)
                time_tensor = -1 + 2 * torch.ones(1, decoder_input.size(1), 1) * t / self.seq_len
                time_tensor = time_tensor.to(decoder_input)

                decoder_input = torch.cat((decoder_input, distance_embeding, time_tensor), -1)

            output, state_tuple = self.decoder(decoder_input, state_tuple)

            if self.rm_vis_type == "attention":
                final_emb, y_softmax, img_patch, noise = self.rm_attention(scene_img, last_pos, state_tuple[0])
            else:
                final_emb = torch.Tensor([]).to(state_tuple[0])
                img_patch = []

            input_final = torch.cat((state_tuple[0], final_emb), 2)

            img_patch_list.append(img_patch)

            # rel_pos = final_pos[0]

            rel_pos = self.final_output(input_final)
            rel_pos = rel_pos.squeeze(0)

            curr_pos = rel_pos + last_pos
            dist_to_goal = dist_to_goal - rel_pos
            pred_traj_fake_rel.append(rel_pos.clone().view(batch_size, -1))
            pred_traj_fake.append(curr_pos.clone().view(batch_size, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake = torch.stack(pred_traj_fake, dim=0)

        output = {"out_xy": pred_traj_fake,
                  "out_dxdy": pred_traj_fake_rel,
                  "h": state_tuple[0]}

        if self.rm_vis_type == "attention":
            output.update({"image_patches": torch.stack(img_patch_list, dim=0)})

        return output




class EncoderPrediction(nn.Module):
    """Part of Discriminator"""

    def __init__(
            self, input_dim=2,
            encoder_h_dim_d=128,
            embedding_dim=64,
            dropout=0.0,

            channels_cnn=4,
            grid_size=16,
            num_layers_cnn=2,
            batch_norm_cnn=True,
            batch_norm=False,
            dropout_cnn=0,
            mlp_dim=32,
            image_scaling=0.5,
            non_lin_cnn='tanh',
            visual_features = False):
        super().__init__()

        self.__dict__.update(locals())
        del self.self

        self.bottleneck_dim = int(grid_size / 2 ** (num_layers_cnn - 1)) ** 2 * channels_cnn * 2 ** (num_layers_cnn - 2)


        activation = ['leakyrelu', None]


        in_channels = 4
        self.bottleneck_dim = int(grid_size / 2 ** (num_layers_cnn - 1)) ** 2
        self.encoder_out = nn.Conv2d(channels_cnn * 2 ** (num_layers_cnn - 2), 1, kernel_size=(1, 1), stride=1)
        self.leakyrelu = nn.LeakyReLU()
        self.inputFeatures = make_mlp(
            [self.embedding_dim + self.bottleneck_dim, mlp_dim, self.embedding_dim],
            activation_list=['leakyrelu', None],
            dropout=dropout)


        self.encoder = nn.LSTM(self.embedding_dim, self.encoder_h_dim_d, dropout=dropout)
        if self.visual_features:
            self.CNN = CNN(channels_cnn=self.channels_cnn,
                           encoder_h_dim=self.encoder_h_dim_d,
                           mlp=self.mlp_dim,
                           need_decoder=False,
                           PhysFeature=True,
                           insert_trajectory=False,
                           grid_size_in=self.grid_size,
                           num_layers=self.num_layers_cnn,
                           dropout=self.dropout_cnn,
                           batch_norm=batch_norm_cnn,
                           non_lin_cnn=self.non_lin_cnn,
                           in_channels=in_channels,
                           )

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

        real_classifier_dims = [self.encoder_h_dim_d, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation_list=activation,
            dropout=dropout)

    def init_hidden(self, batch, obs_traj):
        return (torch.zeros(1, batch, self.encoder_h_dim_d).to(obs_traj),
                torch.zeros(1, batch, self.encoder_h_dim_d).to(obs_traj))

    def forward(self, dxdy, img_patch, state_tuple):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj_fake_rel: tensor of shape (self.seq_len, batch, 2)
        - pred_traj_fake: tensor of shape (self.seq_len, batch, 2)
        - state_tuple[0]: final hidden state
        """

        embedded_pos = self.spatial_embedding(dxdy)

        if self.visual_features:
            l, batch, c, x, y = img_patch.size()
            img_patch = img_patch.reshape(l * batch, c, x, y)
            cnn_out = self.CNN(img_patch)



            visual_features = self.leakyrelu(self.encoder_out(cnn_out["Features"]))
            visual_features = visual_features.view(l, batch, -1)


            encoder_input = torch.cat((embedded_pos, visual_features), -1)
            encoder_input = self.inputFeatures(encoder_input)
        else:
            encoder_input = embedded_pos
        output, input_classifier = self.encoder(encoder_input, state_tuple)
        dynamic_score = self.real_classifier(input_classifier[0])
        return dynamic_score



class get_gumbel_map(nn.Module):
    def __init__(self, grid_size):
        super(get_gumbel_map, self).__init__()

        x = torch.arange(0, grid_size * 2 + 1)
        x = x.unsqueeze(1)
        X = x.repeat(1, grid_size * 2 + 1)

        x1 = X - grid_size
        x2 = x1.T

        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)

        self.gumbel_map = torch.cat((x2, x1), 2).view(1, -1, 2)

    def forward(self, batch_size):
        gumbel_map = self.gumbel_map.repeat(batch_size, 1, 1).float()
        gumbel_map = gumbel_map + torch.rand_like(gumbel_map)
        return gumbel_map


### Gumbel Sampling ###

class GumbelSampler(nn.Module):

    def __init__(self,
                 temp=1,
                 grid_size_out=16,
                 scaling=0.5,
                 force_hard=True,
                 ):
        super(GumbelSampler, self).__init__()
        self.temp = temp
        self.grid_size_out = grid_size_out
        self.scaling = scaling
        self.gumbelsoftmax = GumbelSoftmax(temp=self.temp)
        self.gumbel_map = get_gumbel_map(grid_size=self.grid_size_out)
        self.force_hard = force_hard

    def forward(self, cnn_out):
        """

        :param cnn_out:
        :type cnn_out:
        :return:
            final_pos: Tensor with probability for each position
            final_pos_map: final_pos tensor reshaped
            y_softmax_gumbel: tensor with gumbel probabilities
            y_softmax: tensor with probabilites
        :rtype:
        """

        batch_size, c, hh, w = cnn_out["PosMap"].size()

        gumbel_map = self.gumbel_map(batch_size).to(cnn_out["PosMap"])
        y_scores = cnn_out["PosMap"].view(batch_size, -1)

        final_pos_map, y_softmax_gumbel, y_softmax = self.gumbelsoftmax(y_scores, force_hard=self.force_hard)

        final_pos = torch.sum(gumbel_map * final_pos_map.unsqueeze(2), 1).unsqueeze(0)

        final_pos_map = final_pos_map.view(batch_size, c, hh, w)
        y_softmax_gumbel = y_softmax_gumbel.view(batch_size, c, hh, w)
        y_softmax = y_softmax.view(batch_size, c, hh, w)
        final_pos = final_pos * self.scaling

        return final_pos, final_pos_map, y_softmax_gumbel, y_softmax, y_scores


class Patch_gen():
    def __init__(self, img_scaling=0.5,
                 grid_size=16,
                 type_img="small_image",
                 ):
        self.__dict__.update(locals())

    def get_patch(self, scene_image, last_pos):
        scale = 1. / self.img_scaling
        last_pos_np = last_pos.detach().cpu().numpy()

        image_list = []
        for k in range(len(scene_image)):
            image = scene_image[k][self.type_img]

            center = last_pos_np[k] * scale
            x_center, y_center = center.astype(int)
            cropped_img = image.crop(
                (int(x_center - self.grid_size), int(y_center - self.grid_size), int(x_center + self.grid_size + 1),
                 int(y_center + self.grid_size + 1)))

            cropped_img = -1 + torch.from_numpy(np.array(cropped_img) * 1.) * 2. / 256

            position = torch.zeros((1, self.grid_size * 2 + 1, self.grid_size * 2 + 1, 1))
            position[0, self.grid_size, self.grid_size, 0] = 1.
            image = torch.cat((cropped_img.float().unsqueeze(0), position), dim=3)

            image = image.permute(0, 3, 1, 2)
            image_list.append(image.clone())

        img = torch.cat(image_list)

        img = img.to(last_pos)

        return img


"""
Gumbel Softmax Sampler
Requires 2D input [batchsize, number of categories]

Does not support sinlge binary category. Use two dimensions with softmax instead.
"""


class GumbelSoftmax(nn.Module):
    def __init__(self, hard=False, temp=None):
        super(GumbelSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False
        self.temp = temp

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise).cuda()
        else:
            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, alpha, temperature, eps=1e-10):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = len(alpha.size()) - 1
        gumble_samples_tensor = self.sample_gumbel_like(alpha.data)

        gumble_trick_log_prob_samples = alpha + gumble_samples_tensor
        gumble_log_temp = gumble_trick_log_prob_samples / temperature
        max_gumble, _ = gumble_log_temp.max(1)
        soft_samples_gumble = F.softmax(gumble_log_temp - max_gumble.unsqueeze(1), dim)
        soft_samples_gumble = torch.max(soft_samples_gumble, torch.ones_like(soft_samples_gumble).to(alpha) * eps)
        soft_samples = F.softmax(alpha, dim)
        return soft_samples_gumble, soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        soft_samples_gumble, soft_samples = self.gumbel_softmax_sample(logits, temperature)
        if hard:

            _, max_value_indexes = soft_samples_gumble.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)

            y = y_hard - soft_samples_gumble.data + soft_samples_gumble

        else:
            y = soft_samples_gumble
        return y, soft_samples_gumble, soft_samples

    def forward(self, alpha, temp=None, force_hard=False):
        if not temp:
            if self.temp:
                temp = self.temp
            else:
                temp = 1

        if self.training and not force_hard:

            return self.gumbel_softmax(alpha, temperature=temp, hard=False)
        else:

            return self.gumbel_softmax(alpha, temperature=temp, hard=True)


if __name__ == "__main__":

    print("Test Encoder")
    print(MotionEncoder())

    print("Test Decoder")
    print(RoutingModule())

    print("Test AttentionRoutingModule")
    print(AttentionRoutingModule())

    print("Test AttentionGlobal")
    print(AttentionGlobal())

    print("Test GoalGlobal")
    print(GoalGlobal() )

    print("Test Encoder Discriminator")
    print(EncoderPrediction())