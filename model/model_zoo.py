"""
This script implements GoalGAN
"""

import sys, os, inspect
sys.path.append(os.path.abspath( "."))

import torch
import torch.nn as nn
import traceback

from model.model_modules import make_mlp,  \
     RoutingModule, MotionEncoder, AttentionGlobal,\
     GoalGlobal, EncoderPrediction

# BaseModel
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    def test(self):
        self.eval()
        self.mode = "test"

    def gen(self):
        self.mode = "gen"
        self.train()



"""
Implementation of the generator and discriminator of Goal GAN. 
Hyperparamters of the models are set in 'config/model/model.yaml' and 'experiment'
"""

# GOAL GAN

class GoalGAN(BaseModel):
    """
    Implementation of generator of GOAL GAN
    The model consists out of three main components:
    1. Motion Encoder
    2. Goal Module
    3. Routing Module with visual soft-attention

    """
    def __init__(
            self,
            dropout: float = 0.0,
            batch_norm: bool = False,
            input_dim: int = 2,  #
            pred_len: int = 12,

            # Generator dim
            encoder_h_dim_g: int = 16,
            decoder_h_dim_g: int = 16,

            mlp_dim: int = 32,
            embedding_dim: int = 8,
            load_occupancy: bool = False,
            pretrain_cnn : int = 0,

            # parameters global goal / attention
            temperature_global: int = 1,
            grid_size_in_global: int = 32,
            grid_size_out_global: int = 32,
            num_layers_cnn_global: int = 3,
            batch_norm_cnn_global: bool = True,
            dropout_cnn_global: float = 0.3,
            non_lin_cnn_global: str = "relu",
            scaling_global: float = 1.,
            force_hard_global: bool = True,
            final_embedding_dim_global: int = 4,
            skip_connection_global: bool = False,
            channels_cnn_global: int = 4,
            global_vis_type: str = "goal",

            # parameters routing module
            rm_vis_type : str = "attention",
            num_layers_cnn_rm: int = 3,
            batch_norm_cnn_rm=True,
            dropout_cnn_rm=0.0,
            non_lin_cnn_rm="relu",
            grid_size_rm=16,
            scaling_local=0.20,
            force_hard_rm=True,
            noise_attention_dim_rm=8,
            final_embedding_dim_rm=4,
            skip_connection_rm = False,
            channels_cnn_rm = 4,

            **kwargs
         ):


        super().__init__()
        self.__dict__.update(locals())
        self.cnn = True


        self.scaling = (2 * grid_size_in_global + 1) / (2 * grid_size_out_global + 1) * self.scaling_global

        # 1. Motion Encoder
        self.encoder = MotionEncoder(
            encoder_h_dim=self.encoder_h_dim_g,
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        )

       # 2. Goal Module
        if self.global_vis_type == "goal":

            self.goalmodule = GoalGlobal(
                channels_cnn = self.channels_cnn_global,
                decoder_h_dim=self.decoder_h_dim_g,
                dropout=self.dropout_cnn_global,
                batch_norm=self.batch_norm,
                mlp_dim=self.mlp_dim,
                img_scaling=self.scaling,
                final_embedding_dim=self.final_embedding_dim_global,
                grid_size_in=self.grid_size_in_global,
                grid_size_out=self.grid_size_out_global,
                num_layers=self.num_layers_cnn_global,
                batch_norm_cnn=self.batch_norm_cnn_global,
                non_lin_cnn=self.non_lin_cnn_global,
                temperature = self.temperature_global,
                force_hard = self.force_hard_global,
                skip_connection = self.skip_connection_global)

        elif self.global_vis_type == "attention":
            self.global_attention = AttentionGlobal(
                channels_cnn=self.channels_cnn_global,
                decoder_h_dim=self.decoder_h_dim_g,
                dropout=self.dropout_cnn_global,
                batch_norm=self.batch_norm,
                mlp_dim=self.mlp_dim,
                img_scaling=self.scaling,
                final_embedding_dim=self.final_embedding_dim_global,
                grid_size_in=self.grid_size_in_global,
                grid_size_out=self.grid_size_out_global,
                num_layers=self.num_layers_cnn_global,
                batch_norm_cnn=self.batch_norm_cnn_global,
                non_lin_cnn=self.non_lin_cnn_global,
                skip_connection = self.skip_connection_global)

        self.final_pos_embedding = make_mlp(
            [self.input_dim, self.final_embedding_dim_global],
            activation_list=['tanh'],
            dropout=self.dropout)


        # 3. Routing Module
        self.routingmodule = RoutingModule(seq_len=self.pred_len,
                                    input_dim=self.input_dim,
                                    embedding_dim=self.embedding_dim,
                                    decoder_h_dim=self.decoder_h_dim_g,
                                    dropout=self.dropout,
                                    grid_size_rm = self.grid_size_rm,
                                    final_embedding_dim_rm=self.final_embedding_dim_rm,
                                    img_scaling_local=self.scaling_local,
                                    mlp_dim=self.mlp_dim,
                                    dropout_cnn_rm = self.dropout_cnn_rm,
                                    num_layers_rm= self.num_layers_cnn_rm,
                                    non_lin_cnn_rm=self.non_lin_cnn_rm,
                                    batch_norm_cnn_rm= self.batch_norm_cnn_rm,
                                    noise_attention_dim_rm = self.noise_attention_dim_rm,
                                    rm_vis_type = self.rm_vis_type,
                                    skip_connection_rm = self.skip_connection_rm,
                                    channels_cnn_rm = self.channels_cnn_rm,
                                    global_vis_type = self.global_vis_type
                                )




        if self.mlp_decoder_needed():


            h_dim = self.encoder_h_dim_g
            self.encoder2decoder = make_mlp(
                [h_dim + self.final_embedding_dim_global, self.mlp_dim, self.decoder_h_dim_g],
                activation_list=['relu', 'tanh'],
                dropout=self.dropout)


    def init_c(self, batch_size):
        return torch.zeros((1, batch_size, self.decoder_h_dim_g))

    def mlp_decoder_needed(self):
        if (
                (self.encoder_h_dim_g + self.final_embedding_dim_global) != self.decoder_h_dim_g
        ):
            return True
        else:
            return False


    def forward(self, batch, final_pos_in=torch.Tensor()):
        batch_size = batch["in_xy"].size(1)
        encoder_out, h_enc = self.encoder(batch["in_dxdy"])

        final_pos, final_pos_map_concrete, final_pos_map, y_softmax, y_scores = self.goalmodule(batch["global_patch"],h_enc)
        out = {
            "y_map": final_pos_map_concrete,
            "y_softmax": y_softmax,
            "final_pos": final_pos,
            "y_scores": y_scores
        }
        if self.mode == "pretrain":
            return out

        if len(final_pos_in) > 0:
            final_pos = final_pos_in

        final_pos_embedded = self.final_pos_embedding(final_pos)
        h = self.encoder2decoder(torch.cat((h_enc, final_pos_embedded), 2))
        c = self.init_c(batch_size).to(batch["in_xy"])

        # last position
        x0 = batch["in_xy"][-1]
        v0 = batch["in_dxdy"][-1]
        state_tuple = (h, c)
        out.update(self.routingmodule(last_pos=x0, dist_to_goal=final_pos, rel_pos=v0, state_tuple=state_tuple,
                                scene_img=batch["scene_img"]))

        return {**out,
                "h_encoder": h_enc,
                }


#
class GoalGANDiscriminator(BaseModel):
    """Implementation of discriminator of GOAL GAN

       The model consists out of three main components:
       1. encoder of input trajectory
       2. encoder of
       3. Routing Module with visual soft-attention

       """
    def __init__(
        self,
        float_type = torch.float64,
        encoder_h_dim_d=64,
        mlp_dim=1024,
        embedding_dim = 16,
        dropout_disc=0.0,
        input_dim = 2,
        grid_size_local = 16,
        num_layers_cnn_disc=3,
        batch_norm_disc=False,
        dropout_cnn_disc=0,
        non_lin_cnn_disc = "tanh",
        channels_cnn_disc= 16,
        visual_features_disc = True,
        scaling_local = 0.20,
        **kwargs
        ):

        super().__init__()

        self.__dict__.update(locals())

        self.grad_status = True
        self.type(float_type)


        self.encoder_observation =  MotionEncoder(
                        encoder_h_dim=self.encoder_h_dim_d,
                        input_dim=self.input_dim,
                        embedding_dim = self.embedding_dim,
                        dropout=self.dropout_disc
                        )


        self.EncoderPrediction = EncoderPrediction(
                            input_dim=self.input_dim,
                            encoder_h_dim_d=self. encoder_h_dim_d,
                            embedding_dim=self.embedding_dim,
                            dropout=self.dropout_disc,
                            batch_norm=False,
                            batch_norm_cnn = self.batch_norm_disc,
                            dropout_cnn = self.dropout_cnn_disc,
                            non_lin_cnn = self.non_lin_cnn_disc,
                            channels_cnn = self.channels_cnn_disc,
                            grid_size = self.grid_size_local,
                            num_layers_cnn = self.num_layers_cnn_disc,
                            image_scaling = self.scaling_local,
                            visual_features = self.visual_features_disc

                            )



    def init_c(self, batch_size):
        return torch.zeros((1, batch_size, self.encoder_h_dim_d))


    def forward(self, in_xy, in_dxdy , out_xy, out_dxdy, images_patches = False):


        output_h , h  = self.encoder_observation(in_dxdy)
        batch_size = in_xy.size(1)
        c = self.init_c(batch_size).to(in_xy)
        state_tuple = (h, c)


        dynamic_scores = self.EncoderPrediction(out_dxdy, images_patches, state_tuple)


        return dynamic_scores

    def grad(self, status):
        if not self.grad_status == status:
            self.grad_status = status
            for p in self.parameters():
                p.requires_grad = self.grad_status


if __name__ == "__main__":

    Generator = GoalGAN()
    print("Generator")
    print(Generator)

    Discriminator = GoalGANDiscriminator()
    print("Discriminator")
    print(Discriminator)
