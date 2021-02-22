import torch
import random
from matplotlib.path import Path
import numpy as np
import torch.nn as nn



class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, random_label = 1, delta_rand = 0.15,reduction = "mean" ):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgan.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.register_buffer('delta_rand', torch.tensor(delta_rand))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss(reduction = reduction)
            self.register_buffer('random_label', torch.tensor(0))
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction = reduction)
            self.register_buffer('random_label', torch.tensor(random_label))
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.register_buffer('random_label', torch.tensor(0))
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if self.random_label:
            if target_is_real:
                labels = torch.FloatTensor(prediction.size()).uniform_(self.real_label - self.delta_rand, self.real_label)
            else:
                labels = torch.FloatTensor(prediction.size()).uniform_(self.fake_label,self.fake_label +  self.delta_rand)
        else:

            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            labels = target_tensor.expand_as(prediction)
        return labels

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.

        Example:

            criterionGAN = GANLoss(gan_mode).to(device)
            loss_D_real = self.criterionGAN(self.discriminator_realfake(faces), True)  # give True (1) for real samples
            loss_D_fake = self.criterionGAN(self.discriminator_realfake(generated_images), False)  # give False (0) for generated samples
            loss_D = loss_D_real + loss_D_fake

            loss_G = self.criterionGAN(self.discriminator_realfake(generated_images), True)  # give True (1) labels for generated samples, aka try to fool D
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).to(prediction)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction
            else:
                loss = prediction
        return loss

def cal_gradient_penalty(netD, real_data, fake_data , device, type, constant, lambda_gp):

    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
            *real_data.shape)
        alpha = alpha.to(device)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
    return gradient_penalty





def l2_loss(pred_traj, pred_traj_gt, mode='average', type = "mse"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    # disabled loss mask
    #loss_mask = 1
    seq_len, batch, _ = pred_traj.size()
    #loss_mask = loss_mask.permute(2, 1,0)[:, :, 0]
    d_Traj = pred_traj_gt - pred_traj

    if type =="mse": 
        loss = torch.norm( (d_Traj), 2, -1)
    elif type =="average":
      
        loss = (( torch.norm( d_Traj, 2, -1)) + (torch.norm( d_Traj[-1], 2, -1)))/2.
       
    else: 
        raise AssertionError('Mode {} must be either mse or  average.'.format(type))


    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
     
        return torch.mean(loss, dim =0 )
    elif mode == 'raw':
        return loss.sum(dim=0)


def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, num_traj, _ = pred_traj.size()
    
    loss = pred_traj_gt - pred_traj

    loss = torch.norm(loss, 2, 2).unsqueeze(0)
    
    
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average': 
        return torch.sum(loss)/( seq_len * num_traj)
    elif mode == 'raw':
     
        return torch.sum(loss, 1)


def final_displacement_error(
    pred_pos, pred_pos_gt, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos

    Output:
    - loss: gives the eculidian displacement error
    """
    num_traj, _ = pred_pos_gt.size() 
    loss = pred_pos_gt - pred_pos

    loss = torch.norm(loss, 2, 1).unsqueeze(0)
    if mode == 'raw':
        
        return loss
    elif mode == 'average':
        return torch.sum( loss) / num_traj
    elif mode == 'sum':
        return torch.sum(loss)


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, mode='sum'
    )
   
    return g_l2_loss_abs

    


def cal_ade(pred_traj_gt, pred_traj_fake, mode = "sum"):

    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode = mode)

    return ade


def cal_fde(
    pred_traj_gt, pred_traj_fake, mode = "sum"):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1],  mode = mode )
    return fde


def crashIntoWall( traj, walls):
    length, batch, dim = traj.size()


    wall_crashes = []
    for i in range(batch):
        t = traj[:, i, :]
        for wall in walls[i]:

            polygon = Path(wall)

            wall_crashes.append(1* polygon.contains_points(t).any())
    return wall_crashes
