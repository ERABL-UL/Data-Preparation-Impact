from os.path import join
import numpy as np

def save(params):

    with open(join(params.saving_path, 'parameters.txt'), "w") as text_file:

        text_file.write('# -----------------------------------#\n')
        text_file.write('# Parameters of the training session #\n')
        text_file.write('# -----------------------------------#\n\n')

        # Input parameters
        text_file.write('# Input parameters\n')
        text_file.write('# ****************\n\n')
        text_file.write('dataset = {:s}\n'.format(params.dataset))
        text_file.write('dataset_task = {:s}\n'.format(params.dataset_task))
        if type(params.num_classes) is list:
            text_file.write('num_classes =')
            for n in params.num_classes:
                text_file.write(' {:d}'.format(n))
            text_file.write('\n')
        else:
            text_file.write('num_classes = {:d}\n'.format(params.num_classes))
        text_file.write('in_points_dim = {:d}\n'.format(params.in_points_dim))
        text_file.write('in_features_dim = {:d}\n'.format(params.in_features_dim))
        text_file.write('in_radius = {:.6f}\n'.format(params.in_radius))
        text_file.write('input_threads = {:d}\n\n'.format(params.input_threads))

        # Model parameters
        text_file.write('# Model parameters\n')
        text_file.write('# ****************\n\n')

        text_file.write('architecture =')
        for a in params.architecture:
            text_file.write(' {:s}'.format(a))
        text_file.write('\n')
        text_file.write('equivar_mode = {:s}\n'.format(params.equivar_mode))
        text_file.write('invar_mode = {:s}\n'.format(params.invar_mode))
        text_file.write('num_layers = {:d}\n'.format(params.num_layers))
        text_file.write('first_features_dim = {:d}\n'.format(params.first_features_dim))
        text_file.write('use_batch_norm = {:d}\n'.format(int(params.use_batch_norm)))
        text_file.write('batch_norm_momentum = {:.6f}\n\n'.format(params.batch_norm_momentum))
        text_file.write('segmentation_ratio = {:.6f}\n\n'.format(params.segmentation_ratio))

        # KPConv parameters
        text_file.write('# KPConv parameters\n')
        text_file.write('# *****************\n\n')

        text_file.write('first_subsampling_dl = {:.6f}\n'.format(params.first_subsampling_dl))
        text_file.write('num_kernel_points = {:d}\n'.format(params.num_kernel_points))
        text_file.write('conv_radius = {:.6f}\n'.format(params.conv_radius))
        text_file.write('deform_radius = {:.6f}\n'.format(params.deform_radius))
        text_file.write('fixed_kernel_points = {:s}\n'.format(params.fixed_kernel_points))
        text_file.write('KP_extent = {:.6f}\n'.format(params.KP_extent))
        text_file.write('KP_influence = {:s}\n'.format(params.KP_influence))
        text_file.write('aggregation_mode = {:s}\n'.format(params.aggregation_mode))
        text_file.write('modulated = {:d}\n'.format(int(params.modulated)))
        text_file.write('n_frames = {:d}\n'.format(params.n_frames))
        text_file.write('max_in_points = {:d}\n\n'.format(params.max_in_points))
        text_file.write('max_val_points = {:d}\n\n'.format(params.max_val_points))
        text_file.write('val_radius = {:.6f}\n\n'.format(params.val_radius))

        # Training parameters
        text_file.write('# Training parameters\n')
        text_file.write('# *******************\n\n')

        text_file.write('learning_rate = {:f}\n'.format(params.learning_rate))
        text_file.write('momentum = {:f}\n'.format(params.momentum))
        text_file.write('lr_decay_epochs =')
        for e, d in params.lr_decays.items():
            text_file.write(' {:d}:{:f}'.format(e, d))
        text_file.write('\n')
        text_file.write('grad_clip_norm = {:f}\n\n'.format(params.grad_clip_norm))


        text_file.write('augment_symmetries =')
        for a in params.augment_symmetries:
            text_file.write(' {:d}'.format(int(a)))
        text_file.write('\n')
        text_file.write('augment_rotation = {:s}\n'.format(params.augment_rotation))
        text_file.write('augment_noise = {:f}\n'.format(params.augment_noise))
        text_file.write('augment_occlusion = {:s}\n'.format(params.augment_occlusion))
        text_file.write('augment_occlusion_ratio = {:.6f}\n'.format(params.augment_occlusion_ratio))
        text_file.write('augment_occlusion_num = {:d}\n'.format(params.augment_occlusion_num))
        text_file.write('augment_scale_anisotropic = {:d}\n'.format(int(params.augment_scale_anisotropic)))
        text_file.write('augment_scale_min = {:.6f}\n'.format(params.augment_scale_min))
        text_file.write('augment_scale_max = {:.6f}\n'.format(params.augment_scale_max))
        text_file.write('augment_color = {:.6f}\n\n'.format(params.augment_color))

        text_file.write('weight_decay = {:f}\n'.format(params.weight_decay))
        text_file.write('segloss_balance = {:s}\n'.format(params.segloss_balance))
        text_file.write('class_w =')
        for a in params.class_w:
            text_file.write(' {:.6f}'.format(a))
        text_file.write('\n')
        text_file.write('deform_fitting_mode = {:s}\n'.format(params.deform_fitting_mode))
        text_file.write('deform_fitting_power = {:.6f}\n'.format(params.deform_fitting_power))
        text_file.write('deform_lr_factor = {:.6f}\n'.format(params.deform_lr_factor))
        text_file.write('repulse_extent = {:.6f}\n'.format(params.repulse_extent))
        text_file.write('batch_num = {:d}\n'.format(params.batch_num))
        text_file.write('val_batch_num = {:d}\n'.format(params.val_batch_num))
        text_file.write('max_epoch = {:d}\n'.format(params.max_epoch))
        if params.epoch_steps is None:
            text_file.write('epoch_steps = None\n')
        else:
            text_file.write('epoch_steps = {:d}\n'.format(params.epoch_steps))
        text_file.write('validation_size = {:d}\n'.format(params.validation_size))
        text_file.write('checkpoint_gap = {:d}\n'.format(params.checkpoint_gap))

