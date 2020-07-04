"""
This file runs the main training/val loop, etc... using Lightning Trainer

more details at:
 https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#training-loop-structure
"""

from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
import random

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger


from models.multicam_semisupervised import MonoSemiSupDepth_Packnet
from utils.config_utils import parse_yaml
from utils.colored_terminal_logging_utils import get_terminal_logger



terminal_logger = get_terminal_logger(__name__)


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(gpus, nodes, fast_dev_run, mixed_precision, project_config, hparams):
    torch.manual_seed(0)
    np.random.seed(0)

    # init module
    model = MonoSemiSupDepth_Packnet(hparams)

    # tags associated to the run
    def shape_format(shape):
        # shape = [Height, Width]
        return f"{shape[1]}x{shape[0]}"

    list_of_tags = [
        hparams.model.depth_net.name,
        hparams.model.pose_net.name,
        hparams.optimizer.name,
        hparams.scheduler.name,
        {1: 'gray', 3: 'rgb'}[hparams.input_channels],
        f"train-{shape_format(hparams.datasets.train.data_transform_options.image_shape)}",
        f"val-{shape_format(hparams.datasets.val.data_transform_options.image_shape)}",
        f"test-{shape_format(hparams.datasets.test.data_transform_options.image_shape)}",
    ]
    if mixed_precision:
        list_of_tags += 'mixed_precision'

    base_output_dir = Path(project_config.output_dir)
    experiment_output_dir = base_output_dir / project_config.project_name / project_config.experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    wandb_output_dir = str(experiment_output_dir)
    wandb_logger = WandbLogger(
        project = project_config.project_name,
        save_dir=wandb_output_dir, # the path to a directory where artifacts will be written
        log_model=True,
        tags=list_of_tags
    )
    #wandb_logger.watch(model, log='all', log_freq=5000) # watch model's gradients and params

    run_output_dir = experiment_output_dir / f'{wandb_logger.experiment.id}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = str(run_output_dir)

    checkpoint_callback = ModelCheckpoint(
        filepath=run_output_dir + '/{epoch:04d}-{val-abs_rel:.5f}', # saves a file like: my/path/epoch=2-abs_rel=0.0115.ckpt
        save_top_k=3,
        verbose=True,
        monitor='val-abs_rel',
        mode='min',
    )

    lr_logger = LearningRateLogger()


    if mixed_precision:
        amp_level='01'
        precision=16

    if gpus > 1:
        distributed_backend = 'ddp'
    else:
        distributed_backend = None

    profiler = False
    if fast_dev_run:
        from pytorch_lightning.profiler import AdvancedProfiler
        profiler = AdvancedProfiler(output_filename='./profiler.log')

    trainer = Trainer(
        gpus=gpus,
        distributed_backend=distributed_backend,
        nb_gpu_nodes=nodes,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_logger],
        logger=wandb_logger,
        fast_dev_run=fast_dev_run,
        profiler=profiler,
        early_stop_callback=False,
        #amp_level='O1',
        #precision=16,
        **hparams.trainer
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--model_config_file', type=str)
    parser.add_argument('--model_config_profile', type=str)
    parser.add_argument('--project_config_file', type=str)
    parser.add_argument('--project_config_profile', type=str)
    parser.add_argument("--fast_dev_run", action="store_true", help="if flag given, runs 1 batch of train, test and val to find any bugs")
    parser.add_argument("--mixed_precision", action="store_true", help="if flag given, train with mixed precision.")

    # parse params
    args = parser.parse_args()

    hparams = parse_yaml(args.model_config_file, args.model_config_profile)
    project_config = parse_yaml(args.project_config_file, args.project_config_profile)

    set_random_seed(hparams.seed)

    main(args.gpus, args.nodes, args.fast_dev_run, args.mixed_precision, project_config, hparams)