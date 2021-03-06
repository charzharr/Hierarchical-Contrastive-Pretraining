""" Module run.py (By: Charley Zhang, Nov 2020)
Is called by shell scripts to run experiments.
Main job: call appropriate experiment main.py file and pass on env args.
    (1) Get config file & validate settings
    (2) Parse GPU device info
    (3) Set experiment seed
    (4) Run experiment via the corresponding emain.py

Updates
-------
(2020.11)
  - Added negative seeds for random seed sampling.
  - Added sun grid array job submissions. 
  - Improved exception catching in main experiment to encompass all exceptions.
"""

import sys, os
import pathlib
import signal
import math, random
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)},
                    suppress=True,
                    precision=3,
                    linewidth=150)

import wandb
import click
import torch
import copy

import lib
from lib.utils.train.configs import get_config

# Experiment code modules
import experiments
from experiments.pgl import run_experiment as run_pgl
from experiments.ftbcv import run_experiment as run_ftbcv
from experiments.ftmmwhs import run_experiment as run_mmwhs
from experiments.ftspleen import run_experiment as run_spleen

EXPERIMENTS = {
    'pgl': run_pgl,
    'ftbcv': run_ftbcv,
    'ftmmwhs': run_mmwhs,
    'ftspleen': run_spleen
}


@click.command()
@click.option('--config', required=True, type=click.Path(exists=False))
@click.option('-ddp', '--distributed', is_flag=True, 
              help='Flag to indicate whether to use distributed training.')
def run_cli(config, distributed):
    
    # --- ##  Get experiment configuration  ## --- #
    given_cfg_path = pathlib.Path(config)
    if given_cfg_path.exists():
        print(f'[CFG] Given cfg file "{str(given_cfg_path)}" exists! Loading..')
        cfg = get_config(config, merge_default=False, search_dir='')
    else:
        print(f'[CFG] Given cfg file "{str(given_cfg_path)}" does not exist! '
              'Searching for matching name in experiment\'s config folder..')
        curr_path = pathlib.Path(__file__).parent.absolute()
        exp_cfg_path = None
        for exp in EXPERIMENTS:
            if exp in given_cfg_path.name:
                exp_cfg_path = str(curr_path / 'experiments' / exp / 'configs')
                print(f' ??? Cfg experiment matched at {exp_cfg_path}.')
                break

        if not exp_cfg_path:
            msg = (f'Given config file "{config}" does not contain any of the '
                   f'experiment names in them: {list(EXPERIMENTS.keys())}')
            raise ValueError(msg)
        cfg = cfg = get_config(config, merge_default=False, 
                               search_dir=exp_cfg_path)
    
    # --- ##  Handle array job submissions '-t'  ## --- #
    env_vars = dict(os.environ)
    task_id = env_vars['SGE_TASK_ID']
    
    if task_id != 'undefined':  
        # Change run number if experiment name ends with 'r\d' (e.g. _r1, _r2)
        exp_id = cfg.experiment.id
        matched = re.findall(r'r\d+', exp_id)
        if matched:
            cfg.experiment.id = exp_id.replace(matched[-1], f'r{task_id}')
        else:
            cfg.experiment.id = exp_id + f'_r{task_id}'
        print(f'[QSUB -t] Array job detected! Changing exp_id from {exp_id} to '
              f'{cfg.experiment.id}')
    else:
        print(f'[QSUB -t] No array job detected! exp_id remains '
              f'{cfg.experiment.id}')
    
    # --- ##  GPU device parsing and distributed training init  ## --- #
    gpu_indices = []
    gpu_env = os.getenv('SGE_HGR_gpu_card')
    if gpu_env:
        gpu_indices = [int(i) for i in gpu_env.strip().split(' ')]
        gpu_str_indices = [str(i) for i in gpu_indices]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_str_indices)
        print(f'[GPUs] Cuda Visible: {os.getenv("CUDA_VISIBLE_DEVICES")}')
    
    cfg.experiment.gpu_idxs = gpu_indices
    if distributed and len(gpu_indices) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'              #
        os.environ['MASTER_PORT'] = '8888'
        cfg.experiment.distributed = True
    else:
        cfg.experiment.distributed = False
        cfg.experiment.rank = 0
        device = f'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.experiment.device = device
        print(f'[GPUs] Using device(s) with id(s): {gpu_indices}.')

    # --- ##  Final setup  ## --- #
    
    # Config checking & automatic value adjustments based on training env
    parse_cfg(cfg)
    
    # Experiment-specific arguments to pass to run_experiment.py
    exp_run_args = []
    experiment_main = EXPERIMENTS[cfg['experiment']['name']]
    
    if cfg.experiment.name in ('ftbcv', 'ftmmwhs', 'ftspleen'):
        exp_run_args.append(torch.multiprocessing.Queue())

    # Run within exception wrapper so processes can end gracefully
    try:
        if cfg.experiment.distributed:
            print(f'[RUN] Using distributed. Spawning {len(gpu_indices)} '
                  'processes.')
            spawn_args = tuple([cfg] + exp_run_args)
            torch.multiprocessing.spawn(
                experiment_main.run, 
                args=spawn_args,
                nprocs=len(gpu_indices))
        else:
            experiment_main.run(0, cfg, *exp_run_args)  # rank 0
    except BaseException as err:
        print(f'\nException thrown:\n', '-' * 30, f'\n{err}', sep='')
        print(f'\nTraceback:\n', '-' * 30, sep='')
        import traceback
        traceback.print_exc()

        print('\n\n' + '*' * 80 + '\n[END] Program Exit Cleanup Initiated!\n')
        kill_children()
        if cfg.experiment.distributed:
            torch.distributed.destroy_process_group()
    finally:
        print('???? Ended ????\n')
        


def kill_children():
    print(f'[END] Kill the kids!')
    import psutil
    child_processes = psutil.Process().children(recursive=True)
    for child in child_processes:
        print(f'[END] > Killing child process (PID={child.pid})')
        child.kill()


def parse_cfg(cfg):
    # Make sure 'best' is not in the experiment id or name
    if 'best' in cfg.experiment.id:
        raise ValueError(f'"best" cannot be in the experiment id!')
    if 'best' in cfg.experiment.name:
        raise ValueError(f'"best" cannot be in the experiment name!')

    # Adjust batch sizes based on models
    N_gpus = len(cfg.experiment.gpu_idxs)
    if cfg.experiment.name == 'ftmmwhs':
        changed = False
        if cfg.model.name in ('nnunet3d'):
            new_batch_size = 3
            if cfg.train.patch_size in ([64, 64, 64],):
                new_batch_size = 12
                cfg.train.optimizer.lr = cfg.train.optimizer.lr * 4
            elif cfg.train.patch_size in ([32, 128, 128],):
                new_batch_size = 6
                cfg.train.optimizer.lr = cfg.train.optimizer.lr * 2
            tr_batch = cfg.train.batch_size
            cfg.train.batch_size = new_batch_size
            if 'test' in cfg:
                te_batch = cfg.test.batch_size
                cfg.test.batch_size = new_batch_size
            changed = True
        elif cfg.model.name in ('denseunet3d'):
            new_batch_size = 4
            if cfg.train.patch_size in ([64, 64, 64],):
                new_batch_size = 16
                cfg.train.optimizer.lr = cfg.train.optimizer.lr * 4
            elif cfg.train.patch_size in ([32, 128, 128],):
                new_batch_size = 8
                cfg.train.optimizer.lr = cfg.train.optimizer.lr * 2
            elif cfg.train.patch_size in ([32, 192, 192],):
                # new_batch_size = 3 if cfg.train.deep_sup else 4
                new_batch_size = 4
            tr_batch = cfg.train.batch_size
            cfg.train.batch_size = new_batch_size
            if 'test' in cfg:
                te_batch = cfg.test.batch_size
                cfg.test.batch_size = new_batch_size
            changed = True
        elif cfg.model.name in ('dvn3d'):
            new_batch_size = 3
            if cfg.train.patch_size in ([64, 64, 64],):
                new_batch_size = 12
                cfg.train.optimizer.lr = cfg.train.optimizer.lr * 4
            elif cfg.train.patch_size in ([32, 128, 128],):
                new_batch_size = 6
                cfg.train.optimizer.lr = cfg.train.optimizer.lr * 2
            tr_batch = cfg.train.batch_size
            cfg.train.batch_size = new_batch_size
            if 'test' in cfg:
                te_batch = cfg.test.batch_size
                cfg.test.batch_size = new_batch_size
            changed = True

        if changed:  # print changes
            print(f' Adjusting batch sizes based on model used:')
            print(f'  Train Batch: {tr_batch} -> {cfg.train.batch_size}')
            if 'test' in cfg:
                print(f'  Test Batch: {te_batch} -> {cfg.test.batch_size}')
    
    """
    if N_gpus > 0 and cfg.experiment.name == 'ftbcv':
        changed = False
        if 'custom_dense' in cfg.model.name:
            tr_batch = cfg.train.batch_size
            cfg.train.batch_size = 4
            if 'test' in cfg:
                te_batch = cfg.test.batch_size
                cfg.test.batch_size = 4
            changed = True
        elif 'dense' in cfg.model.name:
            tr_batch = cfg.train.batch_size
            cfg.train.batch_size = 3
            if 'test' in cfg:
                te_batch = cfg.test.batch_size
                cfg.test.batch_size = 4
            changed = True
        elif 'unet' in cfg.model.name:
            tr_batch = cfg.train.batch_size
            cfg.train.batch_size = 2
            if 'test' in cfg:
                te_batch = cfg.test.batch_size
                cfg.test.batch_size = 3
            changed = True

        if changed:  # print changes
            print(f' Adjusting batch sizes based on model used:')
            print(f'  Train Batch: {tr_batch} -> {cfg.train.batch_size}')
            if 'test' in cfg:
                print(f'  Test Batch: {te_batch} -> {cfg.test.batch_size}')
    """

    # Adjust multi-gpu parameters
    if not cfg.experiment.distributed and N_gpus > 1:
        print(f'* {N_gpus} GPUs detected! Adjusting batch size and LR:')
        
        # Train and test (if exists) batch sizes
        tr_batch = cfg.train.batch_size
        cfg.train.batch_size = tr_batch * N_gpus
        print(f'  Train Batch: {tr_batch} -> {cfg.train.batch_size}')
        
        if 'test' in cfg and 'batch_size' in cfg.test:
            te_batch = cfg.test.batch_size
            cfg.test.batch_size = te_batch * N_gpus - 1
            print(f'  Test Batch: {te_batch} -> {cfg.test.batch_size}')
        
        # Data num_workers adjustment
        if cfg.train.num_workers > 0:
            inc = 2 if N_gpus == 2 else 3
            orig_workers = cfg.train.num_workers
            cfg.train.num_workers = orig_workers + inc
            print(f'  # Workers: {orig_workers} -> {cfg.train.num_workers}')

        # LR adjustment
        # old_lr = cfg.train.optimizer.lr
        # print(f'  LR: {old_lr} -> {N_gpus * old_lr}')
        # cfg.train.optimizer.lr = old_lr * N_gpus

    # Adjust cfg if overfit minibatch
    # if cfg.experiment.debug.overfitbatch:
    #     print(f'???? Overfitting a set of minibatches!')
    #     print(f'    Start-Epoch: {cfg.train.start_epoch} -> {0}')
    #     print(f'    Train-Epochs: {cfg.train.epochs} -> {40}')
    #     cfg.train.start_epoch = 0
    #     cfg.train.epochs = 40


def set_seed(seed):
    if seed >= 0:
        print(f'[SEED] Setting seed to {seed}.')
    else:
        seed = random.randrange(2 ** 20)
        print(f'[SEED] Random seed not give, set to: {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # fixed input/model: ~5-10% speedup


def setup_dist(rank, world_size):
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )


if __name__ == '__main__':
    run_cli()