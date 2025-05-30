import argparse
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, update_cfg_by_args, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

torch.backends.cudnn.benchmark = True


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    # basic training options
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    # loading options
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--continue_train', action='store_true', default=False)
    # distributed options
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False, help='whether to find find_unused_parameters')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # config options
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument('--num_epochs_to_eval', type=int, default=5, help='number of checkpoints to be evaluated')
    parser.add_argument('--trainval', action='store_true', default=False, help='')
    parser.add_argument('--imitation', type=str, default="2d")
    parser.add_argument('--eval', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    update_cfg_by_args(cfg, args)
    cfg.TAG = Path(args.cfg_file).stem
    # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '_'.join(args.cfg_file.split('/')[1:-1])
    cfg.DATA_CONFIG.INFER_TIME_PATH = str(Path('outputs') / 'inference_time' / cfg.EXP_GROUP_PATH / (cfg.TAG + '.json'))
    cfg.OPTIMIZATION.USE_AMP = cfg.MODEL.get('USE_AMP', {'TRAIN':False, 'TEST':False})
    
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        print(f'Using Port: {args.tcp_port}')
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'outputs' / cfg.EXP_GROUP_PATH / (cfg.TAG + '.' + args.exp_name)
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log_train.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(
    ) else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('git diff > %s/%s' % (output_dir, 'git.diff'))
        os.system('git log > %s/%s' % (output_dir, 'git.log'))
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(
        output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn or cfg.MODEL.get('SYNC_BN', False):
        if dist_train:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            logger.info('Closed Synchronized BN as dist_train == False.')
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=True, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    elif args.continue_train:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))
            print("using ckpt", ckpt_list[-1])
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=True, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1
            logger.info(f'Continue training from epoch {last_epoch}\n')
        else:
            pass
            # raise FileNotFoundError("no ckpt files found")

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=args.find_unused_parameters)
    logger.info(model)
    
    if not args.eval:
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(train_loader) // (args.epochs if args.merge_all_iters_to_one_epoch else 1), total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )

        # -----------------------start training---------------------------
        logger.info('*******Start training: {} ********'.format(output_dir))
        train_model(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            dist_train=dist_train,
            logger=logger
        )
        logger.info('*******End training: {} ********'.format(output_dir))

    model.eval()

    logger.info('*******Start evaluation: {} ********'.format(output_dir))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # Only evaluate the last 10 epochs
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('*******End evaluation: {} ********'.format(output_dir))


if __name__ == '__main__':
    main()
