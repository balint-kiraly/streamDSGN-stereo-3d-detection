import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.cuda.amp as amp

NaN_times = 0

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    dist_train=False, grad_scaler=None, logger=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        def print_grad_status(model):
            """Call this function after losses.backward()
            and it will find out all variables without grad, which
            means that the varaible is not in the graph.
            """
            for name, p in model.named_parameters():
                print('{:60s}{:15s}{:15s}{}'.format(name,
                    '(Trainable)' if p.requires_grad else '(Fixed)',
                    '(Has grad):' if p.grad is not None else '(No grad backward):',
                    list(p.shape)))

        # print(batch['token']['this_sample_idx'], batch['prev2']['this_sample_idx'], batch['prev']['this_sample_idx'], batch['next']['this_sample_idx'])
        # for k, v in batch.items():
        #     if v is not None:
        #         print(k, ': ', v['this_sample_idx'])
        #     else:
        #         print(k, ': ', v)
        # print('####################################')

        loss, tb_dict, disp_dict = model_func(model, batch)
        
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            if getattr('optim_cfg', 'GRAD_NORM_CLIP', None) is not None:
                grad_scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            # import numpy as np
            # grad_dict = {}
            # for name, param in model.named_parameters():
            #     grad_dict[name] = param.grad.cpu().numpy()
            # np.save('epoch_20.npy', grad_dict)
            # exit()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if getattr('optim_cfg', 'GRAD_NORM_CLIP', None) is not None:
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()

        if cur_it % 100 == 0:
            logger.info(' '.join([f'{k}: {v:.2f}' for k, v in tb_dict.items()]))
        elif cur_it % 10 == 0:
            print(' '.join([f'{k}: {v:.2f}' for k, v in tb_dict.items()]))

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        tb_dict.update({'loss': loss.item()})

        # # might get stuck
        # if dist_train: 
        #     # reduce and compute the mean value of losses from all gpus
        #     with torch.no_grad():
        #         loss_names = []
        #         all_losses = []
        #         for k, v in tb_dict.items():
        #             if isinstance(v, float):
        #                 loss_names.append(k)
        #                 all_losses.append(v)
        #         all_losses = torch.as_tensor(all_losses, dtype=torch.float32).cuda()
        #         dist.all_reduce(all_losses, op=dist.ReduceOp.SUM)
        #         all_losses /= dist.get_world_size()
        #         tb_dict.update({k: v.item() for k, v in zip(loss_names, all_losses)})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                    # NOTE: this is only for comparision with previous tensorboard eventfiles
                    if key == 'rpn_loss_cls':
                        tb_log.add_scalar('rpn3d_cls_loss', val, accumulated_iter)
                    if key == 'rpn_loss_loc':
                        tb_log.add_scalar('rpn3d_reg_loss', val, accumulated_iter)
                    if key == 'loss_depth_0_ce':
                        tb_log.add_scalar('disp_loss', val, accumulated_iter)

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, dist_train=False, logger=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        grad_scaler = amp.GradScaler() if optim_cfg.USE_AMP['TRAIN'] else None

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            train_loader.dataset.set_epoch(cur_epoch)
            
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                dist_train=dist_train,
                grad_scaler=grad_scaler,
                logger=logger
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
