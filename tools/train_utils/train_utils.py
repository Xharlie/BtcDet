import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, pc_dir=None,cur_epoch=None):
    tb_log=None

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

        loss, tb_dict, disp_dict, pc_dict = model_func(model, batch)

        loss.backward()
        # print("self.conv_res.weight.grad: ", model.occ_occ_dense_head.conv_cls.bias.grad) ## check gradient pass

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

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
                    prefix = 'train/'
                    if key.startswith("occ"):
                        prefix = "occ/"
                    if key.endswith('img'):
                        prefix = key.split("_")[0]+"/"
                        tb_log.add_image(prefix + key, val, accumulated_iter, dataformats="HWC")
                    else:
                        tb_log.add_scalar(prefix + key, val, accumulated_iter)
            if bool(pc_dict):
                for key, val in pc_dict.items():
                    if torch.is_tensor(val) and val.is_cuda:
                        pc_dict[key] = val.cpu().numpy()
                np.save(str(pc_dir)+'/pc_{}_{}'.format(cur_epoch, accumulated_iter), pc_dict)


    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_one_epoch_multi_opt(model, optimizer_lst, train_loader, model_func, lr_scheduler_lst, accumulated_iter, optim_cfg_lst, rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, pc_dir=None,cur_epoch=None):

    tb_log=None

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        parameter_lst = [model.module.occ_modules, model.module.det_modules]
    else:
        parameter_lst = [model.occ_modules, model.det_modules]

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        model.train()
        
        cur_lr_lst = []
        for i in range(len(optimizer_lst)):
            optimizer = optimizer_lst[i]
            optimizer.zero_grad()
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']

            if tb_log is not None:
                tb_log.add_scalar('meta_data/learning_rate_{}'.format(i), cur_lr, accumulated_iter)
            cur_lr_lst.append(cur_lr)

        loss, tb_dict, disp_dict, pc_dict = model_func(model, batch)

        loss.backward()

        for i in range(len(optimizer_lst)):
            clip_grad_norm_(parameter_lst[i].parameters(), optim_cfg_lst[i].GRAD_NORM_CLIP)
            optimizer_lst[i].step()
            lr_scheduler_lst[i].step(accumulated_iter)
            optimizer_lst[i].lr = max(optimizer_lst[i].lr, optim_cfg_lst[i].LR_CLIP)

        accumulated_iter += 1
        if len(cur_lr_lst) > 1:
            disp_dict.update({'loss': loss.item(), 'lr_occ': cur_lr_lst[0], 'lr_det': cur_lr_lst[1]})
        else:
            disp_dict.update({'loss': loss.item(), 'lr': cur_lr_lst[0]})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                # tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    prefix = 'train/'
                    if key.startswith("occ"):
                        prefix = "occ/"
                    if key.endswith('img'):
                        prefix = key.split("_")[0]+"/"
                        tb_log.add_image(prefix + key, val, accumulated_iter, dataformats="HWC")
                    else:
                        tb_log.add_scalar(prefix + key, val, accumulated_iter)
            if bool(pc_dict):
                np.save(str(pc_dir)+'/pc_{}_{}'.format(cur_epoch,accumulated_iter), pc_dict)


    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, pc_dir=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

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
                pc_dir=pc_dir,
                cur_epoch=cur_epoch,
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

def train_model_multi_opt(model, optimizer_lst, train_loader, model_func, lr_scheduler_lst, optim_cfg_lst,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler_lst=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, pc_dir=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            cur_scheduler_lst = []
            for i in range(len(optim_cfg_lst)):
                if lr_warmup_scheduler_lst[i] is not None and cur_epoch < optim_cfg_lst[i].WARMUP_EPOCH:
                    cur_scheduler_lst.append(lr_warmup_scheduler_lst[i])
                else:
                    cur_scheduler_lst.append(lr_scheduler_lst[i])
            accumulated_iter = train_one_epoch_multi_opt(
                model, optimizer_lst, train_loader, model_func,
                lr_scheduler_lst=cur_scheduler_lst,
                accumulated_iter=accumulated_iter, optim_cfg_lst=optim_cfg_lst,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                pc_dir=pc_dir,
                cur_epoch=cur_epoch,
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
                    checkpoint_state_mult_opt(model, optimizer_lst, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state_mult_opt(model=None, optimizer_lst=None, epoch=None, it=None):
    optim_state_lst = [optimizer.state_dict() if optimizer is not None else None for optimizer in optimizer_lst]
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import btcdet
        version = 'btcdet+' + btcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state_lst': optim_state_lst, 'version': version}

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
        import btcdet
        version = 'btcdet+' + btcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optim_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
