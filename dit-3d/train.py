"""DiT-3D training driver — baseline (DiT-w) and ours
(spectral-representation-regularized DiT-w).

This is a consolidation of the original ``train.py`` + ``train_ss.py``: the
``--ss`` flag toggles the projector-based spectral-regularization branch.
Both paths share the same forward-pass / sampling / checkpointing scaffold.

Multi-GPU is via ``--distribution_type multi`` (mp.spawn + DDP). On a 4-GPU
node, ``--bs`` is the *total* batch (divided by ngpus internally; see
``--saveIter`` / ``--vizIter`` which are also divided).
"""
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[no-redef]
    except ModuleNotFoundError:
        class SummaryWriter:  # type: ignore[no-redef]
            """No-op fallback for environments without tensorboard support."""

            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

from timm_compat import ensure_timm_mlp

ensure_timm_mlp()

from diffusion import Model, get_betas
from dit3d import DiT3D_models
from shapenet_pc import ShapeNet15kPointClouds
from utils import (copy_source, get_output_dir, make_subdir, set_seed,
                   setup_logging, visualize_pointcloud_batch)


def _open_gpu_mem_log(opt, gpu):
    if not opt.gpu_mem_log_dir:
        return None
    os.makedirs(opt.gpu_mem_log_dir, exist_ok=True)
    path = os.path.join(opt.gpu_mem_log_dir, f"torch_gpu_mem_rank{opt.rank}_gpu{gpu}.csv")
    handle = open(path, "w", buffering=1)
    handle.write(
        "epoch,iter,rank,gpu,allocated_mib,reserved_mib,max_allocated_mib,"
        "max_reserved_mib,free_mib,total_mib\n"
    )
    return handle


def _write_gpu_mem_log(handle, epoch, iteration, rank, gpu):
    if handle is None:
        return
    mib = 1024 * 1024
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
        free_mib = int(free_bytes / mib)
        total_mib = int(total_bytes / mib)
    except Exception:
        free_mib = ""
        total_mib = ""
    handle.write(
        f"{epoch},{iteration},{rank},{gpu},"
        f"{int(torch.cuda.memory_allocated(gpu) / mib)},"
        f"{int(torch.cuda.memory_reserved(gpu) / mib)},"
        f"{int(torch.cuda.max_memory_allocated(gpu) / mib)},"
        f"{int(torch.cuda.max_memory_reserved(gpu) / mib)},"
        f"{free_mib},{total_mib}\n"
    )


# ---------------------------------------------------------------------------
#                              Data helpers
# ---------------------------------------------------------------------------

def build_dataset(opt):
    return ShapeNet15kPointClouds(
        root_dir=opt.dataroot,
        categories=opt.category.split(','),
        split='train',
        tr_sample_size=opt.npoints,
        te_sample_size=opt.npoints,
        random_subsample=True,
    )


def build_dataloader(opt, train_dataset):
    if opt.distribution_type == 'multi':
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.bs, sampler=sampler,
        shuffle=sampler is None, num_workers=int(opt.workers), drop_last=True)
    return loader, sampler


# ---------------------------------------------------------------------------
#                               Training loop
# ---------------------------------------------------------------------------

def train(gpu, opt, output_dir):
    set_seed(opt.manualSeed)
    logger = setup_logging(output_dir)

    is_multi = opt.distribution_type == 'multi'
    is_leader = (gpu == 0) if is_multi else True
    tb_writer = SummaryWriter(output_dir) if (is_leader and opt.use_tb) else None
    outf_syn = make_subdir(output_dir, 'syn') if is_leader else None

    # ---- distributed init + per-GPU rescaling of bs / save/viz intervals
    if is_multi:
        opt.rank = opt.rank * opt.ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0
        opt.saveIter = max(1, int(opt.saveIter / opt.ngpus_per_node))
        opt.vizIter = max(1, int(opt.vizIter / opt.ngpus_per_node))
        torch.cuda.set_device(gpu)
    else:
        raise NotImplementedError("Only distribution_type='multi' is supported.")

    # ---- data
    train_dataset = build_dataset(opt)
    dataloader, train_sampler = build_dataloader(opt, train_dataset)

    # ---- model
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(args=opt, betas=betas).cuda(gpu)

    def _ddp(m):
        if opt.proj_bn_sync:
            m = nn.SyncBatchNorm.convert_sync_batchnorm(m)
        return nn.parallel.DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)
    model.multi_gpu_wrapper(_ddp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0)
    if opt.gpu_mem_log_dir:
        torch.cuda.reset_peak_memory_stats(gpu)
    gpu_mem_log = _open_gpu_mem_log(opt, gpu)

    # ---- optional resume
    start_epoch = 0
    if opt.model:
        ckpt = torch.load(opt.model, map_location=f"cuda:{gpu}")
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1

    if is_leader:
        logger.info(opt)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Total parameters: {n_params:.2f} M")

    # ---- main loop
    for epoch in range(start_epoch, opt.niter):
        if is_multi:
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(dataloader):
            x = data['train_points'].transpose(1, 2).cuda(gpu)
            y = data['cate_idx'].cuda(gpu)

            losses = model.get_loss_iter(x, y=y)
            if opt.ss:
                sm_loss, proj_loss = losses
                loss = sm_loss.mean() + opt.proj_coeff * proj_loss.mean()
            else:
                sm_loss = losses
                proj_loss = None
                loss = sm_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            if tb_writer is not None:
                step = i + len(dataloader) * epoch
                tb_writer.add_scalar('train_loss', loss.item(), step)
                tb_writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], step)

            if is_leader and i % opt.print_freq == 0:
                if opt.ss:
                    logger.info(
                        f"[{epoch:>3d}/{opt.niter:>3d}][{i:>3d}/{len(dataloader):>3d}] "
                        f"   loss: {loss.item():>10.4f}"
                        f"     sm_loss: {sm_loss.detach().mean().item():>10.4f}"
                        f"    proj_loss: {proj_loss.detach().mean().item():>10.4f}")
                else:
                    logger.info(
                        f"[{epoch:>3d}/{opt.niter:>3d}][{i:>3d}/{len(dataloader):>3d}] "
                        f"   loss: {loss.item():>10.4f}")

            if i % opt.gpu_mem_log_freq == 0:
                _write_gpu_mem_log(gpu_mem_log, epoch, i, opt.rank, gpu)

        # ---- periodic sample + save
        if (epoch + 1) % opt.vizIter == 0:
            if is_leader:
                _gen_and_visualize(model, opt, x, y, epoch, outf_syn, logger)
            if is_multi:
                dist.barrier()

        if (epoch + 1) % opt.saveIter == 0:
            if is_leader:
                ck_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, ck_path)
                logger.info(f"Checkpoint saved at {ck_path}")
            if is_multi:
                dist.barrier()
                map_location = {'cuda:0': f'cuda:{gpu}'}
                state = torch.load(os.path.join(output_dir, f"epoch_{epoch}.pth"),
                                   map_location=map_location)
                model.load_state_dict(state['model_state'])

    if tb_writer is not None:
        tb_writer.close()
    if gpu_mem_log is not None:
        gpu_mem_log.close()
    dist.destroy_process_group()


def _gen_and_visualize(model, opt, x, y, epoch, outf_syn, logger):
    """Sample 25 unconditional shapes + one full reverse trajectory."""
    logger.info('Generation: eval')

    # IMPORTANT: unwrap DDP to keep evaluation deterministic across ranks.
    cached = model.model
    model.model = model.model.module
    model.model.eval()

    with torch.no_grad():
        def randn_like_batch(n):
            return torch.randn(n, *x.shape[1:], device=x.device)

        def rand_y(n):
            return torch.randint(0, opt.num_classes, (n,), device=y.device)

        x_eval = model.gen_samples(randn_like_batch(25).shape, x.device, rand_y(25), clip_denoised=False)
        x_traj = model.gen_sample_traj(randn_like_batch(1).shape, x.device, rand_y(1),
                                       freq=40, clip_denoised=False)
        x_traj = torch.cat(x_traj, dim=0)

    logger.info(f"      [{epoch:>3d}/{opt.niter:>3d}]  "
                f"eval_gen_range: [{x_eval.min().item():>10.4f}, {x_eval.max().item():>10.4f}]     "
                f"eval_gen_stats: [mean={x_eval.mean().item():>10.4f}, std={x_eval.std().item():>10.4f}]")

    visualize_pointcloud_batch(os.path.join(outf_syn, f"epoch_{epoch:03d}_samples_eval.png"),
                               x_eval.transpose(1, 2))
    visualize_pointcloud_batch(os.path.join(outf_syn, f"epoch_{epoch:03d}_samples_eval_all.png"),
                               x_traj.transpose(1, 2))
    visualize_pointcloud_batch(os.path.join(outf_syn, f"epoch_{epoch:03d}_x.png"),
                               x.transpose(1, 2))

    logger.info('Generation: train')
    model.model = cached
    model.model.train()


# ---------------------------------------------------------------------------
#                                  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument('--model_dir', type=str, default='./output')
    p.add_argument('--experiment_name', type=str, default='dit3d')

    # data
    p.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    p.add_argument('--category', default='chair')
    p.add_argument('--num_classes', type=int, default=55)
    p.add_argument('--bs', type=int, default=16,
                   help='total batch size (divided by ngpus_per_node internally)')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--niter', type=int, default=10000)
    p.add_argument('--nc', default=3)
    p.add_argument('--npoints', default=2048)
    p.add_argument('--voxel_size', type=int, choices=[16, 32, 64, 128, 256], default=32)

    # model
    p.add_argument('--model_type', type=str, choices=list(DiT3D_models.keys()), default='DiT-S/4')
    p.add_argument('--window_size', type=int, default=0)
    p.add_argument('--window_block_indexes', type=lambda s: tuple(int(x) for x in s.split(',')),
                   default=(0, 3, 6, 9))
    p.add_argument('--loss_type', default='mse')
    p.add_argument('--model_mean_type', default='eps')
    p.add_argument('--model_var_type', default='fixedsmall')
    p.add_argument('--beta_start', type=float, default=0.0001)
    p.add_argument('--beta_end', type=float, default=0.02)
    p.add_argument('--schedule_type', default='linear')
    p.add_argument('--time_num', type=int, default=1000)

    # Spectral representation regularization (ours)
    p.add_argument('--ss', action=argparse.BooleanOptionalAction, default=False,
                   help='Train the spectral-representation-regularized DiT-w (ours).')
    p.add_argument('--encoder-depth', type=int, default=8)
    p.add_argument('--proj-dims', nargs='*', type=int, default=[2048, 768])
    p.add_argument('--proj-adaln', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--proj-bn', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--proj-bn-sync', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--proj-coeff', type=float, default=0.5)
    p.add_argument('--proj-triu-coeff', type=float, default=0.025)
    p.add_argument('--proj-global-batch', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--proj-single-t-batch', action=argparse.BooleanOptionalAction, default=False)

    # optimization
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=None)
    p.add_argument('--model', default='', help='checkpoint path to resume from')

    # distributed
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--node', type=str, default='localhost')
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    p.add_argument('--dist_backend', default='nccl')
    p.add_argument('--distribution_type', default='multi', choices=['multi'])
    p.add_argument('--rank', default=0, type=int)
    p.add_argument('--gpu', default=None, type=int)
    p.add_argument('--ngpus_per_node', type=int, default=None)

    # checkpointing / logging cadence (effective interval = value / ngpus)
    p.add_argument('--saveIter', default=100, type=int, help='unit: epoch')
    p.add_argument('--vizIter', default=100, type=int, help='unit: epoch')
    p.add_argument('--print_freq', default=50, type=int, help='unit: iter')

    p.add_argument('--manualSeed', default=42, type=int)
    p.add_argument('--use_tb', action='store_true', default=False)
    p.add_argument('--gpu_mem_log_dir', default='',
                   help='Optional directory for per-rank torch CUDA memory CSV logs.')
    p.add_argument('--gpu_mem_log_freq', default=1, type=int,
                   help='Log torch CUDA memory every N training iterations when enabled.')
    return p.parse_args()


def main():
    opt = parse_args()
    opt.gpu_mem_log_freq = max(1, int(opt.gpu_mem_log_freq))

    # Original DiT-3D used a different noise schedule for airplane (warm0.1
    # with smaller endpoints). Preserve that behavior here.
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    output_dir = get_output_dir(opt.model_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    # Avoid port collisions when launching parallel jobs.
    if opt.world_size == 1:
        opt.port = np.random.randint(10000, 20000)
    opt.dist_url = f'tcp://{opt.node}:{opt.port}'

    opt.ngpus_per_node = torch.cuda.device_count()
    opt.world_size = opt.ngpus_per_node * opt.world_size
    mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir))


if __name__ == '__main__':
    main()
