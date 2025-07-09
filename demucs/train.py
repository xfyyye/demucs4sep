#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
主训练脚本入口点

主要功能:
- 解析命令行参数和配置
- 初始化分布式训练
- 构建模型、优化器和数据加载器
- 执行训练和验证
"""

import logging
import os
from pathlib import Path
import sys

from dora import hydra_main
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch
from torch import nn
import torchaudio
from torch.utils.data import ConcatDataset

from . import distrib
from .wav import get_wav_datasets, get_musdb_wav_datasets
from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .repitch import RepitchedWrapper
from .solver import Solver
from .states import capture_init
from .utils import random_subset

logger = logging.getLogger(__name__)


class TorchHDemucsWrapper(nn.Module):
    """
    torchaudio HDemucs实现的包装器,提供模型评估所需的元数据
    参考: https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html
    """

    @capture_init
    def __init__(self,  **kwargs):
        super().__init__()
        try:
            from torchaudio.models import HDemucs as TorchHDemucs
        except ImportError:
            raise ImportError("Please upgrade torchaudio for using its implementation of HDemucs")
        self.samplerate = kwargs.pop('samplerate')
        self.segment = kwargs.pop('segment')
        self.sources = kwargs['sources']
        self.torch_hdemucs = TorchHDemucs(**kwargs)

    def forward(self, mix):
        return self.torch_hdemucs.forward(mix)


def get_model(args):
    """
    根据配置参数构建模型
    """
    extra = {
        'sources': list(args.dset.sources),
        'audio_channels': args.dset.channels,
        'samplerate': args.dset.samplerate,
        'segment': args.model_segment or 4 * args.dset.segment,
    }
    klass = {
        'demucs': Demucs,
        'hdemucs': HDemucs,
        'htdemucs': HTDemucs,
        'torch_hdemucs': TorchHDemucsWrapper,
    }[args.model]
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    model = klass(**extra, **kw)
    return model


def get_optimizer(model, args):
    """
    构建优化器,支持不同参数组使用不同的优化设置
    """
    seen_params = set()
    other_params = []
    groups = []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            params = set(group["params"])
            assert params.isdisjoint(seen_params)
            seen_params |= set(params)
            groups.append(group)
    for param in model.parameters():
        if param not in seen_params:
            other_params.append(param)
    groups.insert(0, {"params": other_params})
    parameters = groups
    if args.optim.optim == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.optim == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer %s", args.optim.optimizer)


def get_datasets(args):
    """
    获取训练和验证数据集。
    支持 MUSDB 和自定义 WAV 数据集。
    """
    # 如果指定了 torchaudio 后端，则设置音频后端
    if args.dset.backend:
        torchaudio.set_audio_backend(args.dset.backend)
    
    # 如果使用 MUSDB 数据集，则获取 MUSDB 训练集和验证集
    if args.dset.use_musdb:
        train_set, valid_set = get_musdb_wav_datasets(args.dset)
    else:
        # 否则初始化为空列表
        train_set, valid_set = [], []
    
    # 如果指定了自定义 wav 数据集
    if args.dset.wav:
        # 获取自定义 wav 训练集和验证集
        extra_train_set, extra_valid_set = get_wav_datasets(args.dset)
        # 如果源数量小于等于4，则合并 MUSDB 和自定义 wav 数据集
        if len(args.dset.sources) <= 4:
            train_set = ConcatDataset([train_set, extra_train_set])
            valid_set = ConcatDataset([valid_set, extra_valid_set])
        else:
            # 否则只用自定义 wav 数据集
            train_set = extra_train_set
            valid_set = extra_valid_set

    # 如果指定了第二个自定义 wav2 数据集
    if args.dset.wav2:
        # 获取 wav2 训练集和验证集
        extra_train_set, extra_valid_set = get_wav_datasets(args.dset, "wav2")
        weight = args.dset.wav2_weight
        # 如果设置了权重，则根据权重调整主数据集和 wav2 的采样比例
        if weight is not None:
            b = len(train_set)
            e = len(extra_train_set)
            reps = max(1, round(e / b * (1 / weight - 1)))
        else:
            reps = 1
        # 合并主数据集和 wav2 数据集
        train_set = ConcatDataset([train_set] * reps + [extra_train_set])
        # 如果指定了 wav2_valid，则对验证集也做类似处理
        if args.dset.wav2_valid:
            if weight is not None:
                b = len(valid_set)
                n_kept = int(round(weight * b / (1 - weight)))
                valid_set = ConcatDataset(
                    [valid_set, random_subset(extra_valid_set, n_kept)]
                )
            else:
                valid_set = ConcatDataset([valid_set, extra_valid_set])
    # 如果指定了验证集采样数量，则对验证集进行随机采样
    if args.dset.valid_samples is not None:
        valid_set = random_subset(valid_set, args.dset.valid_samples)
    # 保证训练集和验证集不为空
    assert len(train_set)
    assert len(valid_set)
    return train_set, valid_set


def get_solver(args, model_only=False):
    """
    构建训练器(Solver)
    
    参数:
        args: 配置参数对象
        model_only: 是否只返回模型,不构建数据加载器
        
    返回:
        Solver对象,包含模型、优化器和数据加载器
        
    主要步骤:
    1. 初始化分布式训练
    2. 构建模型
    3. 初始化CUDA
    4. 构建优化器
    5. 准备数据集和数据加载器
    6. 构建并返回Solver
    """
    # 初始化分布式训练环境
    distrib.init()

    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 构建模型
    model = get_model(args)
    
    # 如果需要显示模型信息则打印并退出
    if args.misc.show:
        logger.info(model)
        # 计算模型参数大小(MB)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        # 如果模型有valid_length属性则计算感受野大小
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.dset.samplerate * 1000)
        sys.exit(0)

    # 如果可用则初始化CUDA
    if torch.cuda.is_available():
        model.cuda()

    # 构建优化器
    optimizer = get_optimizer(model, args)

    # 根据分布式训练的进程数调整batch size
    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    # 如果只需要模型则直接返回
    if model_only:
        return Solver(None, model, optimizer, args)

    # 获取训练和验证数据集
    train_set, valid_set = get_datasets(args)

    # 处理音高增强
    if args.augment.repitch.proba:
        vocals = []
        if 'vocals' in args.dset.sources:
            vocals.append(args.dset.sources.index('vocals'))
        else:
            logger.warning('No vocal source found')
        if args.augment.repitch.proba:
            train_set = RepitchedWrapper(train_set, vocals=vocals, **args.augment.repitch)

    # 打印数据集大小信息
    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    
    # 构建训练数据加载器
    train_loader = distrib.loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.misc.num_workers, drop_last=True)
        
    # 构建验证数据加载器
    if args.dset.full_cv:
        # 完整交叉验证模式
        valid_loader = distrib.loader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=args.misc.num_workers)
    else:
        # 普通验证模式
        valid_loader = distrib.loader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.misc.num_workers, drop_last=True)
            
    # 组合数据加载器
    loaders = {"train": train_loader, "valid": valid_loader}

    # 构建并返回Solver对象
    return Solver(loaders, model, optimizer, args)


def get_solver_from_sig(sig, model_only=False):
    """
    从实验签名构建训练器
    """
    inst = GlobalHydra.instance()
    hyd = None
    if inst.is_initialized():
        hyd = inst.hydra
        inst.clear()
    xp = main.get_xp_from_sig(sig)
    if hyd is not None:
        inst.clear()
        inst.initialize(hyd)

    with xp.enter(stack=True):
        return get_solver(xp.cfg, model_only)


@hydra_main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    """
    主函数:解析参数、初始化环境、开始训练
    
    参数:
        args: 配置参数对象,包含训练所需的各项配置
    
    主要步骤:
    1. 设置文件路径
    2. 处理数据集路径
    3. 设置环境变量
    4. 配置日志级别
    5. 初始化训练器
    6. 开始训练
    """
    # 设置当前文件的绝对路径
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    
    # 处理数据集相关的路径,转换为绝对路径
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val is not None:
            setattr(args.dset, attr, hydra.utils.to_absolute_path(val))

    # 设置OpenMP和MKL的线程数为1
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 如果开启详细日志,设置日志级别为DEBUG
    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)

    # 输出日志、检查点和样本的保存路径
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    
    # 导入并记录实验配置
    from dora import get_xp
    logger.debug(get_xp().cfg)

    # 初始化训练器并开始训练
    solver = get_solver(args)
    solver.train()


if '_DORA_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_DORA_TEST_PATH'])


if __name__ == "__main__":
    main()