"""ShapeNetCore v2 point-cloud dataset (the 15k subsampled PointFlow split).

Strip-down of the original Uniform15KPC/ShapeNet15kPointClouds loader:
unused branches (per-shape / box normalization, masked variants, partnet, sv)
are removed. We expose a single class ``ShapeNet15kPointClouds`` configured
the same way the original ``train.py``/``train_ss.py`` did.

Directory layout expected at ``root_dir``::

    ShapeNetCore.v2.PC15k/
    └── <synset_id>/{train,val,test}/<model_id>.npy   # each (15000, 3)
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


# ShapeNetCore synset id -> readable category name.
SYNSET_TO_CATE = {
    '02691156': 'airplane', '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',  '02818832': 'bed',        '02828884': 'bench',
    '02876657': 'bottle',   '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',  '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',      '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',    '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',    '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',    '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',   '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',    '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',  '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave','03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',    '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',      '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',    '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',     '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',   '02992529': 'cellphone',  '02843684': 'birdhouse',
    '02871439': 'bookshelf',
}
CATE_TO_SYNSET = {v: k for k, v in SYNSET_TO_CATE.items()}

# The 15k point cloud splits use the first 10k points for train, the rest
# for test (PointFlow convention). We keep the same split here.
_N_TRAIN_PTS = 10000
_N_TEST_PTS_MAX = 5000


class ShapeNet15kPointClouds(Dataset):
    """ShapeNet point clouds (uniformly sampled 15k per shape).

    Args:
        root_dir: directory containing per-synset subfolders.
        categories: list of category names (or ['all']).
        split: one of 'train' | 'val' | 'test'.
        tr_sample_size/te_sample_size: number of points to subsample for the
            train/test halves of each shape's 15k points (default 2048).
        random_subsample: when True, the subsample is random per __getitem__.
        all_points_mean/std: optional pre-computed normalization stats; if
            None, dataset-wide stats are computed from the loaded shapes.
    """

    def __init__(self, root_dir, categories=('airplane',), split='train',
                 tr_sample_size=10000, te_sample_size=2048,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        assert split in ('train', 'val', 'test')
        self.root_dir = root_dir
        self.split = split
        self.cates = list(categories)
        self.synset_ids = (list(CATE_TO_SYNSET.values()) if 'all' in self.cates
                           else [CATE_TO_SYNSET[c] for c in self.cates])
        self.random_subsample = random_subsample
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        all_points, cate_idx_lst, all_cate_mids = [], [], []
        for cate_idx, subd in enumerate(self.synset_ids):
            sub_path = os.path.join(root_dir, subd, split)
            if not os.path.isdir(sub_path):
                print(f"[shapenet_pc] missing dir: {sub_path}")
                continue
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                obj_fname = os.path.join(sub_path, x)
                try:
                    pc = np.load(obj_fname)
                except OSError:
                    continue
                assert pc.shape[0] == 15000, f"{obj_fname} has {pc.shape[0]} pts"
                all_points.append(pc[np.newaxis, ...])
                cate_idx_lst.append(cate_idx)
                all_cate_mids.append((subd, os.path.join(split, x[:-len('.npy')])))

        # Deterministic shuffle of indices, seeded by data count.
        shuffle_idx = list(range(len(all_points)))
        random.Random(38383).shuffle(shuffle_idx)
        self.cate_idx_lst = [cate_idx_lst[i] for i in shuffle_idx]
        self.all_cate_mids = [all_cate_mids[i] for i in shuffle_idx]
        all_points = np.concatenate([all_points[i] for i in shuffle_idx])  # (N, 15000, 3)

        # Dataset-wide normalization (single mean/std for all shapes/axes).
        if all_points_mean is not None and all_points_std is not None:
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        else:
            self.all_points_mean = all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            self.all_points_std = all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        all_points = (all_points - self.all_points_mean) / self.all_points_std
        self.train_points = all_points[:, :_N_TRAIN_PTS]
        self.test_points = all_points[:, _N_TRAIN_PTS:]
        self.tr_sample_size = min(_N_TRAIN_PTS, tr_sample_size)
        self.te_sample_size = min(_N_TEST_PTS_MAX, te_sample_size)
        print(f"[shapenet_pc] split={split} cates={self.cates} "
              f"N={len(self.train_points)} "
              f"npts(train,test)=({self.tr_sample_size},{self.te_sample_size})")

    def __len__(self):
        return len(self.train_points)

    def _subsample(self, points, k):
        if self.random_subsample:
            idx = np.random.choice(points.shape[0], k)
        else:
            idx = np.arange(k)
        return torch.from_numpy(points[idx]).float()

    def __getitem__(self, idx):
        sid, mid = self.all_cate_mids[idx]
        return {
            'idx': idx,
            'train_points': self._subsample(self.train_points[idx], self.tr_sample_size),
            'test_points': self._subsample(self.test_points[idx], self.te_sample_size),
            'mean': self.all_points_mean.reshape(1, -1),
            'std': self.all_points_std.reshape(1, -1),
            'cate_idx': self.cate_idx_lst[idx],
            'sid': sid,
            'mid': mid,
        }
