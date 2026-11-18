from __future__ import annotations

import torch


class RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class EpochBatchSampler(torch.utils.data.Sampler):
    """Stable epoch permutation with a batch cursor that does not fetch skipped data."""
    def __init__(self, dataset, batch_size, *, seed, epoch, start_batch=0):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=torch.distributed.get_world_size() if distributed else 1,
            rank=torch.distributed.get_rank() if distributed else 0,
            shuffle=True, seed=seed, drop_last=False,
        )
        self.sampler.set_epoch(epoch)
        self.batch_size = int(batch_size)
        self.start_batch = int(start_batch)
        self.full_length = len(self.sampler) // self.batch_size

    def __iter__(self):
        indices = list(self.sampler)
        for batch_index in range(self.start_batch, self.full_length):
            start = batch_index * self.batch_size
            yield indices[start:start + self.batch_size]

    def __len__(self):
        return max(0, self.full_length - self.start_batch)


def seed_dataset_epoch(dataset, seed):
    """Control built-in per-index RNGs, including mixed-domain dataset children."""
    if hasattr(dataset, 'deterministic'):
        dataset.deterministic = True
    if hasattr(dataset, 'seed'):
        dataset.seed = int(seed)
    children = getattr(dataset, 'datasets', [])
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        seed_dataset_epoch(child, seed)


def build_epoch_loader(dataset, config, *, epoch, start_batch=0):
    seed = int(config['runtime'].get('seed', 200099))
    seed_dataset_epoch(dataset, seed + epoch * 1_000_003)
    train = config['train']
    sampler = EpochBatchSampler(dataset, int(train.get('batch_size', 1)), seed=seed, epoch=epoch, start_batch=start_batch)
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=int(train.get('workers', 4)),
        pin_memory=torch.cuda.is_available(), generator=generator,
        persistent_workers=False,
    )
