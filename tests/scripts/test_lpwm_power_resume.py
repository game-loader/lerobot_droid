import pytest
import torch
from torch.utils.data import DataLoader

from scripts.lpwm_monitor.resume_train import RestoredBatchSampler


def test_reconstructed_sampler_matches_original_persistent_loader():
    size, batch = 1001, 8
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        torch.arange(size),
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=2,
        persistent_workers=True,
    )
    iterator = iter(loader)
    # Cross one complete epoch, then stop away from the prefetch boundary.
    start = size // batch + 16
    for _ in range(start):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    resumed = RestoredBatchSampler(size, batch, start, start + 5, generator.get_state())
    assert list(resumed) == [next(iterator).tolist() for _ in range(5)]
    del iterator, loader


def test_reject_inexact_sampler_state():
    with pytest.raises(ValueError, match="exactly"):
        RestoredBatchSampler(1001, 8, 16, 20, torch.Generator().manual_seed(999).get_state())
