#!/usr/bin/env python
"""Benchmark pi0 inference latency with LIBERO-like inputs (synthetic).

Measures end-to-end time from visual observation to action output.
Bypasses the gated PaliGemma tokenizer by constructing token tensors directly.
"""

import time

import torch

from lerobot.policies.pi0 import PI0Policy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "lerobot/pi0_base"

    print(f"Device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}" if device == "cuda" else "")
    print(f"Loading model: {model_id}")
    t0 = time.perf_counter()
    model = PI0Policy.from_pretrained(model_id)
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s")
    print(f"  PaliGemma variant: {model.config.paligemma_variant}")
    print(f"  Action expert: {model.config.action_expert_variant}")
    print(f"  Flow matching steps: {model.config.num_inference_steps}")
    print(f"  Chunk size: {model.config.chunk_size}")
    print(f"  n_action_steps: {model.config.n_action_steps}")
    print(f"  Image resolution: {model.config.image_resolution}")
    print(f"  Max state dim: {model.config.max_state_dim}")
    print(f"  Max action dim: {model.config.max_action_dim}")
    print(f"  Tokenizer max length: {model.config.tokenizer_max_length}")

    img_keys = [k for k in model.config.input_features if "image" in k]
    print(f"  Image features: {img_keys}")

    def make_batch():
        """Construct a batch matching what select_action expects after preprocessing."""
        batch = {}
        # Images: [B, C, H, W] normalized to [0, 1] (model will convert to [-1, 1])
        for key in img_keys:
            batch[key] = torch.rand(1, 3, 224, 224, device=device)
        # State: [B, state_dim] — LIBERO has 7-DOF
        batch["observation.state"] = torch.rand(1, 7, device=device)
        # Language tokens: [B, max_length] — simulate tokenized task description
        max_len = model.config.tokenizer_max_length
        batch["observation.language.tokens"] = torch.randint(0, 250000, (1, max_len), device=device)
        batch["observation.language.attention_mask"] = torch.ones(1, max_len, dtype=torch.bool, device=device)
        return batch

    # Warmup
    print("\nWarmup...")
    batch = make_batch()
    with torch.no_grad():
        _ = model.select_action(batch)
    model.reset()
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs — measure full chunk prediction (the expensive call)
    n_runs = 20
    print(f"\nTiming {n_runs} full action chunk predictions...")
    print(
        "(Each predicts a full chunk of {model.config.chunk_size} actions via {model.config.num_inference_steps}-step flow matching)\n"
    )

    times = []
    for _i in range(n_runs):
        model.reset()
        batch = make_batch()

        if device == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        with torch.no_grad():
            model.select_action(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000
        times.append(elapsed_ms)

    # Also time what happens on subsequent calls (queue pop, near-zero)
    queue_times = []
    batch = make_batch()
    model.reset()
    with torch.no_grad():
        _ = model.select_action(batch)  # fills queue
    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(min(model.config.n_action_steps - 1, 10)):
        if device == "cuda":
            torch.cuda.synchronize()
        t_s = time.perf_counter()
        with torch.no_grad():
            _ = model.select_action(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        queue_times.append((time.perf_counter() - t_s) * 1000)

    # Results
    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    med = sorted(times)[len(times) // 2]

    print("=" * 70)
    print("RESULTS: pi0 inference latency (vision → action)")
    print("=" * 70)
    print(f"  Full chunk prediction ({model.config.num_inference_steps}-step flow matching):")
    print(f"    Average:  {avg:8.2f} ms")
    print(f"    Median:   {med:8.2f} ms")
    print(f"    Min:      {mn:8.2f} ms")
    print(f"    Max:      {mx:8.2f} ms")
    print()
    print("  Queue pop (subsequent actions from same chunk):")
    if queue_times:
        print(f"    Average:  {sum(queue_times) / len(queue_times):8.4f} ms")
    print()
    print(f"  Action chunk: {model.config.n_action_steps} steps")
    print(f"  Amortized per-step: {avg / model.config.n_action_steps:.2f} ms")
    print(f"  Effective control freq (amortized): {model.config.n_action_steps / (avg / 1000):.1f} Hz")
    print(f"  Real-time chunk freq: {1000 / avg:.1f} Hz")
    print("=" * 70)

    # Breakdown: vision encoding vs flow matching denoising
    print("\n--- Breakdown: prefix encoding vs denoising loop ---")
    model.reset()
    batch = make_batch()
    images, img_masks = model._preprocess_images(batch)
    state = model.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    if device == "cuda":
        torch.cuda.synchronize()

    # Time prefix encoding (vision + language)
    prefix_times = []
    for _ in range(10):
        if device == "cuda":
            torch.cuda.synchronize()
        t_s = time.perf_counter()
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        if device == "cuda":
            torch.cuda.synchronize()
        prefix_times.append((time.perf_counter() - t_s) * 1000)

    # Time full sample_actions (includes prefix + denoising)
    sample_times = []
    for _ in range(10):
        if device == "cuda":
            torch.cuda.synchronize()
        t_s = time.perf_counter()
        with torch.no_grad():
            _ = model.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
        if device == "cuda":
            torch.cuda.synchronize()
        sample_times.append((time.perf_counter() - t_s) * 1000)

    avg_prefix = sum(prefix_times) / len(prefix_times)
    avg_sample = sum(sample_times) / len(sample_times)
    avg_denoise = avg_sample - avg_prefix

    print(f"  Prefix encoding (vision+lang): {avg_prefix:8.2f} ms")
    print(f"  Denoising loop ({model.config.num_inference_steps} steps):  {avg_denoise:8.2f} ms")
    print(f"  Total sample_actions:          {avg_sample:8.2f} ms")
    print(f"  Per denoising step:            {avg_denoise / model.config.num_inference_steps:8.2f} ms")


if __name__ == "__main__":
    main()
