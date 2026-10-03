"""Prepare a row-preserving, resized LIBERO cache and real frozen text embeddings."""

import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CAMERAS = ("observation.images.image", "observation.images.image2")


def decode_image(encoded: dict, image_size: int) -> np.ndarray:
    if encoded.get("bytes") is None:
        raise ValueError("Expected embedded image bytes; refusing an ambiguous external image path.")
    with Image.open(io.BytesIO(encoded["bytes"])) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def prepare(root: Path, output: Path, image_size: int, workers: int) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Cache already complete: {output}")
    output.mkdir(parents=True, exist_ok=True)
    n = info["total_frames"]
    images = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.uint8, shape=(n, len(CAMERAS), 3, image_size, image_size)
    )
    arrays = {
        "states": np.empty((n, 8), dtype=np.float32),
        "actions": np.empty((n, 7), dtype=np.float32),
        "episode_index": np.empty(n, dtype=np.int64),
        "frame_index": np.empty(n, dtype=np.int64),
        "task_index": np.empty(n, dtype=np.int64),
        "timestamps": np.empty(n, dtype=np.float32),
    }
    seen = np.zeros(n, dtype=bool)
    hashes = {}
    source_files = sorted((root / "data").rglob("*.parquet"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for file_no, source in enumerate(source_files):
            table = pq.read_table(source)
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            if (
                (indices < 0).any()
                or (indices >= n).any()
                or seen[indices].any()
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(f"Duplicate/out-of-bounds frame indices in {source}")
            hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
            for name, column in (
                ("states", "observation.state"),
                ("actions", "action"),
                ("episode_index", "episode_index"),
                ("frame_index", "frame_index"),
                ("task_index", "task_index"),
                ("timestamps", "timestamp"),
            ):
                arrays[name][indices] = np.asarray(table[column].to_pylist(), dtype=arrays[name].dtype)
            for view, camera in enumerate(CAMERAS):
                decoded = executor.map(lambda item: decode_image(item, image_size), table[camera].to_pylist())
                for index, image in zip(indices, decoded, strict=True):
                    images[index, view] = image
            seen[indices] = True
            print(f"Prepared {file_no + 1}/{len(source_files)} files, {seen.sum()}/{n} rows", flush=True)
    if not seen.all() or not np.isfinite(arrays["states"]).all() or not np.isfinite(arrays["actions"]).all():
        raise ValueError("Incomplete or nonfinite cache")
    episodes = []
    for episode in np.unique(arrays["episode_index"]):
        indices = np.flatnonzero(arrays["episode_index"] == episode)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"Episode {episode} is not contiguous")
        if not np.array_equal(arrays["frame_index"][indices], np.arange(len(indices))):
            raise ValueError(f"Episode {episode} frame indices are not contiguous")
        tasks = np.unique(arrays["task_index"][indices])
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode} has multiple task ids")
        episodes.append(
            {
                "episode_index": int(episode),
                "start": int(indices[0]),
                "end": int(indices[-1] + 1),
                "task_index": int(tasks[0]),
            }
        )
    images.flush()
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)
    task_table = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    tasks = {
        str(index): text for index, text in zip(task_table["task_index"], task_table["task"], strict=True)
    }
    manifest = {
        "schema": "lpwm_libero_cache_v1",
        "source": str(root),
        "source_info": info,
        "source_file_sha256": hashes,
        "cameras": list(CAMERAS),
        "image_size": image_size,
        "image_preprocessing": "PIL RGB bilinear resize; no frame dropping or image flip",
        "temporal_semantics": "Original row sequence retained; control-rate provenance audited separately",
        "action_normalization": "identity; native LIBERO controller action values retained",
        "num_frames": n,
        "episodes": episodes,
        "tasks": tasks,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Cache complete: {output}", flush=True)


def encode_language(output: Path, model_path: Path, max_length: int) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    manifest = json.loads((output / "manifest.json").read_text())
    tasks = sorted(((int(k), v) for k, v in manifest["tasks"].items()))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inputs = tokenizer(
        [text for _, text in tasks],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32
    )
    vlm.eval().requires_grad_(False)
    core = getattr(vlm, "model", vlm)
    text_model = core.text_model
    with torch.no_grad():
        embeddings = (
            text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            .last_hidden_state.float()
            .cpu()
        )
    if not embeddings.isfinite().all():
        raise ValueError("Nonfinite frozen text embeddings")
    np.save(output / "language_embeddings.npy", embeddings.numpy())
    np.save(output / "language_masks.npy", inputs["attention_mask"].numpy().astype(bool))
    manifest["language"] = {
        "encoder": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "snapshot": model_path.name,
        "method": "frozen pretrained text_model final hidden states (no visual encoder, no task-ID embedding)",
        "hidden_dim": embeddings.shape[-1],
        "max_length": max_length,
        "task_ids": [index for index, _ in tasks],
        "weights_sha256": hashlib.sha256((model_path / "model.safetensors").read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Frozen language embeddings: {tuple(embeddings.shape)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--language-model", type=Path)
    parser.add_argument("--language-only", action="store_true")
    args = parser.parse_args()
    if not args.language_only:
        if args.root is None:
            parser.error("--root is required unless --language-only")
        prepare(args.root, args.output, args.image_size, args.workers)
    if args.language_model:
        encode_language(args.output, args.language_model, 48)


if __name__ == "__main__":
    main()
