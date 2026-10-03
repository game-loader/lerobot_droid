"""Frozen native SmolVLM language tokens, exact-text duplicates computed once."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main():
    import hashlib

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    def atomic(path, value):
        path.write_text(json.dumps(value, indent=2))

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    catalog = json.loads(a.catalog.read_text())
    tasks = catalog["tasks"]
    texts = list(dict.fromkeys(t["language"] for t in tasks))
    lookup = {t: i for i, t in enumerate(texts)}
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    vlm = (
        AutoModelForImageTextToText.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.float32)
        .eval()
        .requires_grad_(False)
    )
    core = getattr(vlm, "model", vlm)
    emb = []
    masks = []
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            inputs = tokenizer(
                texts[start : start + 8],
                padding="max_length",
                max_length=48,
                truncation=True,
                return_tensors="pt",
            )
            emb.append(
                core.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                .last_hidden_state.float()
                .cpu()
                .numpy()
            )
            masks.append(inputs["attention_mask"].numpy().astype(bool))
    embeddings = np.concatenate(emb)
    mask = np.concatenate(masks)
    indices = [lookup[t["language"]] for t in tasks]
    a.output.mkdir(parents=True, exist_ok=True)
    np.save(a.output / "language_embeddings.npy", embeddings[indices])
    np.save(a.output / "language_masks.npy", mask[indices])
    atomic(
        a.output / "language_manifest.json",
        {
            "encoder": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "snapshot": a.model.name,
            "method": "frozen pretrained text_model final hidden states; identical task language reuses exactsame row; no task-ID feature",
            "hidden_dim": embeddings.shape[-1],
            "max_length": 48,
            "task_ids": [t["global_task_id"] for t in tasks],
            "weights_sha256": digest(a.model / "model.safetensors"),
            "unique_texts": len(texts),
            "task_count": len(tasks),
        },
    )
    print("LANGUAGE_DONE", embeddings[indices].shape)


if __name__ == "__main__":
    main()
