#!/usr/bin/env python3
"""Fetch the base model weights.

One model per run, used for two jobs at once: it is the agent being trained,
and it is the world model that grades. That is deliberate rather than
convenient. If the world model were a stronger model, any improvement could be
distillation from it, and the result would say nothing about world models. With
the same backbone on both sides there is no stronger model in the loop for the
agent to learn from.

    python -m ml_research.data.fetch_models --out models
    python -m ml_research.data.fetch_models --out models --model qwen3.5-4b
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

# The two scales in the paper. Point --repo elsewhere to use a different model:
# nothing in the method depends on this particular family.
MODELS = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
}

__all__ = ["fetch", "main"]


def _looks_complete(path: pathlib.Path) -> bool:
    """A config plus at least one weight shard."""
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def fetch(repo_id: str, dest: pathlib.Path, revision: str | None = None) -> pathlib.Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is needed to fetch weights:\n"
            "  pip install huggingface_hub\n"
            f"Or download {repo_id} yourself and point model_path at it."
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetching {repo_id} -> {dest}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
    )
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory to hold model directories")
    ap.add_argument("--model", default="qwen3.5-9b", choices=sorted(MODELS) + ["all"])
    ap.add_argument("--repo", help="fetch this hub repo instead of a known name")
    ap.add_argument("--revision", help="pin a revision")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)

    if args.repo:
        wanted = {pathlib.Path(args.repo).name.lower(): args.repo}
    elif args.model == "all":
        wanted = dict(MODELS)
    else:
        wanted = {args.model: MODELS[args.model]}

    free_gb = shutil.disk_usage(out.parent if out.parent.exists() else ".").free / 2**30
    need_gb = 20 * len(wanted)
    if free_gb < need_gb:
        print(f"warning: {free_gb:.0f}GB free, roughly {need_gb}GB needed", file=sys.stderr)

    for name, repo in wanted.items():
        dest = out / name
        if _looks_complete(dest) and not args.force:
            print(f"{name}: already present in {dest}", flush=True)
            continue
        fetch(repo, dest, args.revision)
        if not _looks_complete(dest):
            print(f"error: {name} downloaded but looks incomplete in {dest}", file=sys.stderr)
            return 1
        print(f"{name}: ready", flush=True)

    print(f"\nSet model_path in your config to one of: "
          f"{', '.join(str(out / n) for n in wanted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
