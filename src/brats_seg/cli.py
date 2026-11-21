"""Command line: download, preprocess, train, evaluate, analyse."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="brats-seg")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--dest", required=True); d.add_argument("--entity", default="syn51514105")
    pr = sub.add_parser("preprocess"); pr.add_argument("--raw", required=True); pr.add_argument("--out", required=True)
    pr.add_argument("--size", type=int, default=128); pr.add_argument("--workers", type=int, default=4)
    pr.add_argument("--limit", type=int, default=None)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model", choices=["unet", "swinunetr"], required=True); t.add_argument("--seeds", default="0")
    t.add_argument("--epochs", type=int, default=100); t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--val-every", type=int, default=5); t.add_argument("--limit-steps", type=int, default=0)
    t.add_argument("--max-val-cases", type=int, default=0)
    e = sub.add_parser("evaluate"); e.add_argument("--data", required=True); e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True); e.add_argument("--limit", type=int, default=0)
    a = sub.add_parser("analyse"); a.add_argument("--reports", required=True)
    args = p.parse_args(argv)

    if args.cmd == "download":
        from .data import download_from_synapse
        root = download_from_synapse(args.dest, args.entity, os.environ.get("SYNAPSE_AUTH_TOKEN"))
        print((root / "download_manifest.json").read_text())
    elif args.cmd == "preprocess":
        from .data import preprocess_all
        df = preprocess_all(args.raw, args.out, args.size, args.workers, args.limit)
        print(df.groupby("split").agg(cases=("case_id", "size"), patients=("patient", "nunique")).to_string())
    elif args.cmd == "train":
        from .train import train_one
        for seed in [int(s) for s in args.seeds.split(",")]:
            if (Path(args.out) / f"{args.model}_s{seed}_meta.json").exists():
                print(f"{args.model} seed {seed}: already trained, skipping", flush=True)
                continue
            print(json.dumps(train_one(args.model, args.data, args.out, seed, args.epochs, args.batch_size,
                                       args.val_every, args.limit_steps, args.max_val_cases)), flush=True)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate_models
        df = evaluate_models(args.data, args.runs, args.out, limit=args.limit)
        print(df.groupby(["model", "seed"])[[c for c in df.columns if c.startswith("dice_")]].mean().round(4).to_string())
    elif args.cmd == "analyse":
        import pandas as pd
        from .evaluate import summarise, summary_markdown
        summary = summarise(pd.read_csv(Path(args.reports) / "test_per_case.csv"))
        (Path(args.reports) / "summary.json").write_text(json.dumps(summary, indent=2))
        md = summary_markdown(summary)
        (Path(args.reports) / "summary.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
