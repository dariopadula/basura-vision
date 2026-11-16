from __future__ import annotations

import argparse
from pathlib import Path

from .config import ROOT, load_settings
from .storage import download_run, photo_storage, upload_run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="basura-vision"); parser.add_argument("--env-file", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect"); detect.add_argument("--manifest", type=Path, required=True); detect.add_argument("--run-id", required=True)
    embed = commands.add_parser("embed"); embed.add_argument("--run-id", required=True)
    group = commands.add_parser("cluster"); group.add_argument("--run-id", required=True); group.add_argument("--clusters", type=int, default=15)
    viewer = commands.add_parser("viewer"); viewer.add_argument("--run-id", required=True); viewer.add_argument("--no-browser", action="store_true")
    annotator = commands.add_parser("annotator"); annotator.add_argument("--manifest", type=Path, required=True); annotator.add_argument("--no-browser", action="store_true")
    sync = commands.add_parser("upload-run"); sync.add_argument("--run-id", required=True); sync.add_argument("--destination")
    download = commands.add_parser("download-run"); download.add_argument("--run-id", required=True); download.add_argument("--source")
    prepare = commands.add_parser("prepare-dataset"); prepare.add_argument("--version", required=True); prepare.add_argument("--output", type=Path, default=ROOT / "training_data")
    training = commands.add_parser("train"); training.add_argument("--dataset", type=Path, required=True); training.add_argument("--initial-weight", type=Path, required=True); training.add_argument("--output", type=Path, required=True); training.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv); settings = load_settings(args.env_file); storage = photo_storage(settings)
    if args.command == "detect":
        from .detection import run_detection
        run_detection(settings, storage, args.manifest.resolve(), args.run_id)
    elif args.command == "embed":
        from .clustering import extract_embeddings
        extract_embeddings(settings.runs_root / args.run_id)
    elif args.command == "cluster":
        from .clustering import cluster_run
        cluster_run(settings.runs_root / args.run_id, args.clusters)
    elif args.command == "viewer":
        from .viewer import serve_viewer
        serve_viewer(settings, storage, args.run_id, not args.no_browser)
    elif args.command == "annotator":
        from .annotator import serve_annotator
        serve_annotator(settings, storage, args.manifest.resolve(), not args.no_browser)
    elif args.command == "upload-run": print("\n".join(upload_run(storage, settings.runs_root / args.run_id, args.destination or f"runs/{args.run_id}")))
    elif args.command == "download-run": print("\n".join(str(path) for path in download_run(storage, args.source or f"runs/{args.run_id}", settings.runs_root / args.run_id)))
    elif args.command == "prepare-dataset":
        from .training import prepare_dataset
        prepare_dataset(settings, storage, settings.annotations_path, args.version, args.output.resolve())
    elif args.command == "train":
        from .training import train
        train(args.dataset.resolve(), args.initial_weight.resolve(), args.output.resolve(), args.epochs)


if __name__ == "__main__": main()

