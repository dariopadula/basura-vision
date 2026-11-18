from __future__ import annotations

import argparse
from pathlib import Path

from .config import ROOT, load_settings
from .storage import download_run, photo_storage, upload_run


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("debe ser un entero mayor que cero")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="basura-vision"); parser.add_argument("--env-file", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect"); detect.add_argument("--manifest", type=Path, required=True); detect.add_argument("--run-id", required=True)
    embed = commands.add_parser("embed"); embed.add_argument("--run-id", required=True)
    group = commands.add_parser("cluster"); group.add_argument("--run-id", required=True); group.add_argument("--clusters", type=positive_integer, default=15); group.add_argument("--pca-components", type=positive_integer, default=30)
    viewer = commands.add_parser("viewer"); viewer.add_argument("--run-id", required=True); viewer.add_argument("--no-browser", action="store_true")
    annotator = commands.add_parser("annotator"); annotator.add_argument("--manifest", type=Path, required=True); annotator.add_argument("--no-browser", action="store_true")
    sync = commands.add_parser("upload-run"); sync.add_argument("--run-id", required=True); sync.add_argument("--destination")
    download = commands.add_parser("download-run"); download.add_argument("--run-id", required=True); download.add_argument("--source")
    prepare = commands.add_parser("prepare-dataset"); prepare.add_argument("--version", required=True); prepare.add_argument("--output", type=Path, default=ROOT / "training_data")
    training = commands.add_parser("train"); training.add_argument("--dataset", type=Path, required=True); training.add_argument("--initial-weight", type=Path, required=True); training.add_argument("--output", type=Path, required=True); training.add_argument("--epochs", type=int, default=50)
    bedrock = commands.add_parser("bedrock-one"); bedrock.add_argument("--run-id", required=True); bedrock.add_argument("--photo-id", required=True); bedrock.add_argument("--model-id"); bedrock.add_argument("--region")
    batch = commands.add_parser("bedrock-batch"); batch.add_argument("--run-id", required=True); batch.add_argument("--label-run-id", required=True); batch.add_argument("--model-id"); batch.add_argument("--region"); batch.add_argument("--checkpoint-every", type=positive_integer, default=5); batch.add_argument("--no-upload", action="store_true")
    args = parser.parse_args(argv); settings = load_settings(args.env_file)
    if args.command == "detect":
        from .detection import run_detection
        storage = photo_storage(settings)
        run_detection(settings, storage, args.manifest.resolve(), args.run_id)
    elif args.command == "embed":
        from .clustering import extract_embeddings
        extract_embeddings(settings.runs_root / args.run_id, settings.device)
    elif args.command == "cluster":
        from .clustering import cluster_run
        cluster_run(settings.runs_root / args.run_id, clusters=args.clusters, pca_components=args.pca_components)
    elif args.command == "viewer":
        from .viewer import serve_viewer
        storage = photo_storage(settings)
        serve_viewer(settings, storage, args.run_id, not args.no_browser)
    elif args.command == "annotator":
        from .annotator import serve_annotator
        storage = photo_storage(settings)
        serve_annotator(settings, storage, args.manifest.resolve(), not args.no_browser)
    elif args.command == "upload-run":
        storage = photo_storage(settings)
        print("\n".join(upload_run(storage, settings.runs_root / args.run_id, args.destination or f"runs/{args.run_id}")))
    elif args.command == "download-run":
        storage = photo_storage(settings)
        print("\n".join(str(path) for path in download_run(storage, args.source or f"runs/{args.run_id}", settings.runs_root / args.run_id)))
    elif args.command == "prepare-dataset":
        from .training import prepare_dataset
        storage = photo_storage(settings)
        prepare_dataset(settings, storage, settings.annotations_path, args.version, args.output.resolve())
    elif args.command == "train":
        from .training import train
        train(args.dataset.resolve(), args.initial_weight.resolve(), args.output.resolve(), args.epochs)
    elif args.command == "bedrock-one":
        from .bedrock import label_one_photo
        storage = photo_storage(settings)
        destination = label_one_photo(
            settings,
            storage,
            args.run_id,
            args.photo_id,
            args.model_id or settings.bedrock_model_id,
            args.region or settings.bedrock_region,
        )
        print(destination)
    elif args.command == "bedrock-batch":
        from .bedrock import label_batch
        storage = photo_storage(settings)
        destination = None
        if settings.storage == "s3" and not args.no_upload:
            destination = f"runs/{args.run_id}/bedrock/{args.label_run_id}"
        label_dir = label_batch(
            settings,
            storage,
            args.run_id,
            args.label_run_id,
            args.model_id or settings.bedrock_model_id,
            args.region or settings.bedrock_region,
            args.checkpoint_every,
            destination,
        )
        print(label_dir)
        if destination:
            print(f"S3: {destination}")


if __name__ == "__main__": main()

