from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from .config import ROOT, load_settings
from .storage import canonical_run_files, download_run, photo_storage, upload_run


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
    sync = commands.add_parser("upload-run"); sync.add_argument("--run-id", required=True); sync.add_argument("--destination"); sync.add_argument("--canonical", action="store_true"); sync.add_argument("--dry-run", action="store_true")
    download = commands.add_parser("download-run"); download.add_argument("--run-id", required=True); download.add_argument("--source")
    prepare = commands.add_parser("prepare-dataset"); prepare.add_argument("--version", required=True); prepare.add_argument("--output", type=Path, default=ROOT / "training_data")
    training = commands.add_parser("train"); training.add_argument("--dataset", type=Path, required=True); training.add_argument("--initial-weight", type=Path, required=True); training.add_argument("--output", type=Path, required=True); training.add_argument("--epochs", type=int, default=50)
    bedrock = commands.add_parser("bedrock-one"); bedrock.add_argument("--run-id", required=True); bedrock.add_argument("--photo-id", required=True); bedrock.add_argument("--model-id"); bedrock.add_argument("--region")
    bedrock_prepare = commands.add_parser("bedrock-prepare"); bedrock_prepare.add_argument("--run-id", required=True); bedrock_prepare.add_argument("--label-run-id", required=True); bedrock_prepare.add_argument("--model-id"); bedrock_prepare.add_argument("--region"); bedrock_prepare.add_argument("--max-cost-usd"); bedrock_prepare.add_argument("--approve-execution", action="store_true")
    batch = commands.add_parser("bedrock-batch"); batch.add_argument("--run-id", required=True); batch.add_argument("--label-run-id", required=True); batch.add_argument("--model-id"); batch.add_argument("--region"); batch.add_argument("--checkpoint-every", type=positive_integer, default=5); batch.add_argument("--max-attempts", type=positive_integer, default=3); batch.add_argument("--destination"); batch.add_argument("--no-upload", action="store_true"); batch.add_argument("--execute-approved", action="store_true")
    bedrock_restore = commands.add_parser("bedrock-restore"); bedrock_restore.add_argument("--run-id", required=True); bedrock_restore.add_argument("--label-run-id", required=True); bedrock_restore.add_argument("--source", required=True)
    args = parser.parse_args(argv); settings = load_settings(args.env_file)
    if args.command == "detect":
        from .detection import run_detection
        storage = photo_storage(settings)
        run_detection(settings, storage, args.manifest.resolve(), args.run_id)
    elif args.command == "embed":
        from .clustering import extract_embeddings
        storage = photo_storage(settings)
        extract_embeddings(settings.runs_root / args.run_id, storage, settings.device)
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
        run_dir = settings.runs_root / args.run_id
        sources = canonical_run_files(run_dir) if args.canonical else sorted(
            path for path in run_dir.rglob("*") if path.is_file()
        )
        if args.dry_run:
            locations = Counter(
                relative.parts[0] if len(relative.parts) > 1 else "[raiz]"
                for path in sources
                for relative in [path.relative_to(run_dir)]
            )
            root_files = sorted(
                path.name for path in sources if path.parent == run_dir
            )
            print(f"Archivos seleccionados: {len(sources)}")
            print(f"Por ubicación: {dict(sorted(locations.items()))}")
            print(f"Archivos raíz: {root_files}")
        else:
            storage = photo_storage(settings)
            print("\n".join(upload_run(
                storage,
                run_dir,
                args.destination or f"runs/{args.run_id}",
                sources,
            )))
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
    elif args.command == "bedrock-prepare":
        from .bedrock import prepare_label_run
        region = (
            args.region
            or settings.bedrock_region
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
        )
        if args.approve_execution and args.max_cost_usd is None:
            raise ValueError("--approve-execution requiere --max-cost-usd")
        label_dir = prepare_label_run(
            settings,
            args.run_id,
            args.label_run_id,
            args.model_id or settings.bedrock_model_id,
            region or "",
            max_estimated_cost_usd=args.max_cost_usd,
            approved=args.approve_execution,
        )
        print(label_dir)
    elif args.command == "bedrock-batch":
        from .bedrock import label_batch
        if not args.execute_approved:
            raise ValueError(
                "Preparar y aprobar primero; luego usar --execute-approved"
            )
        storage = photo_storage(settings)
        destination = None
        if settings.storage == "s3" and not args.no_upload:
            destination = args.destination or f"runs/{args.run_id}/bedrock/{args.label_run_id}"
        region = (
            args.region
            or settings.bedrock_region
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
        )
        label_dir = label_batch(
            settings,
            storage,
            args.run_id,
            args.label_run_id,
            args.model_id or settings.bedrock_model_id,
            region,
            args.checkpoint_every,
            destination,
            require_approval=True,
            max_attempts_per_photo=args.max_attempts,
        )
        print(label_dir)
        if destination:
            print(f"S3: {destination}")
    elif args.command == "bedrock-restore":
        from .bedrock import restore_label_run
        storage = photo_storage(settings)
        destination = settings.runs_root / args.run_id / "bedrock" / args.label_run_id
        print(restore_label_run(storage, args.source, destination))


if __name__ == "__main__": main()

