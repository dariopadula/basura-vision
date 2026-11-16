import argparse
import json
import math
from pathlib import Path


def rows(path):
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir", type=Path); parser.add_argument("--ratio-tolerance", type=float, default=1e-9); args = parser.parse_args()
    reference = rows(Path(__file__).resolve().parents[1] / "manifests/reference_30.jsonl")
    detected = rows(args.run_dir / "detections.jsonl")
    differences = []
    for photo_id, expected in reference.items():
        actual = detected.get(photo_id)
        expected_status = "agrupable" if expected["reference_cluster"] is not None else expected["reference_status"]
        if actual is None: differences.append({"id": photo_id, "field": "missing"}); continue
        if actual["status"] != expected_status: differences.append({"id": photo_id, "field": "status", "expected": expected_status, "actual": actual["status"]})
        if actual.get("crop_box") != expected.get("reference_crop_box"): differences.append({"id": photo_id, "field": "crop_box", "expected": expected.get("reference_crop_box"), "actual": actual.get("crop_box")})
        if expected.get("reference_ratio") is not None and not math.isclose(actual.get("ratio", float("nan")), expected["reference_ratio"], rel_tol=args.ratio_tolerance, abs_tol=args.ratio_tolerance):
            differences.append({"id": photo_id, "field": "ratio", "expected": expected["reference_ratio"], "actual": actual.get("ratio")})
    print(json.dumps({"checked": len(reference), "differences": differences}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(differences))


if __name__ == "__main__": main()

