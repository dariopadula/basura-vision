import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from basura_vision.bedrock import CATEGORIES
from basura_vision.bedrock_review import (
    load_blind_reviews,
    persist_blind_reviews,
    save_blind_reviews,
    validate_blind_review,
)
from basura_vision.cli import main


def valid_review():
    return {
        "categories": {
            category: {
                "presente": category == "poda",
                "severidad": "medio" if category == "poda" else None,
                "evidencia_dudosa": category == "contenedor_roto",
            }
            for category in CATEGORIES
        },
        "notes": "Caso revisado sin ver el resultado del modelo.",
    }


class BlindReviewTests(unittest.TestCase):
    def test_validates_and_roundtrips_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind_reviews.json"
            review = validate_blind_review(valid_review())
            save_blind_reviews(path, {"photo_1": {"id": "photo_1", **review}})

            loaded = load_blind_reviews(path)

            self.assertEqual(loaded["photo_1"]["categories"]["poda"]["severidad"], "medio")
            self.assertTrue(
                loaded["photo_1"]["categories"]["contenedor_roto"]["evidencia_dudosa"]
            )

    def test_persists_review_to_selected_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind_reviews.json"
            storage = unittest.mock.Mock()
            persist_blind_reviews(
                path,
                {"photo_1": {"id": "photo_1"}},
                storage,
                "runs/reference/bedrock/pilot40",
            )
            storage.upload_file.assert_called_once_with(
                path,
                "runs/reference/bedrock/pilot40/blind_reviews.json",
            )

    def test_rejects_inconsistent_severity(self):
        review = valid_review()
        review["categories"]["basura_fuera"]["severidad"] = "severo"
        with self.assertRaisesRegex(ValueError, "severidad debe ser null"):
            validate_blind_review(review)

    def test_review_cli_uses_storage_but_not_bedrock(self):
        settings = SimpleNamespace(runs_root=Path("runs"), storage="local")
        storage = object()
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", return_value=storage) as photo_storage,
            patch("basura_vision.bedrock_review.serve_blind_review") as serve,
        ):
            main([
                "bedrock-review",
                "--run-id", "reference",
                "--label-run-id", "pilot40",
                "--no-browser",
            ])
        photo_storage.assert_called_once_with(settings)
        serve.assert_called_once_with(settings, storage, "reference", "pilot40", False, None)


if __name__ == "__main__":
    unittest.main()
