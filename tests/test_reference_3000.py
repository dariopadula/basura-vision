import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Reference3000Tests(unittest.TestCase):
    def test_frozen_manifest(self):
        rows = [
            json.loads(line)
            for line in (ROOT / "manifests/reference_3000.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

        self.assertEqual(len(rows), 3000)
        self.assertEqual(len({row["id"] for row in rows}), 3000)
        self.assertEqual(len({row["path"] for row in rows}), 3000)
        self.assertTrue(all(row["type"] == 1 for row in rows))
        self.assertEqual(
            Counter(row["reference_status"] for row in rows),
            Counter(
                {
                    "1": 183,
                    "2": 151,
                    "3": 225,
                    "4": 162,
                    "5": 71,
                    "6": 243,
                    "7": 202,
                    "8": 160,
                    "9": 112,
                    "10": 125,
                    "11": 199,
                    "12": 181,
                    "13": 164,
                    "14": 87,
                    "15": 59,
                    "sin_contenedor": 57,
                    "sin_residuos": 619,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
