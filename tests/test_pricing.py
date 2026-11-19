import tempfile
import unittest
from pathlib import Path

from basura_vision.pricing import freeze_price, pricing_sha256


HEADER = (
    "provider,model_name,model_id,region,routing,service_tier,inference_mode,"
    "currency,input_usd_per_million,output_usd_per_million,effective_from,"
    "checked_at,source_url\n"
)
QUERY = {
    "model_id": "amazon.nova-lite-v1:0",
    "region": "us-east-1",
    "routing": "direct",
    "service_tier": "standard",
    "inference_mode": "on_demand",
    "currency": "USD",
}


class PricingTests(unittest.TestCase):
    def test_frozen_price_does_not_change_when_catalog_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "pricing.csv"
            destination = root / "pricing.json"
            catalog.write_text(
                HEADER
                + "AWS,Nova Lite,amazon.nova-lite-v1:0,us-east-1,direct,standard,"
                "on_demand,USD,0.06,0.24,,2026-11-19,https://example.test/pricing\n",
                encoding="utf-8",
            )
            first = freeze_price(destination, QUERY, catalog)
            first_hash = pricing_sha256(destination)

            catalog.write_text(
                HEADER
                + "AWS,Nova Lite,amazon.nova-lite-v1:0,us-east-1,direct,standard,"
                "on_demand,USD,9.99,9.99,,2026-11-20,https://example.test/pricing\n",
                encoding="utf-8",
            )
            second = freeze_price(destination, QUERY, catalog)

            self.assertEqual(first, second)
            self.assertEqual(pricing_sha256(destination), first_hash)
            self.assertEqual(second["rate"]["input_usd_per_million"], "0.06")
            self.assertEqual(second["rate"]["output_usd_per_million"], "0.24")


if __name__ == "__main__":
    unittest.main()
