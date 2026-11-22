"""Offline checks for prompt.py and tools.py. Run them with `python run.py test`.

The first class checks that your files load the way a run loads them. The second
shows how to test a tool on a practice case without paying: copy a test and adapt
it for each tool you write.
"""

import json
import unittest
from pathlib import Path

from sales_sim import tool
from sales_sim.notebook import try_tool, try_tools
from sales_sim.student_setup import model_input, tool_definitions
from sales_sim.student_submission import VARIANTS, load_submission

ROOT = Path(__file__).resolve().parents[1]


class YourFilesLoad(unittest.TestCase):
    def test_every_variant_loads(self):
        """baseline, prompt and tools all build before any paid run."""
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                prompt, functions = load_submission(ROOT, variant)._configuration()
                self.assertTrue(prompt.strip())
                self.assertTrue(functions)

    def test_tool_definitions_are_plain_json(self):
        definitions = tool_definitions(ROOT)
        names = [item["name"] for item in definitions]
        self.assertEqual(len(names), len(set(names)))
        json.dumps(definitions, allow_nan=False)
        for item in definitions:
            with self.subTest(tool=item["name"]):
                self.assertTrue(item["description"].strip(), "Tell the model when to use this tool.")

    def test_the_model_can_end_its_day(self):
        """The calendar belongs to the harness: no tool may be called advance_day."""
        self.assertNotIn("advance_day", [item["name"] for item in tool_definitions(ROOT)])

    def test_your_prompt_is_the_whole_first_message(self):
        """Nothing is added to SYSTEM_PROMPT: it must explain the job itself."""
        seen = model_input(ROOT, case="requirements_update")
        prompt, _ = load_submission(ROOT, "tools")._configuration()
        self.assertEqual(seen["developer"], prompt)
        self.assertIn("Day 1 of 1", seen["user"])


class ExampleToolTests(unittest.TestCase):
    """Patterns for testing tools offline on the deterministic practice cases."""

    def test_a_read_returns_data_the_model_can_use(self):
        check = try_tool("read_customer_history", {"customer_id": "nimbus"}, case="requirements_update",
                         root=ROOT, variant="baseline")
        self.assertIsNone(check["error"])
        self.assertEqual(check["result"]["customer"]["id"], "nimbus")

    def test_a_refusal_reaches_the_model_as_an_error(self):
        check = try_tool("read_customer_history", {"customer_id": "nobody"}, case="requirements_update",
                         root=ROOT, variant="baseline")
        self.assertEqual(check["result"]["error"]["code"], "not_found")

    def test_a_new_tool_before_it_goes_into_tools_py(self):
        @tool
        def legal_prices(ctx, product_id: str) -> dict:
            """The lowest and highest monthly price allowed today for one product, in cents."""
            catalog = ctx.get_catalog()
            policy = catalog["current_pricing_policy"]
            for product in catalog["products"]:
                if product["id"] == product_id:
                    list_price = product["monthly_price_cents"]
                    discounted = -(-list_price * (100 - policy["max_discount_percent"]) // 100)  # rounded up
                    return {"min_cents": max(discounted, product["monthly_service_cost_cents"]),
                            "max_cents": list_price}
            return {"error": f"No product {product_id!r}."}

        check = try_tool("legal_prices", {"product_id": "pro"}, case="requirements_update",
                         root=ROOT, tools=[legal_prices])
        self.assertEqual(check["result"], {"min_cents": 5100, "max_cents": 6000})
        self.assertLess(check["calls"][0]["approx_tokens"], 50)     # a small output keeps the window small
        # The environment agrees: one cent below the minimum is refused.
        refused = try_tool("create_quote", {"customer_id": "nimbus", "product_id": "pro", "monthly_price_cents": 5099},
                           case="requirements_update", root=ROOT, variant="baseline")
        self.assertEqual(refused["result"]["error"]["code"], "invalid_price")

    def test_a_whole_campaign_settles_with_a_profit(self):
        sequence = try_tools([
            ("create_quote", {"customer_id": "nimbus", "product_id": "pro", "monthly_price_cents": 6000}),
            ("present_quote", {"quote_id": "run:quote:Q0001", "message": "Pro covers BookFlow and reminders."}),
            "end_day",
        ], case="requirements_update", root=ROOT, variant="baseline")
        self.assertTrue(sequence["finished"])
        self.assertEqual(sequence["outcome"]["accepted_orders"], 1)


if __name__ == "__main__":
    unittest.main()
