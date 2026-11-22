"""Offline regression tests for the failures seen in run tools-20261121-154803 (full_campaign).

Each test names the account and day it comes from. They describe the behaviour tools.py
should have, so a test that fails marks a fix still to make. Everything here is free:
the logic tests drive tools._assess / tools._build_queue / tools.day_check through an
in-memory FakeCtx, and the environment tests use try_tools on the scripted practice cases.

Failure map:
  F1 due callbacks pushed behind other offers           -> QueuePriority
  F2 day ended with minutes and work left (day_check)   -> DayCheck
  F3 first contacts too late, lost to the day-7 cut     -> FirstContacts
  F4 contact before the decision day / decision offer   -> DecisionDay
  F5 repeated offers to a customer below the floor      -> BudgetBlocked
  F6 offer sent with no discovery                       -> ContactGuards
  F7 offer to a customer who already bought             -> ContactGuards
  F8 unfit accounts stay closed                         -> Regression
  F9 callback kept on its day                           -> Regression
"""

import importlib
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sales_sim.notebook import try_tools  # noqa: E402

tools = importlib.import_module("tools")


# --------------------------------------------------------------------------- the campaign as it was

PRODUCTS = [
    {"id": "basic", "name": "Basic", "monthly_price_cents": 3000, "monthly_service_cost_cents": 1000,
     "features": ["scheduling"], "integrations": ["CalendarLink"]},
    {"id": "pro", "name": "Pro", "monthly_price_cents": 6000, "monthly_service_cost_cents": 2000,
     "features": ["scheduling", "reminders"], "integrations": ["CalendarLink", "BookFlow"]},
    {"id": "premium", "name": "Premium", "monthly_price_cents": 10000, "monthly_service_cost_cents": 3500,
     "features": ["scheduling", "reminders", "location_reporting"], "integrations": ["CalendarLink", "BookFlow"]},
]
LAUNCH = {"id": "launch", "effective_day": 1, "max_discount_percent": 15, "quote_valid_days": 3}
STANDARD = {"id": "standard", "effective_day": 7, "max_discount_percent": 5, "quote_valid_days": 2}
CHANGE_DAY7 = {"effective_day": 7, "discount": 5, "valid": 2}     # as register_price_change stored it
CAMPAIGN_DAYS = 12
MINUTES_PER_EXCHANGE = 5


def lead(**fields):
    """A record_lead entry with the run's defaults; fields override."""
    base = {"needs": ["scheduling"], "nice_to_haves": [], "integrations": ["CalendarLink"],
            "budget_cents": None, "max_budget_cents": None, "ready_day": None, "scope_final": True,
            "open_question": "", "unsellable_reason": "", "notes": "", "plan_caps": {},
            "best_price_offered": False, "persuasions": 0, "last_persuade_day": None}
    base.update(fields)
    return base


# Leads copied from the run's custom_tool_state.json (trimmed to what the logic reads).
FABLE = lead(needs=["scheduling", "reminders"], budget_cents=5700, max_budget_cents=6000, ready_day=5,
             open_question="What happens to the trial and any deduction from a refund?")      # lead859
RIVERSIDE = lead(needs=["scheduling", "reminders"], nice_to_haves=["location_reporting"],
                 budget_cents=5800, max_budget_cents=10000, ready_day=5)                       # lead363
SOLSTICE = lead(budget_cents=3000, max_budget_cents=3000, ready_day=10, scope_final=False)       # lead440
SPRUCE_AFTER = lead(needs=["scheduling", "reminders"], integrations=["BookFlow"],
                    budget_cents=5800, max_budget_cents=5800, ready_day=6)                     # lead599, day 7
ANCHOR_LIKE = lead(needs=["scheduling", "reminders"], budget_cents=5100, max_budget_cents=5100,
                   ready_day=9)                     # fits only at the launch discount: lock before day 7
JUNIPER_CLOSED = lead(budget_cents=2700, max_budget_cents=2700, ready_day=1, plan_caps={"basic": 2700},
                      best_price_offered=True, dropped=True, close="budget")                    # lead956
ATLAS = lead(needs=["scheduling", "inventory_tracking"], ready_day=1, budget_cents=14000)        # lead162
SAFFRON = lead(needs=["scheduling", "card_payments"], ready_day=1, budget_cents=9000)            # lead628
EMBER = lead(needs=["scheduling", "reminders"], integrations=["OtherSuite"], budget_cents=7000)  # lead985


# --------------------------------------------------------------------------- a fake ctx

class FakeCtx:
    """Just the ctx surface the queue logic reads: state, status, catalog, customers, documents."""

    def __init__(self, day, minutes=60, price_change=True):
        self.day, self.minutes = day, minutes
        self.state = {"leads": {}, "price_changes": [dict(CHANGE_DAY7)] if price_change else []}
        self.accounts = {}

    def add(self, cid, lead_record=None, contacted=None, callback_day=None, quotes=(), last_role=None,
            status="prospect", background="A small service business."):
        """contacted: last contact day (None = never contacted)."""
        self.accounts[cid] = {
            "customer": {"id": cid, "name": f"Owner {cid}", "business": f"Business {cid}", "background": background},
            "status": status,
            "contact": {"eligible": True, "reason": None, "available_day": 1,
                        "last_contact_day": contacted, "last_decision": None},
            "last_message": {"role": last_role} if last_role else None,
            "pending_callbacks": ([{"id": f"cb-{cid}", "day": callback_day, "reason": "Customer can decide: offer."}]
                                  if callback_day else []),
            "quotes": list(quotes), "orders": [],
        }
        if lead_record is not None:
            self.state["leads"][cid] = dict(lead_record)
        return self

    def touched_today(self, cid, times=1):
        touches = self.state.setdefault("touches", {"day": self.day})
        touches["day"] = self.day
        touches[cid] = times
        return self

    # ctx surface ------------------------------------------------------------------
    def get_status(self):
        return {"day": self.day, "campaign_days": CAMPAIGN_DAYS, "minutes_remaining_today": self.minutes,
                "minutes_per_exchange": MINUTES_PER_EXCHANGE, "cost_per_exchange_cents": 50}

    def get_catalog(self):
        policy = STANDARD if (self.state["price_changes"] and self.day >= 7) else LAUNCH
        return {"day": self.day, "current_pricing_policy": dict(policy), "products": PRODUCTS}

    def list_customers(self):
        return [{"id": cid, "name": a["customer"]["name"], "business": a["customer"]["business"],
                 "background": a["customer"]["background"], "status": a["status"], "contact": a["contact"]}
                for cid, a in self.accounts.items()]

    def read_customer(self, customer_id):
        return self.accounts[customer_id]

    def list_documents(self):
        return []

    def read_document(self, document_id):
        raise KeyError(document_id)


def queue(ctx):
    return tools._build_queue(ctx)


def items_for(q, cid):
    return [item for item in q["items"] if item["cid"] == cid]


def messages_left(ctx):
    return ctx.minutes // MINUTES_PER_EXCHANGE


def pad_with_locks(ctx, count, day_ready=9):
    """Accounts that must lock a launch price today (day 6) to keep it valid to their decision day."""
    for i in range(count):
        ctx.add(f"lock{i}", ANCHOR_LIKE | {"ready_day": day_ready}, contacted=2)


# --------------------------------------------------------------------------- F1

class QueuePriority(unittest.TestCase):
    """F1. Days 5-6: callbacks for Riverside, Maple, Fable, Prairie and Spruce were listed after
    LOCK PRICE / OFFER items; day 6 ran out of minutes and 5 of 18 callbacks were missed."""

    def test_due_callback_ranks_before_other_offers(self):
        ctx = FakeCtx(day=6)
        pad_with_locks(ctx, 3)
        ctx.add("lead363", RIVERSIDE | {"ready_day": 6}, contacted=3, callback_day=6)   # added last on purpose
        q = queue(ctx)
        self.assertEqual(q["items"][0]["cid"], "lead363",
                         f"a due callback must come first, got {[i['text'] for i in q['items'][:4]]}")

    def test_every_due_callback_fits_in_todays_messages(self):
        for minutes in (20, 60, 120):
            with self.subTest(minutes=minutes):
                ctx = FakeCtx(day=6, minutes=minutes)
                pad_with_locks(ctx, 6)
                due = [f"due{i}" for i in range(min(3, messages_left(ctx)))]
                for cid in due:
                    ctx.add(cid, FABLE | {"ready_day": 6, "open_question": ""}, contacted=1, callback_day=6)
                head = {item["cid"] for item in queue(ctx)["items"][:messages_left(ctx)]}
                self.assertTrue(set(due) <= head, f"due callbacks outside today's {messages_left(ctx)} messages")

    def test_message_budget_comes_from_status_not_60(self):
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                ctx = FakeCtx(day=2, minutes=minutes)
                ctx.add("a", RIVERSIDE, contacted=1)
                self.assertEqual(queue(ctx)["messages_left_today"], minutes // MINUTES_PER_EXCHANGE)


# --------------------------------------------------------------------------- F2

class DayCheck(unittest.TestCase):
    """F2. Day 5: the model wrote "no contact minutes left" with 10 messages left and three
    callbacks due (Riverside, Maple, Fable). day_check must say whether the day may end."""

    def day_check(self, ctx):
        fn = getattr(tools, "day_check", None)
        if fn is None:
            self.fail("tools.day_check does not exist yet")
        result = fn(ctx)
        self.assertIn("ok_to_end", result)
        return result

    def test_day_check_is_a_tool(self):
        fn = getattr(tools, "day_check", None)
        self.assertIsNotNone(fn, "tools.day_check does not exist yet")
        self.assertTrue(hasattr(fn, "tool_schema"), "day_check must be decorated with @tool")

    def test_not_ok_to_end_with_minutes_and_work(self):
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                ctx = FakeCtx(day=5, minutes=minutes - 10)
                ctx.add("lead363", RIVERSIDE, contacted=3, callback_day=5)
                result = self.day_check(ctx)
                self.assertFalse(result["ok_to_end"])
                self.assertIn("lead363", str(result.get("do_next", "")), "say which account to contact next")
                self.assertEqual(result.get("messages_left_today"), (minutes - 10) // MINUTES_PER_EXCHANGE)

    def test_not_ok_to_end_with_first_contacts_left(self):
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                ctx = FakeCtx(day=5, minutes=minutes)
                ctx.add("new1").add("new2")
                self.assertFalse(self.day_check(ctx)["ok_to_end"])

    def test_ok_to_end_when_out_of_minutes(self):
        for minutes in (0, MINUTES_PER_EXCHANGE - 1):
            with self.subTest(minutes=minutes):
                ctx = FakeCtx(day=5, minutes=minutes)
                ctx.add("lead363", RIVERSIDE, contacted=3, callback_day=5)
                self.assertTrue(self.day_check(ctx)["ok_to_end"])

    def test_ok_to_end_when_nobody_needs_contact(self):
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                ctx = FakeCtx(day=2, minutes=minutes)
                ctx.add("lead859", FABLE | {"open_question": ""}, contacted=1)   # decides on day 5
                ctx.add("lead162", ATLAS | {"dropped": True, "close": "capability"}, contacted=1)
                self.assertTrue(self.day_check(ctx)["ok_to_end"])

    def test_budget_blocked_customer_counts_only_on_its_extra_day(self):
        ctx = FakeCtx(day=11)
        ctx.add("lead956", JUNIPER_CLOSED, contacted=10)
        self.assertFalse(self.day_check(ctx)["ok_to_end"], "the one extra contact is still due")
        ctx = FakeCtx(day=12)
        ctx.add("lead956", JUNIPER_CLOSED | {"persuasions": 1, "last_persuade_day": 11}, contacted=11)
        self.assertTrue(self.day_check(ctx)["ok_to_end"], "the extra day is used: nobody needs contact")


# --------------------------------------------------------------------------- F3

class FirstContacts(unittest.TestCase):
    """F3. Juniper (cap $27, ready day 1) was first contacted on day 10, when Basic's lowest price
    was $28.50 instead of $25.50. Same for Alder ($28, day 7) and Cedar ($55 vs Pro $51 -> $57, day 9).
    15 accounts were still uncontacted on day 6."""

    def scenario(self, minutes):
        ctx = FakeCtx(day=4, minutes=minutes)
        for i in range(2):
            ctx.add(f"due{i}", FABLE | {"ready_day": 4, "open_question": ""}, contacted=1, callback_day=4)
        for i in range(3):   # decide on day 9, have an open question: not time-critical today
            ctx.add(f"wait{i}", FABLE | {"ready_day": 9, "budget_cents": 6000}, contacted=2)
        for i in range(15):
            ctx.add(f"new{i:02d}")
        return ctx

    def test_first_contacts_fill_the_day_before_the_price_change(self):
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                ctx = self.scenario(minutes)
                q = queue(ctx)
                budget = messages_left(ctx)
                target = max(min(tools.DISCOVERY_TARGET_DAY, CHANGE_DAY7["effective_day"] - 1), ctx.day)
                needed = math.ceil(15 / (target - ctx.day + 1))
                expected = min(needed, budget - 2)              # only the 2 due callbacks may go first
                firsts = sum(item["kind"] == "first" for item in q["items"][:budget])
                self.assertGreaterEqual(firsts, expected,
                                        f"{firsts} first contacts inside today's {budget} messages")


# --------------------------------------------------------------------------- F4

class DecisionDay(unittest.TestCase):
    """F4. Fable got two offers on day 1 valid through day 4 (decides day 5); Solstice and Oakbridge
    got quotes on day 2 valid through day 5 (decide days 10 and 9). Before the decision day a customer
    is contacted only to lock a price; on the decision day one offer answers the question, asks
    whether anything changed, and quotes from what was recorded."""

    def test_no_contact_before_decision_day_without_a_lock(self):
        cases = [("lead859", FABLE, range(1, 5)), ("lead440", SOLSTICE | {"open_question": "Any setup charges?"},
                                                   range(2, 10))]
        for cid, record, days in cases:
            for day in days:
                with self.subTest(cid=cid, day=day):
                    ctx = FakeCtx(day=day)
                    ctx.add(cid, record, contacted=1)
                    self.assertEqual(items_for(queue(ctx), cid), [], "contact only on the decision day")

    def test_lock_quote_stays_valid_to_the_decision_day(self):
        for day in (3, 4, 5):
            with self.subTest(day=day):
                ctx = FakeCtx(day=day)
                ctx.add("lock", ANCHOR_LIKE, contacted=2)
                self.assertEqual(items_for(queue(ctx), "lock"), [], "too early to lock")
        ctx = FakeCtx(day=6)
        ctx.add("lock", ANCHOR_LIKE, contacted=2)
        own = items_for(queue(ctx), "lock")
        self.assertEqual([i["kind"] for i in own], ["lock"])
        valid = ctx.get_catalog()["current_pricing_policy"]["quote_valid_days"]
        self.assertGreaterEqual(ctx.day + valid, ANCHOR_LIKE["ready_day"])
        self.assertIn("price_cents=5100", own[0]["do"])

    def test_decision_day_offer_answers_then_asks_then_quotes(self):
        ctx = FakeCtx(day=5)
        ctx.add("lead859", FABLE, contacted=1, callback_day=5)
        own = items_for(queue(ctx), "lead859")
        self.assertEqual(len(own), 1)
        do = own[0]["do"]
        self.assertTrue(do.startswith("offer lead859"), "the decision-day message carries a quote")
        self.assertIn("price_cents=5700", do, "quote from the recorded budget")
        message = do.split("Message:", 1)[-1]
        answer_at = message.find(FABLE["open_question"][:40])
        changed = re.search(r"chang", message[answer_at + 1:]) if answer_at >= 0 else None
        self.assertGreaterEqual(answer_at, 0, "answer the open question first")
        self.assertIsNotNone(changed, "after the answer, ask whether needs or budget changed")

    def test_unsettled_scope_still_gets_the_original_quote(self):
        ctx = FakeCtx(day=10)
        ctx.add("lead440", SOLSTICE, contacted=2, callback_day=10)
        own = items_for(queue(ctx), "lead440")
        self.assertEqual(len(own), 1)
        self.assertTrue(own[0]["do"].startswith("offer lead440 (product_id='basic', price_cents=3000)"),
                        own[0]["do"][:120])

    def test_changed_scope_gets_a_new_quote(self):
        a = tools._assess(SPRUCE_AFTER, PRODUCTS, 7, CAMPAIGN_DAYS, {"discount": 5, "valid": 2}, [], [])
        self.assertEqual((a["kind"], a["plan"], a["price_cents"]), ("offer", "pro", 5800))


# --------------------------------------------------------------------------- F5

class BudgetBlocked(unittest.TestCase):
    """F5. About 25 losing offers to 9 accounts whose cap was below the lowest legal price (Chorus,
    Lumen, Finch, Meadow, Cedar, Juniper, Alder, Solstice, Oakbridge), Alder 3 times on day 8.
    Rule: after the decline, one more contact on one more day (moved to the next day with room if
    a day is full), then stop."""

    def test_declined_lowest_price_closes_on_budget(self):
        record = {k: v for k, v in JUNIPER_CLOSED.items() if k not in ("dropped", "close")}
        a = tools._assess(record, PRODUCTS, 10, CAMPAIGN_DAYS, {"discount": 5, "valid": 2}, [], [])
        self.assertEqual((a["kind"], a.get("close")), ("closed", "budget"))

    def test_no_second_offer_on_the_day_of_the_decline(self):
        ctx = FakeCtx(day=10)
        ctx.add("lead956", JUNIPER_CLOSED, contacted=10).touched_today("lead956", 2)
        self.assertEqual(items_for(queue(ctx), "lead956"), [])

    def test_one_extra_day_then_stop(self):
        ctx = FakeCtx(day=11)
        ctx.add("lead956", JUNIPER_CLOSED, contacted=10)
        self.assertEqual([i["kind"] for i in items_for(queue(ctx), "lead956")], ["persuade"])
        ctx = FakeCtx(day=12)
        ctx.add("lead956", JUNIPER_CLOSED | {"persuasions": 1, "last_persuade_day": 11}, contacted=11)
        self.assertEqual(items_for(queue(ctx), "lead956"), [], "the extra day is used")

    def test_a_full_day_moves_the_extra_contact_forward(self):
        ctx = FakeCtx(day=12)
        ctx.add("lead956", JUNIPER_CLOSED, contacted=10)        # day 11 had no room
        self.assertEqual([i["kind"] for i in items_for(queue(ctx), "lead956")], ["persuade"])

    def test_extra_contact_ranks_after_time_critical_work(self):
        ctx = FakeCtx(day=6)
        ctx.add("lead956", JUNIPER_CLOSED, contacted=5)
        ctx.add("lead363", RIVERSIDE | {"ready_day": 6}, contacted=3, callback_day=6)
        kinds = [i["cid"] for i in queue(ctx)["items"]]
        self.assertLess(kinds.index("lead363"), kinds.index("lead956"))


# --------------------------------------------------------------------------- F6, F7

def outputs(sequence):
    """Tool outputs in order; "end_day" steps have none and are skipped."""
    return [call["output"] for call in sequence["calls"] if "output" in call]


def as_dict(output):
    import json
    return json.loads(output) if isinstance(output, str) and output.startswith("{") else output


class ContactGuards(unittest.TestCase):
    """F6: Copper Bicycle's first message on day 10 was a Basic offer; it needs OtherSuite.
    F7: Keystone (lead292) accepted on day 5 and was offered again."""

    def test_offer_refused_before_discovery(self):
        sequence = try_tools([
            ("offer", {"customer_id": "nimbus", "product_id": "pro", "price_cents": 6000,
                       "message": "Pro is $60.00 per month."}),
            ("work_queue", {}),
        ], case="requirements_update", root=ROOT)
        refused, after = (as_dict(o) for o in outputs(sequence))
        self.assertIn("error", refused, "an offer needs a recorded first contact")
        self.assertEqual(after["messages_left_today"], 24, "a refused offer costs no minutes")

    def test_offer_to_a_customer_who_bought_is_refused_and_free(self):
        sequence = try_tools([
            ("work_queue", {}),                                             # sends the first contact
            ("record_lead", {"customer_id": "nimbus", "needs": ["scheduling", "reminders"], "nice_to_haves": [],
                             "integrations": ["BookFlow"], "budget_cents": 6000, "max_budget_cents": None,
                             "ready_day": None, "scope_final": True, "open_question": "",
                             "unsellable_reason": "", "notes": ""}),
            ("offer", {"customer_id": "nimbus", "product_id": "pro", "price_cents": 6000,
                       "message": "Pro covers BookFlow and reminders at $60.00 per month."}),
            ("offer", {"customer_id": "nimbus", "product_id": "pro", "price_cents": 6000,
                       "message": "Pro is still $60.00 per month."}),
            ("work_queue", {}),
        ], case="requirements_update", root=ROOT)
        _, _, accepted, again, after = (as_dict(o) for o in outputs(sequence))
        self.assertEqual(accepted.get("decision"), "accept")
        self.assertTrue(str(accepted.get("do_next", "")).startswith("work_queue"), "stop after an acceptance")
        self.assertIn("error", again)
        self.assertEqual(after["messages_left_today"], 22, "only the first contact and one offer used minutes")


# --------------------------------------------------------------------------- F8, F9

class Regression(unittest.TestCase):
    """F8: Atlas (inventory), Saffron (deposits), Ember (OtherSuite) got one message and nothing more.
    F9: a callback is completed on its day."""

    def test_missing_capability_closes(self):
        for cid, record in (("lead162", ATLAS), ("lead628", SAFFRON), ("lead985", EMBER)):
            with self.subTest(cid=cid):
                a = tools._assess(record, PRODUCTS, 3, CAMPAIGN_DAYS, {"discount": 15, "valid": 3}, [], [])
                self.assertEqual((a["kind"], a.get("close")), ("closed", "capability"))
                ctx = FakeCtx(day=3)
                ctx.add(cid, record | {"dropped": True, "close": "capability"}, contacted=1)
                self.assertEqual(items_for(queue(ctx), cid), [])

    def test_callback_kept_on_its_day(self):
        sequence = try_tools([
            ("work_queue", {}),                                             # first contact to atlas
            ("record_lead", {"customer_id": "atlas", "needs": ["scheduling"], "nice_to_haves": [],
                             "integrations": ["CalendarLink"], "budget_cents": 3000, "max_budget_cents": 3000,
                             "ready_day": 2, "scope_final": True, "open_question": "",
                             "unsellable_reason": "", "notes": "Follow up on day 2 with Basic at $30."}),
            "end_day",
            ("work_queue", {}),
            ("offer", {"customer_id": "atlas", "product_id": "basic", "price_cents": 3000,
                       "message": "Basic at $30.00 per month, as promised for today."}),
            "end_day",
        ], case="callback_followthrough", root=ROOT)
        day2 = as_dict(outputs(sequence)[2])
        self.assertIn("atlas", day2["todo"][0])
        self.assertTrue(sequence["finished"])
        self.assertEqual(sequence["outcome"]["missed_callbacks"], 0)
        self.assertEqual(sequence["outcome"]["accepted_orders"], 1)


if __name__ == "__main__":
    unittest.main()
