"""Day-by-day lifecycle tests for locked prices and budget-blocked customers.

They drive the real tools (offer, send_message, record_lead, _build_queue) against scripted
customers in an in-memory campaign, the way the model does: take the queue's first item, make
that call, record the reply, repeat until the day's minutes or work run out, end the day.

Bugs from run tools-20261124-224731 they pin down:
  A. Kestrel's $92 lock was presented 4 times on day 6 and Cobalt's twice, so the day ran out
     before Anchor's $51 lock was sent; Anchor was lost on day 9.
  B. Meadow (cap $80) was re-presented the $85 quote it had declined, 3 times on day 8.
  C. Juniper (willing $27, approved $30) got 3 offers at $28.50 on day 7.
Rules they also hold: one final offer, one extra spare-time contact on a later day, CONFIRM before
quoting a pending-decision lock, a fresh higher quote when the budget rises past today's floor.
"""

import re
import unittest

from test_failures import CHANGE_DAY7, LAUNCH, MINUTES_PER_EXCHANGE, PRODUCTS, STANDARD, tools

CAMPAIGN_DAYS = 12
MINUTES_PER_DAY = 60


def money(cents):
    return f"${cents / 100:.2f}"


class Customer:
    """A scripted customer.

    cap: what they will pay per month, from a day on ({day: cents}, latest key <= today wins).
    ready: earliest day they can approve. scope_final: False while their budget hangs on a decision,
    until decided_day. stated: the "willing to spend" figure they give at discovery (default: cap).
    """

    def __init__(self, cid, needs, cap, ready, integrations=("CalendarLink",), scope_final=True,
                 decided_day=None, approved=None, later_needs=None, background="A small service business.",
                 pending=()):
        self.cid, self.integrations = cid, list(integrations)
        self.needs_by_day = {1: list(needs), **(later_needs or {})}   # needs that change on a decision day
        self.background = background
        self.pending = list(pending)      # features the customer says a pending decision could add
        self.caps, self.ready = dict(cap), ready
        self.scope_final, self.decided_day = scope_final, decided_day
        self.approved = approved          # an approved limit above the willing amount (Juniper)

    def cap(self, day):
        return self.caps[max(d for d in self.caps if d <= day)]

    def final(self, day):
        return self.scope_final or (self.decided_day is not None and day >= self.decided_day)

    def needs_on(self, day):
        return self.needs_by_day[max(d for d in self.needs_by_day if d <= day)]

    @property
    def needs(self):
        return self.needs_by_day[1]

    def fits(self, product, day):
        return (set(self.needs_on(day)) <= set(product["features"])
                and set(self.integrations) <= set(product["integrations"]))


class Campaign:
    """Just enough of ctx for the tools, with scripted customers answering."""

    def __init__(self, customers, day=1, minutes=MINUTES_PER_DAY):
        self.day, self.minutes, self.per_day = day, minutes, minutes
        self.state = {"leads": {}, "price_changes": [dict(CHANGE_DAY7)]}
        self.customers = {c.cid: c for c in customers}
        self.accounts, self.quotes, self.log, self.missed = {}, {}, [], 0
        for c in customers:
            self.accounts[c.cid] = {
                "customer": {"id": c.cid, "name": f"Owner {c.cid}", "business": f"Business {c.cid}",
                             "background": c.background},
                "status": "prospect",
                "contact": {"eligible": True, "reason": None, "available_day": 1,
                            "last_contact_day": None, "last_decision": None},
                "last_message": None, "pending_callbacks": [], "quotes": [], "orders": []}

    # ctx surface -------------------------------------------------------------------------
    def get_status(self):
        return {"day": self.day, "campaign_days": CAMPAIGN_DAYS, "minutes_remaining_today": self.minutes,
                "minutes_per_exchange": MINUTES_PER_EXCHANGE, "cost_per_exchange_cents": 50}

    def policy(self):
        return STANDARD if self.day >= CHANGE_DAY7["effective_day"] else LAUNCH

    def get_catalog(self):
        return {"day": self.day, "current_pricing_policy": dict(self.policy()), "products": PRODUCTS}

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

    def call(self, name, **kwargs):
        assert name == "schedule_callback", name
        account = self.accounts[kwargs["customer_id"]]
        account["pending_callbacks"].append({"id": f"cb-{kwargs['customer_id']}-{kwargs['day']}",
                                             "day": kwargs["day"], "reason": kwargs["reason"]})
        return {"ok": True}

    def create_quote(self, customer_id, product_id, monthly_price_cents):
        quote = {"id": f"Q{len(self.quotes) + 1:04d}", "customer_id": customer_id, "product_id": product_id,
                 "monthly_price_cents": monthly_price_cents, "issued_day": None, "expires_day": None}
        self.quotes[quote["id"]] = quote
        self.accounts[customer_id]["quotes"].append(quote)
        return quote

    def _spend(self, customer_id):
        assert self.minutes >= MINUTES_PER_EXCHANGE, "contacted with no minutes left"
        self.minutes -= MINUTES_PER_EXCHANGE
        account = self.accounts[customer_id]
        account["contact"]["last_contact_day"] = self.day
        account["pending_callbacks"] = [cb for cb in account["pending_callbacks"] if cb["day"] > self.day]

    def present_quote(self, quote_id, message):
        quote = self.quotes[quote_id]
        cid = quote["customer_id"]
        self._spend(cid)
        if quote["issued_day"] is None:
            quote["issued_day"] = self.day
            quote["expires_day"] = self.day + self.policy()["quote_valid_days"]
        customer = self.customers[cid]
        product = next(p for p in PRODUCTS if p["id"] == quote["product_id"])
        price, cap = quote["monthly_price_cents"], customer.cap(self.day)
        self.log.append({"day": self.day, "cid": cid, "kind": "offer", "quote": quote_id,
                         "plan": product["id"], "price": price})
        if not customer.fits(product, self.day):
            return self._reply(cid, "decline", "We need features this plan does not include.")
        if price > cap:
            return self._reply(cid, "decline", f"For this offer, I would pay at most {money(cap)} per month.")
        if self.day < customer.ready:
            return self._reply(cid, "continue", f"I can authorize a purchase from day {customer.ready} onward.")
        account = self.accounts[cid]
        account["status"] = "purchased"
        order = {"quote": quote}
        account["orders"].append(order)
        self.log[-1]["accepted"] = True
        result = self._reply(cid, "accept", f"I accept quote {quote_id} at {money(price)} per month.")
        result["order"] = order
        return result

    def talk_to_customer(self, customer_id, message):
        self._spend(customer_id)
        customer = self.customers[customer_id]
        self.log.append({"day": self.day, "cid": customer_id, "kind": "message", "text": message})
        return self._reply(customer_id, "continue",
                           f"Our approved account limit is now {money(customer.cap(self.day))} per month.")

    def _reply(self, cid, decision, text):
        self.accounts[cid]["last_message"] = {"role": "customer"}
        self.log[-1]["decision"] = decision
        return {"decision": decision, "reply": {"id": "r", "day": self.day, "role": "customer", "text": text}}

    # the model -----------------------------------------------------------------------------
    def record(self, cid):
        """What the model writes with record_lead after a reply: what the customer has said so far."""
        customer = self.customers[cid]
        cap = customer.cap(self.day)
        return tools.record_lead(
            self, customer_id=cid, needs=customer.needs_on(self.day), nice_to_haves=[],
            integrations=customer.integrations,
            budget_cents=cap, max_budget_cents=customer.approved or cap, ready_day=customer.ready,
            scope_final=customer.final(self.day),
            pending_features=[] if customer.final(self.day) else customer.pending,
            open_question="", unsellable_reason="", notes="")

    def discover(self, *cids):
        """First contacts already made on this day: every account is marked reached before any reply
        is recorded, so record_lead never chains a first contact."""
        for cid in cids:
            self.accounts[cid]["contact"]["last_contact_day"] = self.day
            self.accounts[cid]["last_message"] = {"role": "customer"}
        for cid in cids:
            self.record(cid)

    def act(self, do):
        match = re.match(r"offer (\S+) \(product_id='(\w+)', price_cents=(\d+)\)", do)
        if match:
            cid = match.group(1)
            out = tools.offer(self, customer_id=cid, product_id=match.group(2),
                              price_cents=int(match.group(3)), message="As discussed.")
        else:
            match = re.match(r"send_message (\S+):", do)
            assert match, f"unexpected instruction: {do[:80]}"
            cid = match.group(1)
            out = tools.send_message(self, customer_id=cid, message=do)
        assert "error" not in out, out
        if self.accounts[cid]["status"] == "prospect":
            self.record(cid)
        return cid

    def run_day(self, limit=40):
        for _ in range(limit):
            if self.minutes < MINUTES_PER_EXCHANGE:
                break
            queue = tools._build_queue(self)
            if not queue["items"]:
                break
            item = queue["items"][0]
            assert item["kind"] != "first", "every account was discovered in these scenarios"
            self.act(item["do"])
        self.missed += sum(cb["day"] == self.day for a in self.accounts.values()
                           if a["status"] == "prospect" for cb in a["pending_callbacks"])
        self.day += 1
        self.minutes = self.per_day

    def run_until(self, last_day):
        while self.day <= last_day:
            self.run_day()

    # what happened ---------------------------------------------------------------------------
    def contacts(self, cid, day=None):
        return [e for e in self.log if e["cid"] == cid and (day is None or e["day"] == day)]

    def offers(self, cid, day=None):
        return [e for e in self.contacts(cid, day) if e["kind"] == "offer"]

    def bought(self, cid):
        return self.accounts[cid]["status"] == "purchased"

    def sale_price(self, cid):
        return self.accounts[cid]["orders"][0]["quote"]["monthly_price_cents"]


# --------------------------------------------------------------------------- customers from the run

def kestrel():            # Premium $92, decides day 9, budget fixed
    return Customer("lead127", ["scheduling", "reminders", "location_reporting"], {1: 9200}, ready=9)


def cobalt():             # Pro $53, decides day 8, budget fixed
    return Customer("lead369", ["scheduling", "reminders"], {1: 5300}, ready=8)


def anchor(day9_cap=5400):  # Pro, $48 until the day-9 funding decision
    return Customer("lead222", ["scheduling", "reminders"], {1: 4800, 9: day9_cap}, ready=9,
                    scope_final=False, decided_day=9)


def meadow():             # Premium needed, cap $80 forever, decides day 3
    return Customer("lead678", ["scheduling", "reminders", "location_reporting"], {1: 8000}, ready=3)


def juniper():            # Basic, willing $27, approved $30, decides day 1
    return Customer("lead956", ["scheduling"], {1: 2700}, ready=1, approved=3000)


def due_today(cid, day, cap=5800):
    """A customer whose callback falls on `day` and who buys Pro at list-ish price."""
    return Customer(cid, ["scheduling", "reminders"], {1: cap}, ready=day)


# --------------------------------------------------------------------------- A. locks

class LockOnce(unittest.TestCase):
    """Bug A: a lock already issued was offered again and again on day 6."""

    def test_issued_lock_is_not_offered_again_before_the_decision_day(self):
        c = Campaign([kestrel()], day=2)
        c.discover("lead127")
        c.run_until(8)
        offers = c.offers("lead127")
        self.assertEqual(len(offers), 1, offers)
        self.assertEqual((offers[0]["day"], offers[0]["price"]), (6, 9200), "lock on day 6 at the launch price")

    def test_assess_after_the_lock_waits_for_the_decision_day(self):
        lead = {"needs": ["scheduling", "reminders", "location_reporting"], "integrations": ["CalendarLink"],
                "budget_cents": 9200, "max_budget_cents": 9200, "ready_day": 9, "scope_final": True}
        quote = {"id": "Q25", "product_id": "premium", "monthly_price_cents": 9200, "issued_day": 6,
                 "expires_day": 9}
        launch, standard = {"discount": 15, "valid": 3}, {"discount": 5, "valid": 2}
        for day, base, changes, kind in ((6, launch, [CHANGE_DAY7], "later"), (7, standard, [], "later"),
                                         (8, standard, [], "later"), (9, standard, [], "offer")):
            with self.subTest(day=day):
                a = tools._assess(lead, PRODUCTS, day, CAMPAIGN_DAYS, base, changes, [quote])
                self.assertEqual((a["kind"], a.get("reuse_quote"), a["price_cents"]), (kind, "Q25", 9200))

    def test_locked_customer_buys_at_the_lock_on_the_decision_day(self):
        c = Campaign([kestrel()], day=2)
        c.discover("lead127")
        c.run_until(9)
        self.assertTrue(c.bought("lead127"))
        self.assertEqual(c.sale_price("lead127"), 9200)
        self.assertEqual(len(c.offers("lead127")), 2, "the lock, then its re-presentation on day 9")
        self.assertEqual({o["quote"] for o in c.offers("lead127")}, {c.offers("lead127")[0]["quote"]},
                         "day 9 re-presents the same quote")

    def test_day_6_has_room_for_every_lock(self):
        """The run's day 6: three due callbacks, then Kestrel, Cobalt and Anchor to lock."""
        due = [due_today(f"due{i}", 6) for i in range(3)]
        c = Campaign([kestrel(), cobalt(), anchor(), *due], day=2)
        c.discover("lead127", "lead369", "lead222", "due0", "due1", "due2")
        c.run_until(6)
        for cid in ("lead127", "lead369", "lead222"):
            with self.subTest(cid=cid):
                day6 = c.offers(cid, day=6)
                self.assertEqual(len(day6), 1, f"exactly one lock on day 6: {day6}")
        for cid in ("due0", "due1", "due2"):
            self.assertTrue(c.bought(cid))
        self.assertEqual(c.missed, 0)

    def test_settled_locks_are_not_contacted_between_lock_and_decision(self):
        c = Campaign([kestrel(), cobalt()], day=2)
        c.discover("lead127", "lead369")
        c.run_until(9)
        self.assertEqual([e["day"] for e in c.contacts("lead369")], [6, 8])
        self.assertEqual([e["day"] for e in c.contacts("lead127")], [6, 9])
        self.assertEqual((c.sale_price("lead369"), c.sale_price("lead127")), (5300, 9200))


class PendingDecisionLock(unittest.TestCase):
    """Anchor: locked at $51 under a $48 budget that hung on the day-9 funding decision."""

    def test_lock_on_day_6_then_confirm_then_lock_on_day_9(self):
        c = Campaign([anchor(day9_cap=5400)], day=2)
        c.discover("lead222")
        c.run_until(9)
        self.assertEqual([(o["day"], o["price"]) for o in c.offers("lead222", day=6)], [(6, 5100)])
        day9 = c.contacts("lead222", day=9)
        self.assertEqual([e["kind"] for e in day9], ["message", "offer"], "ask first, then quote")
        self.assertNotRegex(day9[0]["text"], r"price_cents", "the CONFIRM message carries no price")
        self.assertTrue(c.bought("lead222"))
        self.assertEqual(c.sale_price("lead222"), 5100, "$54 is below day 9's $57: the lock is the best price")

    def test_budget_risen_past_todays_floor_is_quoted_higher(self):
        c = Campaign([anchor(day9_cap=6000)], day=2)
        c.discover("lead222")
        c.run_until(9)
        self.assertTrue(c.bought("lead222"))
        self.assertEqual(c.sale_price("lead222"), 6000, "a fresh quote at the new budget, not the $51 lock")

    def test_budget_risen_between_lock_and_floor(self):
        c = Campaign([anchor(day9_cap=5600)], day=2)
        c.discover("lead222")
        c.run_until(9)
        self.assertEqual(c.sale_price("lead222"), 5100, "$56 is still below $57: keep the lock")

    def test_nothing_between_lock_and_decision_day(self):
        c = Campaign([anchor()], day=2)
        c.discover("lead222")
        c.run_until(8)
        self.assertEqual([e["day"] for e in c.contacts("lead222")], [6])


def oakbridge(day9_needs=("scheduling", "reminders", "location_reporting"), day9_cap=9000,
              pending=("location_reporting",)):
    """Pro at $60 until a day-9 review that may add location reporting (what the model records in
    pending_features); the review adds it with a $90 limit."""
    return Customer("lead582", ["scheduling", "reminders"], {1: 6000, 9: day9_cap}, ready=9, scope_final=False,
                    decided_day=9, later_needs={9: list(day9_needs)}, pending=pending)


class UpgradeLock(unittest.TestCase):
    """Oakbridge (lead582): sellable only through a Premium quote presented by day 6 (audit.json:
    early_quote presented_by day 6). Day 9's lowest Premium price is $95, above its new $90 limit."""

    def test_review_adds_reporting_sells_premium_at_the_lock(self):
        c = Campaign([oakbridge()], day=2)
        c.discover("lead582")
        c.run_until(9)
        day6 = c.offers("lead582", day=6)
        self.assertEqual([(o["plan"], o["price"]) for o in day6], [("premium", 8500)], "lock Premium on day 6")
        self.assertTrue(c.bought("lead582"))
        self.assertEqual(c.accounts["lead582"]["orders"][0]["quote"]["product_id"], "premium")
        self.assertEqual(c.sale_price("lead582"), 8500)

    def test_review_keeps_pro_sells_pro_at_its_budget(self):
        c = Campaign([oakbridge(day9_needs=("scheduling", "reminders"), day9_cap=6000)], day=2)
        c.discover("lead582")
        c.run_until(9)
        self.assertTrue(c.bought("lead582"))
        self.assertEqual((c.accounts["lead582"]["orders"][0]["quote"]["product_id"], c.sale_price("lead582")),
                         ("pro", 6000), "the unused Premium lock costs one message, not the sale")

    def test_lock_only_on_the_last_day_it_covers_the_decision(self):
        c = Campaign([oakbridge()], day=2)
        c.discover("lead582")
        c.run_until(8)
        self.assertEqual([e["day"] for e in c.contacts("lead582")], [6])

    def test_no_upgrade_lock_without_pending_features(self):
        c = Campaign([oakbridge(pending=())], day=2)
        c.discover("lead582")
        c.run_until(8)
        self.assertEqual(c.contacts("lead582"), [], "nothing may be added: no speculative Premium quote")

    def test_pending_feature_already_in_the_plan_adds_nothing(self):
        c = Campaign([oakbridge(pending=("reminders",))], day=2)
        c.discover("lead582")
        c.run_until(8)
        self.assertEqual(c.contacts("lead582"), [])

    def test_record_lead_drops_pending_features_once_scope_is_final(self):
        c = Campaign([oakbridge()], day=2)
        c.discover("lead582")
        self.assertEqual(c.state["leads"]["lead582"]["pending_features"], ["location_reporting"])
        tools.record_lead(c, customer_id="lead582", needs=["scheduling", "reminders"], nice_to_haves=[],
                          integrations=["CalendarLink"], budget_cents=6000, max_budget_cents=6000, ready_day=9,
                          scope_final=True, pending_features=["location_reporting"], open_question="",
                          unsellable_reason="", notes="")
        self.assertEqual(c.state["leads"]["lead582"]["pending_features"], [])

    def test_works_with_any_catalog_feature_names(self):
        """Held-out campaigns may name features differently: nothing is tied to this catalog's ids."""
        products = [
            {"id": "lite", "name": "Lite", "monthly_price_cents": 4000, "monthly_service_cost_cents": 1500,
             "features": ["booking"], "integrations": ["SyncHub"]},
            {"id": "plus", "name": "Plus", "monthly_price_cents": 8000, "monthly_service_cost_cents": 3000,
             "features": ["booking", "analytics"], "integrations": ["SyncHub"]},
        ]
        lead = {"needs": ["booking"], "integrations": ["SyncHub"], "ready_day": 9, "scope_final": False,
                "pending_features": ["analytics"]}
        account = {"customer": {"background": ""}, "quotes": []}
        launch, change = {"discount": 15, "valid": 3}, [{"effective_day": 7, "discount": 5, "valid": 2}]
        product, price, feature = tools._upgrade_lock(lead, account, products, 6, launch, change)
        self.assertEqual((product["id"], price, feature), ("plus", 6800, "analytics"))
        for day in (5, 7):
            with self.subTest(day=day):
                base = launch if day < 7 else {"discount": 5, "valid": 2}
                self.assertIsNone(tools._upgrade_lock(lead, account, products, day, base,
                                                      change if day < 7 else []))

    def test_no_upgrade_lock_when_scope_is_final(self):
        customer = oakbridge()
        customer.scope_final, customer.decided_day = True, None
        c = Campaign([customer], day=2)
        c.discover("lead582")
        c.run_until(8)
        self.assertEqual(c.contacts("lead582"), [])

    def test_no_upgrade_lock_for_a_decision_after_day_9(self):
        """A day-6 quote expires on day 9: nothing can be held for a day-10 decision."""
        customer = oakbridge()
        customer.ready, customer.decided_day = 10, 10
        customer.caps, customer.needs_by_day = {1: 6000, 10: 9000}, {1: ["scheduling", "reminders"],
                                                                      10: ["scheduling", "reminders", "location_reporting"]}
        c = Campaign([customer], day=2)
        c.discover("lead582")
        c.run_until(9)
        self.assertEqual(c.contacts("lead582"), [])

    def test_upgrade_lock_ranks_with_last_chances(self):
        due = [due_today(f"due{i}", 6) for i in range(3)]
        c = Campaign([oakbridge(), juniper(), *due], day=2)
        c.discover("lead582", "lead956", *[d.cid for d in due])
        c.day = 6
        order = [(i["cid"], i["kind"]) for i in tools._build_queue(c)["items"]]
        self.assertEqual({k for _, k in order[:3]}, {"offer"}, order)          # the 3 due callbacks
        self.assertEqual({cid for cid, _ in order[3:5]}, {"lead582", "lead956"}, order)


# --------------------------------------------------------------------------- B, C. budget below the floor

class BudgetBelowFloor(unittest.TestCase):
    """Bugs B and C: repeated offers to customers whose cap is below every legal price."""

    def assert_rules(self, c, cid):
        offers = c.offers(cid)
        days = [o["day"] for o in offers]
        self.assertLessEqual(len(offers), 2, f"final offer + one extra contact at most: {offers}")
        self.assertEqual(len(days), len(set(days)), f"never two offers on one day: {offers}")
        if len(offers) == 2:
            self.assertGreater(days[1], days[0])
        self.assertFalse(c.bought(cid))

    def test_meadow_gets_final_offer_and_one_extra_contact_only(self):
        """The run re-presented Meadow's declined $85 quote 3 times on day 8."""
        c = Campaign([meadow()], day=3)
        c.discover("lead678")
        c.run_until(CAMPAIGN_DAYS)
        self.assert_rules(c, "lead678")
        self.assertEqual(len(c.offers("lead678")), 2)
        self.assertTrue(c.state["leads"]["lead678"]["dropped"], "closed after the extra contact")

    def test_juniper_gets_one_offer_per_day_despite_a_higher_approved_limit(self):
        c = Campaign([juniper()], day=7)
        c.discover("lead956")
        c.run_until(CAMPAIGN_DAYS)
        self.assert_rules(c, "lead956")
        self.assertEqual(len(c.offers("lead956", day=7)), 1)
        self.assertEqual(len(c.offers("lead956")), 2, "the one extra contact on a later day is still made")

    def test_extra_contact_waits_for_spare_minutes(self):
        """A day full of paying work pushes the extra contact to a later day; it is not lost."""
        busy = [due_today(f"busy{i}", 8) for i in range(12)]
        c = Campaign([juniper(), *busy], day=7)
        c.discover("lead956", *[b.cid for b in busy])
        c.run_until(CAMPAIGN_DAYS)
        self.assert_rules(c, "lead956")
        extra = c.offers("lead956")[1]
        self.assertGreater(extra["day"], 8, "day 8 was full of callbacks")
        self.assertTrue(all(c.bought(b.cid) for b in busy))

    def test_final_offer_flag_set_by_an_at_most_reply(self):
        c = Campaign([juniper()], day=7)
        c.discover("lead956")
        out = tools.offer(c, customer_id="lead956", product_id="basic", price_cents=2850, message="Basic.")
        self.assertEqual(out["decision"], "decline")
        lead = c.state["leads"]["lead956"]
        self.assertTrue(lead["best_price_offered"])
        self.assertIn(out["quote"], lead["declined_quotes"])
        c.record("lead956")
        self.assertTrue(c.state["leads"]["lead956"]["dropped"])
        self.assertEqual(c.state["leads"]["lead956"]["close"], "budget")

    def test_declined_quotes_survive_record_lead_and_reset_on_a_new_budget(self):
        c = Campaign([juniper()], day=7)
        c.discover("lead956")
        tools.offer(c, customer_id="lead956", product_id="basic", price_cents=2850, message="Basic.")
        c.record("lead956")
        self.assertTrue(c.state["leads"]["lead956"]["declined_quotes"])
        c.customers["lead956"].caps[7] = 3000            # funding found: a new budget
        c.record("lead956")
        lead = c.state["leads"]["lead956"]
        self.assertEqual((lead["declined_quotes"], lead["best_price_offered"]), ([], False))

    def test_provisional_customer_may_see_a_declined_lock_again(self):
        """Anchor declined the $51 lock on day 6 and accepted it on day 9: a pending decision reopens it."""
        quote = {"id": "Q26", "product_id": "pro", "monthly_price_cents": 5100, "issued_day": 6, "expires_day": 9}
        lead = {"needs": ["scheduling", "reminders"], "integrations": ["CalendarLink"], "budget_cents": 4800,
                "max_budget_cents": 4800, "ready_day": 9, "best_price_offered": True, "declined_quotes": ["Q26"]}
        standard = {"discount": 5, "valid": 2}
        open_ = tools._assess(lead | {"scope_final": False}, PRODUCTS, 9, CAMPAIGN_DAYS, standard, [], [quote])
        self.assertEqual(open_.get("reuse_quote"), "Q26")
        settled = tools._assess(lead | {"scope_final": True}, PRODUCTS, 9, CAMPAIGN_DAYS, standard, [], [quote])
        self.assertEqual((settled["kind"], settled.get("close")), ("closed", "budget"))


# --------------------------------------------------------------------------- the minutes

class Minutes(unittest.TestCase):

    def test_every_scenario_respects_the_daily_minutes(self):
        """Campaign._spend asserts minutes are never overdrawn; run a mixed campaign at 60 and 120."""
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                people = [kestrel(), cobalt(), anchor(), meadow(), juniper(),
                          *[due_today(f"d{i}", 3 + i % 6) for i in range(10)]]
                c = Campaign(people, day=2, minutes=minutes)
                c.discover(*[p.cid for p in people])
                c.run_until(CAMPAIGN_DAYS)
                for cid in ("lead127", "lead369", "lead222"):
                    self.assertTrue(c.bought(cid), cid)
                BudgetBelowFloor.assert_rules(self, c, "lead678")
                # Reached on day 2, Juniper ($27) buys at the launch price: $25.50 was legal before day 7.
                self.assertTrue(c.bought("lead956"))
                self.assertEqual(c.sale_price("lead956"), 2700)
                self.assertEqual(c.missed, 0)


if __name__ == "__main__":
    unittest.main()
