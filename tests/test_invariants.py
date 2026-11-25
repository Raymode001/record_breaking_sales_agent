"""Black-box ordering invariants for the daily work queue (tools._build_queue).

Every expectation below comes from the written ordering specification, not from the
implementation. Each generated customer is classified by how it was BUILT (we know which
type we constructed), never by the kind/rank the queue assigns to it.

Tiers (lower ranks earlier):
  0 reply          an unrecorded customer reply                                rule 1
  2 due_cb         callback due today                                          rule 2
    reask          decision day, must-have unavailable, decision was pending   rule 2
    locked_due     callback due today on a still-valid launch quote (days 7-9) rule 2
  3 lock           day 6, budget fits only the launch price, decides day 7-9   rule 3a
    prov_lock      day 6, provisional, budget below every price, decides 7-9   rule 3a
    last_chance    day 6, affordable today, not tomorrow (decision day passed) rule 3b
  4 decision       decision day today, in budget, also affordable tomorrow     rule 4
  5 waitable       decision day passed, affordable today and tomorrow          rule 5
  6 final          scope final, budget below every legal price                 rule 6a
    persuade       closed on budget after declining the final offer earlier    rule 6b
  - early          decision day later and no lock can help: must not appear    rule 7
  - declined_today declined the final offer today: no offer today              rule 8
  - persuaded      the one extra contact already used: must not appear         rule 6b

Run from the project root:  python -m unittest discover -s tests -p "test_invariants.py" -v
"""

import math
import random
import re
import unittest

from test_failures import FakeCtx, lead, queue, items_for, PRODUCTS, tools  # noqa: F401

MINUTES_PER_EXCHANGE = 5
LAST_DAY = 12
PLAN = {p["id"]: p for p in PRODUCTS}
BASIC_NEEDS = {"needs": ["scheduling"], "integrations": ["CalendarLink"]}
PRO_NEEDS = {"needs": ["scheduling", "reminders"], "integrations": ["CalendarLink"]}
PREMIUM_NEEDS = {"needs": ["scheduling", "reminders", "location_reporting"], "integrations": ["CalendarLink"]}
NEEDS_BY_PLAN = {"basic": BASIC_NEEDS, "pro": PRO_NEEDS, "premium": PREMIUM_NEEDS}

TIER = {"reply": 0, "due_cb": 2, "reask": 2, "locked_due": 2,
        "lock": 3, "prov_lock": 3, "last_chance": 3,
        "decision": 4, "waitable": 5, "final": 6, "persuade": 6}
MUST_NOT_APPEAR = ("early", "persuaded")
SEEDS = range(600)


# --------------------------------------------------------------------------- pricing oracle

def discount_on(day):
    return 15 if day <= 6 else 5


def floor_price(plan, day):
    """Lowest legal price of a plan on a day (list x (100-d)/100 rounded up, never below cost)."""
    p = PLAN[plan]
    return max(math.ceil(p["monthly_price_cents"] * (100 - discount_on(day)) / 100),
               p["monthly_service_cost_cents"])


def money(cents):
    return f"${cents / 100:.2f}"


def parse_offer(do):
    """(plan, price) from an offer `do`, or None for a message/record item."""
    m = re.match(r"offer \S+ \(product_id='(\w+)', price_cents=(\d+)\)", do)
    return (m.group(1), int(m.group(2))) if m else None


# --------------------------------------------------------------------------- customer builders

def fixed(budget, ready, plan="basic", **extra):
    base = dict(NEEDS_BY_PLAN[plan])
    base.update(budget_cents=budget, max_budget_cents=budget, ready_day=ready, scope_final=True)
    base.update(extra)
    return lead(**base)


class Customer:
    def __init__(self, cid, ctype, record, info, contacted=None, callback_day=None, quotes=(),
                 last_role=None, touched=False):
        self.cid, self.ctype, self.record, self.info = cid, ctype, record, info
        self.contacted, self.callback_day, self.quotes = contacted, callback_day, list(quotes)
        self.last_role, self.touched = last_role, touched

    def describe(self):
        bits = [f"{self.cid}[{self.ctype}]"]
        r = self.record
        if r is None:
            bits.append("no lead recorded, last message from customer")
        else:
            bits.append("needs=" + "+".join(r["needs"]))
            if r.get("budget_cents") is not None:
                bits.append(f"budget={money(r['budget_cents'])}")
            bits.append(f"ready_day={r.get('ready_day')}")
            bits.append(f"scope_final={r.get('scope_final')}")
            if r.get("dropped"):
                bits.append(f"closed={r.get('close')} persuasions={r.get('persuasions')}")
        bits.append(f"last_contact={self.contacted}")
        if self.callback_day:
            bits.append(f"callback_day={self.callback_day}")
        for q in self.quotes:
            bits.append(f"quote {q['id']} {q['product_id']} {money(q['monthly_price_cents'])} "
                        f"days {q['issued_day']}-{q['expires_day']}")
        if self.touched:
            bits.append("contacted today (declined)")
        if self.info:
            bits.append(f"({self.info})")
        return " ".join(bits)


def build_ctx(day, minutes, customers):
    ctx = FakeCtx(day=day, minutes=minutes)
    for c in customers:
        ctx.add(c.cid, c.record, contacted=c.contacted, callback_day=c.callback_day,
                quotes=c.quotes, last_role=c.last_role)
        if c.touched:
            ctx.touched_today(c.cid, 1)
    return ctx


def earlier(rng, day, low=1):
    """A contact day before today (day 1 has no earlier day, use 1)."""
    return rng.randint(low, day - 1) if day - 1 >= low else max(1, min(low, day))


# Each maker returns a Customer or None when the type cannot exist on that day.

def make_reply(rng, day, cid):
    return Customer(cid, "reply", None, "", contacted=day, last_role="customer")


def make_due_cb(rng, day, cid):
    plan = rng.choice(["basic", "pro"])
    budget = rng.randint(floor_price(plan, day), PLAN[plan]["monthly_price_cents"])
    ready = rng.randint(max(1, day - 2), day)
    return Customer(cid, "due_cb", fixed(budget, ready, plan), f"fits {plan}",
                    contacted=earlier(rng, day), callback_day=day)


def make_reask(rng, day, cid):
    rec = lead(needs=["scheduling", "inventory_tracking"], integrations=["CalendarLink"],
               budget_cents=9000, max_budget_cents=9000, ready_day=day, scope_final=False)
    return Customer(cid, "reask", rec, "must-have unavailable, decision pending today",
                    contacted=earlier(rng, day))


def make_locked_due(rng, day, cid):
    if day not in (7, 8, 9):
        return None
    price = rng.randint(floor_price("pro", 6), floor_price("pro", day) - 1)
    quote = {"id": f"Q-{cid}", "product_id": "pro", "monthly_price_cents": price,
             "issued_day": 6, "expires_day": 9}
    return Customer(cid, "locked_due", fixed(price, day, "pro"), "budget below today's floor, holds a lock",
                    contacted=6, callback_day=day, quotes=[quote])


def make_lock(rng, day, cid):
    if day != 6:
        return None
    plan = rng.choice(["basic", "pro"])
    budget = rng.randint(floor_price(plan, 6), floor_price(plan, 7) - 1)
    return Customer(cid, "lock", fixed(budget, rng.randint(7, 9), plan),
                    f"fits {plan} only at the launch price", contacted=earlier(rng, day))


def make_prov_lock(rng, day, cid):
    if day != 6:
        return None
    budget = rng.randint(3500, floor_price("pro", 6) - 1)
    rec = fixed(budget, rng.randint(7, 9), "pro", scope_final=False)
    return Customer(cid, "prov_lock", rec, "provisional, below every price, lockable today",
                    contacted=earlier(rng, day))


def make_last_chance(rng, day, cid):
    if day != 6:
        return None
    plan = rng.choice(["basic", "pro"])
    budget = rng.randint(floor_price(plan, 6), floor_price(plan, 7) - 1)
    return Customer(cid, "last_chance", fixed(budget, rng.randint(1, 5), plan),
                    f"{plan} affordable today ({money(floor_price(plan, 6))}) not tomorrow "
                    f"({money(floor_price(plan, 7))})", contacted=earlier(rng, day))


def make_decision(rng, day, cid):
    plan = rng.choice(["basic", "pro", "premium"])
    low = max(floor_price(plan, day), floor_price(plan, min(day + 1, LAST_DAY)))
    budget = rng.randint(low, PLAN[plan]["monthly_price_cents"])
    return Customer(cid, "decision", fixed(budget, day, plan), f"decision day, fits {plan}",
                    contacted=earlier(rng, day))


def make_waitable(rng, day, cid):
    if day < 2 or day >= LAST_DAY:
        return None
    plan = rng.choice(["basic", "pro"])
    low = max(floor_price(plan, day), floor_price(plan, day + 1))
    budget = rng.randint(low, PLAN[plan]["monthly_price_cents"])
    return Customer(cid, "waitable", fixed(budget, rng.randint(1, day - 1), plan),
                    f"fits {plan} today and tomorrow", contacted=earlier(rng, day))


def make_final(rng, day, cid):
    plan = rng.choice(["basic", "pro"])
    low = 2000 if plan == "basic" else 3000
    budget = rng.randint(low, floor_price(plan, day) - 1)
    return Customer(cid, "final", fixed(budget, rng.randint(1, day), plan),
                    f"budget below every legal {plan}+ price ({money(floor_price(plan, day))})",
                    contacted=earlier(rng, day))


def _closed(budget, ready, **extra):
    return fixed(budget, ready, "basic", plan_caps={"basic": budget}, best_price_offered=True,
                 dropped=True, close="budget", **extra)


def make_persuade(rng, day, cid):
    if day < 2:
        return None
    budget = rng.randint(2000, floor_price("basic", day) - 1)
    declined = rng.randint(1, day - 1)
    return Customer(cid, "persuade", _closed(budget, rng.randint(1, declined)),
                    f"declined the final offer on day {declined}", contacted=declined)


def make_persuaded(rng, day, cid):
    if day < 3:
        return None
    budget = rng.randint(2000, floor_price("basic", day) - 1)
    used = rng.randint(2, day - 1)
    rec = _closed(budget, 1, persuasions=1, last_persuade_day=used)
    return Customer(cid, "persuaded", rec, f"extra contact used on day {used}", contacted=used)


def make_declined_today(rng, day, cid):
    budget = rng.randint(2000, floor_price("basic", day) - 1)
    return Customer(cid, "declined_today", _closed(budget, rng.randint(1, day)),
                    "declined the final offer today", contacted=day, touched=True)


def make_early(rng, day, cid):
    if day >= LAST_DAY:
        return None
    ready = rng.randint(day + 1, LAST_DAY)
    plan = rng.choice(["basic", "pro"])
    # Affordable at the decision day's own lowest price: a lock gains nothing.
    budget = rng.randint(max(floor_price(plan, ready), floor_price(plan, day)), PLAN[plan]["monthly_price_cents"])
    return Customer(cid, "early", fixed(budget, ready, plan),
                    f"fits {plan} on its decision day, no lock needed", contacted=earlier(rng, day))


MAKERS = {"reply": make_reply, "due_cb": make_due_cb, "reask": make_reask, "locked_due": make_locked_due,
          "lock": make_lock, "prov_lock": make_prov_lock, "last_chance": make_last_chance,
          "decision": make_decision, "waitable": make_waitable, "final": make_final,
          "persuade": make_persuade, "persuaded": make_persuaded, "declined_today": make_declined_today,
          "early": make_early}


def generate(seed):
    rng = random.Random(seed)
    day = 6 if rng.random() < 0.35 else rng.randint(1, LAST_DAY)
    minutes = rng.choice([60, 60, 120, 120, 10, 15, 20])
    customers = []
    for ctype in rng.sample(list(MAKERS), rng.randint(3, len(MAKERS))):
        for _ in range(rng.randint(1, 3)):
            c = MAKERS[ctype](rng, day, f"{ctype}{len(customers)}")
            if c is not None:
                customers.append(c)
    return day, minutes, customers


# --------------------------------------------------------------------------- invariant checks

def message_budget(minutes):
    return minutes // MINUTES_PER_EXCHANGE


def is_record(item):
    return item["kind"] == "record" or item["do"].startswith("record_lead")


def tier_order_violation(items, by_cid, tier):
    """First item of a tier above `tier` that comes before an item of exactly `tier`."""
    seen_lower = None
    for idx, item in enumerate(items):
        c = by_cid.get(item["cid"])
        t = TIER.get(c.ctype) if c else None
        if t is None:
            continue
        if t > tier and seen_lower is None:
            seen_lower = (idx, item)
        elif t == tier and seen_lower is not None:
            j, low = seen_lower
            return (f"#{idx + 1} {item['cid']} [{c.ctype}, rule-{tier} work] is listed after "
                    f"#{j + 1} {low['cid']} [{by_cid[low['cid']].ctype}, rule-{TIER[by_cid[low['cid']].ctype]}]")
    return None


def check(rule, day, minutes, customers, q):
    """Return a violation string for `rule`, or None."""
    items = q["items"]
    by_cid = {c.cid: c for c in customers}
    n = message_budget(minutes)

    if rule == 1:
        for c in customers:
            if c.ctype == "reply" and not any(is_record(i) for i in items_for(q, c.cid)):
                return f"{c.cid} has an unrecorded reply but no record item"
        first_other = next((k for k, i in enumerate(items) if not is_record(i)), len(items))
        late = [k for k, i in enumerate(items) if is_record(i) and k > first_other]
        if late:
            return f"record item #{late[0] + 1} {items[late[0]]['cid']} comes after #{first_other + 1}"
        return None

    if rule in (2, 3, 4, 5):
        bad = tier_order_violation(items, by_cid, rule)
        if bad:
            return bad
        if rule == 4:
            seq = []
            for k, item in enumerate(items):
                c = by_cid.get(item["cid"])
                off = parse_offer(item["do"])
                if c and c.ctype == "decision" and off:
                    seq.append((k, item["cid"], off[1] - PLAN[off[0]]["monthly_service_cost_cents"]))
            for (k1, a, p1), (k2, b, p2) in zip(seq, seq[1:]):
                if p2 > p1:
                    return (f"decision-day offers out of profit order: #{k1 + 1} {a} profit {money(p1)} "
                            f"before #{k2 + 1} {b} profit {money(p2)}")
        return None

    if rule == 6:
        for c in customers:
            if c.ctype == "persuaded" and items_for(q, c.cid):
                return f"{c.cid} already had its one extra contact but is listed again"
        last_other = max((k for k, i in enumerate(items)
                          if by_cid.get(i["cid"]) and TIER.get(by_cid[i["cid"]].ctype, 6) < 6), default=-1)
        for k, item in enumerate(items[:max(last_other, 0)]):
            c = by_cid.get(item["cid"])
            if c and TIER.get(c.ctype) == 6:
                other = items[last_other]
                return (f"#{k + 1} {c.cid} [{c.ctype}] is listed before #{last_other + 1} "
                        f"{other['cid']} [{by_cid[other['cid']].ctype}]")
        return None

    if rule == 7:
        for c in customers:
            if c.ctype == "early" and items_for(q, c.cid):
                return f"{c.cid} is contacted before its decision day {c.record['ready_day']} without a lock"
        return None

    if rule == 8:
        for c in customers:
            if c.ctype == "declined_today":
                offers = [i for i in items_for(q, c.cid) if parse_offer(i["do"])]
                if offers:
                    return f"{c.cid} declined the final offer today but gets another offer: {offers[0]['text']}"
        return None

    if rule == 9:
        for item in items:
            off = parse_offer(item["do"])
            if not off:
                continue
            plan, price = off
            low, high = floor_price(plan, day), PLAN[plan]["monthly_price_cents"]
            if low <= price <= high:
                continue
            m = re.search(r"re-presents ([^\s)]+)\)", item["do"])
            c = by_cid.get(item["cid"])
            if m and c:
                quote = next((x for x in c.quotes if x["id"] == m.group(1)), None)
                if (quote and quote["product_id"] == plan and quote["monthly_price_cents"] == price
                        and quote["issued_day"] <= day <= quote["expires_day"]):
                    continue
            return (f"{item['cid']} offered {plan} at {money(price)}; legal today is "
                    f"{money(low)}..{money(high)} and no valid quote at that price is re-presented")
        return None

    if rule == 10:
        due = [c.cid for c in customers if TIER.get(c.ctype) == 2]
        for cid in due:
            if not items_for(q, cid):
                return f"{cid} [{by_cid[cid].ctype}] is a promised contact due today but is not listed"
        msgs = [i for i in items if not is_record(i)]
        head = [i["cid"] for i in msgs[:n]]
        inside = sum(cid in head for cid in set(due))
        need = min(len(set(due)), n)
        if inside < need:
            missing = [cid for cid in due if cid not in head]
            return (f"{need} promised contacts due, only {inside} inside today's {n} messages; "
                    f"outside: {missing}")
        return None

    if rule == "listed":
        for c in customers:
            if TIER.get(c.ctype) == 3 and not items_for(q, c.cid):
                return f"{c.cid} [{c.ctype}] is a last chance but is not listed"
        return None
    raise ValueError(rule)


def run_rule(rule, day, minutes, customers):
    return check(rule, day, minutes, customers, queue(build_ctx(day, minutes, customers)))


def signature_of(bad):
    """The kinds of customers a violation names, so shrinking keeps the same kind of failure."""
    return tuple(re.findall(r"\[(\w+)", bad)) or (bad.split()[0].rstrip("0123456789"),)


def shrink(rule, day, minutes, customers):
    """Greedily drop customers while the same kind of violation of the rule persists."""
    current = list(customers)
    target = signature_of(run_rule(rule, day, minutes, current))
    changed = True
    while changed:
        changed = False
        for k in range(len(current)):
            trial = current[:k] + current[k + 1:]
            bad = run_rule(rule, day, minutes, trial) if trial else None
            if bad and signature_of(bad) == target:
                current, changed = trial, True
                break
    return current


def explain(rule, seed, day, minutes, customers):
    q = queue(build_ctx(day, minutes, customers))
    lines = [f"rule {rule} violated (seed {seed}): {check(rule, day, minutes, customers, q)}",
             f"day {day}, {minutes} minutes ({message_budget(minutes)} messages left)",
             "customers:"]
    lines += ["  " + c.describe() for c in customers]
    lines.append("first items:")
    lines += [f"  {k + 1}. {i['text']}" for k, i in enumerate(q["items"][:8])]
    if len(q["items"]) > 8:
        lines.append(f"  ... {len(q['items']) - 8} more")
    return "\n".join(lines)


_SCENARIOS = None


def scenarios():
    global _SCENARIOS
    if _SCENARIOS is None:
        _SCENARIOS = []
        for seed in SEEDS:
            day, minutes, customers = generate(seed)
            _SCENARIOS.append((seed, day, minutes, customers, queue(build_ctx(day, minutes, customers))))
    return _SCENARIOS


class RandomizedInvariants(unittest.TestCase):
    """600 seeded days; each test checks one rule of the specification over all of them."""

    def assert_rule(self, rule):
        """Fail with one minimal counterexample per distinct kind of violation (up to 3)."""
        groups, total = {}, 0
        for seed, day, minutes, customers, q in scenarios():
            bad = check(rule, day, minutes, customers, q)
            if bad:
                total += 1
                groups.setdefault(signature_of(bad), (seed, day, minutes, customers))
        if groups:
            parts = []
            for seed, day, minutes, customers in list(groups.values())[:3]:
                parts.append(explain(rule, seed, day, minutes, shrink(rule, day, minutes, customers)))
            self.fail(f"{total} of {len(scenarios())} scenarios break rule {rule}; "
                      f"{len(groups)} kind(s) of violation. Minimal counterexample(s):\n\n"
                      + "\n\n".join(parts))

    def test_rule01_unrecorded_reply_first(self):
        self.assert_rule(1)

    def test_rule02_promised_contacts_before_other_contacts(self):
        self.assert_rule(2)

    def test_rule03_last_chances_before_decision_day_and_later_work(self):
        self.assert_rule(3)

    def test_rule03_last_chances_are_listed(self):
        self.assert_rule("listed")

    def test_rule04_decision_day_offers_before_waitable_and_by_profit(self):
        self.assert_rule(4)

    def test_rule05_waitable_offers_before_final_offers(self):
        self.assert_rule(5)

    def test_rule06_final_offers_and_extra_contact_last_and_once(self):
        self.assert_rule(6)

    def test_rule07_no_contact_before_decision_day_without_lock(self):
        self.assert_rule(7)

    def test_rule08_no_second_offer_on_the_day_of_a_decline(self):
        self.assert_rule(8)

    def test_rule09_offer_prices_are_legal_today(self):
        self.assert_rule(9)

    def test_rule10_promised_contacts_fit_in_todays_messages(self):
        self.assert_rule(10)

    def test_generator_covers_every_type(self):
        seen = {c.ctype for _, _, _, customers, _ in scenarios() for c in customers}
        self.assertEqual(seen, set(MAKERS), "every customer type appears in some scenario")


# --------------------------------------------------------------------------- fixed regression

class Day6Regression(unittest.TestCase):
    """Real run, day 6, 12 messages left: Juniper (Basic $27, launch floor $25.50, standard $28.50)
    must rank before the two final offers and inside the day's messages."""

    def build(self, minutes=60):
        customers = []
        for i in range(3):
            customers.append(Customer(f"due{i}", "due_cb", fixed(5700, 6, "pro"), "", contacted=1, callback_day=6))
        for i in range(3):
            customers.append(Customer(f"lock{i}", "lock", fixed(5100, 9, "pro"), "", contacted=2))
        customers.append(Customer("final35", "final", fixed(3500, 5, "pro"), "", contacted=2))
        customers.append(Customer("final48", "final", fixed(4800, 5, "pro"), "", contacted=2))
        customers.append(Customer("juniper", "last_chance", fixed(2700, 1, "basic"), "", contacted=2))
        return customers

    def test_juniper_before_final_offers_and_inside_the_day(self):
        for minutes in (60, 120):
            with self.subTest(minutes=minutes):
                customers = self.build(minutes)
                q = queue(build_ctx(6, minutes, customers))
                order = [i["cid"] for i in q["items"]]
                report = explain(3, "fixed", 6, minutes, customers)
                self.assertIn("juniper", order, report)
                j = order.index("juniper")
                for final in ("final35", "final48"):
                    self.assertIn(final, order, report)
                    self.assertLess(j, order.index(final), report)
                self.assertLess(j, message_budget(minutes), report)

    def test_final_offers_price_and_callbacks_inside_the_day(self):
        customers = self.build(60)
        q = queue(build_ctx(6, 60, customers))
        head = [i["cid"] for i in q["items"][:12]]
        for i in range(3):
            self.assertIn(f"due{i}", head)
        for final in ("final35", "final48"):
            offers = [parse_offer(i["do"]) for i in items_for(q, final)]
            self.assertIn(("pro", 5100), offers, f"{final}: final offer at the lowest legal Pro price")


if __name__ == "__main__":
    unittest.main()
