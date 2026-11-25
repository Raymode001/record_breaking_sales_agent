"""The agent's tools. The model can call exactly the @tool functions in this file.

Strategy these tools support (see prompt.py):
  1. Reach every account early. A first contact is one templated message (first_contact):
     the plans from the catalog, the answer to the concern in the account's CRM record
     (from company documents), and the discovery questions (needs, budget, approval day,
     whether the scope is final). First contacts rank right after time-critical work.
  2. Chaining: when the next task is a first contact, record_lead and work_queue send it
     themselves and return the reply, so discovery never depends on the model choosing
     to continue. The model's job is to record every reply with record_lead.
  3. record_lead does the arithmetic: which plans fit, the legal price under each pricing
     policy, the most profitable plan the budget allows (upsell), and the day to present
     a quote so it is still valid on the customer's decision day (price locks).
  4. Every result ends with do_next: the single next call to make.

Rules: tools reach the company only through ctx, never call a language model, never
read the homework files and never hard-code facts about specific customers. A tool
that sends a customer a message does so as its last step and returns the reply.
"""

import math
import re

from sales_sim import tool


DISCOVERY_TARGET_DAY = 4        # reach every account by this day (the day before a worse price change if sooner)
MAX_TOUCHES_DISCOVERY = 4       # messages per customer per day while accounts are still uncontacted
MAX_TOUCHES_AFTER = 8           # relaxed once every account has been reached
MAX_PERSUASIONS = 1             # after the final offer is declined: one more spare-time contact, then close
MAX_FIRST_CONTACT_DOCS = 4      # documents attached to a first contact
DOC_MATCH = 0.3                 # cosine similarity a document needs to be attached
SENTENCE_MATCH = 0.15           # cosine similarity a sentence needs to be quoted

STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it
its of on or our the their them they this to we what when which with you your will
would could should any there that than about into after before more most not no""".split())

# Documents that are not answers to customer questions (internal guidance, stories, old terms).
NOT_ANSWERS = ("product:", "policy:", "archive", "sales:", "case:", "partners:", "glossary", "release:", "faq")

# General product concerns a customer may raise (trigger words), and a BUNDLE of searches:
# each search attaches its best document, so related follow-up questions are answered at once.
TOPICS = [
    (("record", "vanish", "locked out", "leave", "leaving", "export"),
     ["records after cancellation retention", "export records csv trial", "cancel subscription account"]),
    (("trial", "refund", "cancel"), ["trial refund cancellation", "cancel subscription account"]),
    (("migrat", "import", "history", "move off", "previous years", "kept separately"),
     ["migration import records", "bookflow connector setup"]),
    (("secur", "compliance", "stored", "sensitive", "privacy", "questionnaire"),
     ["security data hosting encryption", "staff logins two-factor", "privacy data processing agreement"]),
    (("othersuite", "connector", "compatib"), ["othersuite integration connector", "supported integrations"]),
    (("deposit", "payment"), ["payments deposits"]),
    (("inventory", "work-order", "work order", "equipment"), ["product limits inventory work order"]),
    (("get started", "setup", "set up", "onboard", "helps us"), ["onboarding setup help", "support hours channels"]),
    (("per-tutor", "per user", "several staff", "account charges", "total account price", "group of"),
     ["billing account price users locations", "subscription prices quote terms charges"]),
    (("spanish", "language", "english"), ["reminder languages"]),
    (("sites", "locations", "venues", "network", "clinics", "studios", "compare"), ["location reporting dashboard"]),
]


# --------------------------------------------------------------------------- pricing helpers

def _money(cents):
    return f"${cents / 100:.2f}"


def _min_price(product, discount_percent):
    """Lowest legal price: list less the discount, rounded up, never below service cost."""
    discounted = (product["monthly_price_cents"] * (100 - discount_percent) + 99) // 100
    return max(product["monthly_service_cost_cents"], discounted)


def _policies(ctx, catalog, today):
    """Today's policy plus announced changes registered with register_price_change."""
    policy = catalog["current_pricing_policy"]
    base = {"discount": policy["max_discount_percent"], "valid": policy["quote_valid_days"]}
    changes = sorted((c for c in ctx.state.get("price_changes", []) if c["effective_day"] > today),
                     key=lambda c: c["effective_day"])
    return base, changes


def _policy_on(day, base, changes):
    current = base
    for change in changes:
        if change["effective_day"] <= day:
            current = {"discount": change["discount"], "valid": change["valid"]}
    return current


def _lower(items):
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _quote_view(quote, today, orders):
    accepted = any(order["quote"]["id"] == quote["id"] for order in orders)
    if accepted:
        state = "accepted"
    elif quote["issued_day"] is None:
        state = "draft"
    elif today > quote["expires_day"]:
        state = "expired"
    else:
        state = f"valid through day {quote['expires_day']}"
    return {"id": quote["id"], "product": quote["product_id"],
            "price": _money(quote["monthly_price_cents"]), "state": state}


def _valid_issued(quotes, today):
    return [q for q in quotes if q["issued_day"] is not None and today <= q["expires_day"]]


# --------------------------------------------------------------------------- lead assessment

def _best_price_day(product, ready, today, base, changes):
    """The lowest legal price whose quote is still valid on the ready day, and the day to present it."""
    candidates = []
    for day in range(today, ready + 1):
        policy = _policy_on(day, base, changes)
        if day + policy["valid"] >= ready:
            candidates.append((_min_price(product, policy["discount"]), -day))
    price, negative_day = min(candidates)
    return price, -negative_day


def _assess(lead, products, today, last_day, base, changes, quotes):
    """Turn what the customer said into a plan, price and day. kind is one of:
    closed (no sale possible), later (nothing today), reask (capability may change today),
    need_budget, offer, lock (offer today so the price holds to the decision day),
    best_price (budget below every legal price: one final offer at the lowest legal price)."""
    provisional = not lead.get("scope_final", True)
    ready = lead.get("ready_day")
    if ready is not None and ready > last_day:
        # They cannot approve inside the campaign: abandon, no follow-up and no final offer.
        return {"kind": "closed", "close": "window",
                "why": f"cannot approve before the campaign ends (day {last_day})"}
    ready = max(ready or today, today)

    def capability_block(why):
        # A missing capability: selling anyway would be refunded. Revisit only if a decision is pending.
        if provisional:
            kind = "reask" if ready <= today else "later"
            return {"kind": kind, "why": why + "; a pending decision may change this", "ready_day": ready}
        return {"kind": "closed", "close": "capability", "why": why}

    if lead.get("unsellable_reason"):
        return capability_block(lead["unsellable_reason"])
    needs, integrations = _lower(lead.get("needs", [])), _lower(lead.get("integrations", []))
    nice = _lower(lead.get("nice_to_haves", []))
    features_offered = {f.lower() for p in products for f in p["features"]}
    integrations_offered = {i.lower() for p in products for i in p["integrations"]}
    missing = sorted((needs - features_offered) | (integrations - integrations_offered))
    if missing:
        return capability_block(f"no plan offers: {', '.join(missing)} (check find_docs for a workaround first)")
    fits = [p for p in products
            if needs <= {f.lower() for f in p["features"]} and integrations <= {i.lower() for i in p["integrations"]}]
    if not fits:
        return capability_block("no single plan covers all needs")
    fits.sort(key=lambda p: p["monthly_price_cents"])
    cheapest = fits[0]
    budget, upper = lead.get("budget_cents"), lead.get("max_budget_cents")
    caps = lead.get("plan_caps", {})
    if budget is None and upper is None and not caps:
        return {"kind": "need_budget" if ready <= today else "later", "ready_day": ready,
                "plan": cheapest["id"], "price": _money(cheapest["monthly_price_cents"]),
                "price_cents": cheapest["monthly_price_cents"], "present_day": ready,
                "why": "budget unknown: offer the cheapest fitting plan at list price and use the reply"}

    def cap_for(product):
        """What the customer will pay for this plan: a cap learned from an offer reply wins;
        a plan adding a nice-to-have may use the approved limit; otherwise the budget."""
        if product["id"] in caps:
            return caps[product["id"]]
        adds_nice = nice & ({f.lower() for f in product["features"]} - needs)
        if adds_nice and upper:
            return max(upper, budget or 0)
        return budget if budget is not None else upper

    options = []
    for product in fits:
        cap = cap_for(product)
        if cap is None:
            continue
        price = min(product["monthly_price_cents"], cap)
        plan = None
        # Best: present on the ready day itself (one message, customer can accept at once).
        if _min_price(product, _policy_on(ready, base, changes)["discount"]) <= price:
            plan = {"present_day": ready, "accept_day": ready}
        else:
            # Otherwise lock a price earlier: the latest day whose quote is still valid on the ready day.
            for day in range(ready - 1, today - 1, -1):
                policy = _policy_on(day, base, changes)
                if _min_price(product, policy["discount"]) <= price and day + policy["valid"] >= ready:
                    plan = {"present_day": day, "accept_day": ready}
                    break
        # A quote already issued for this plan within the cap and valid on the ready day counts too:
        # it already holds any earlier lock, so it is never re-issued. Only a fresh quote on the ready
        # day itself (a budget that rose above that day's legal price) beats the cheaper lock.
        fresh_on_ready_day = plan is not None and plan["present_day"] == ready
        for quote in ([] if fresh_on_ready_day else _valid_issued(quotes, today)):
            if (quote["product_id"] == product["id"] and quote["monthly_price_cents"] <= cap
                    and quote["expires_day"] >= ready):
                plan = {"present_day": ready, "accept_day": ready, "reuse_quote": quote["id"]}
                price = quote["monthly_price_cents"]
                break
        if plan:
            options.append({"plan": product["id"], "price_cents": price,
                            "profit_cents": price - product["monthly_service_cost_cents"], **plan})

    if not options:
        # Budget below every legal price. A final offer at the lowest legal price may still be
        # accepted (budgets can move); make it once, timed to be valid on the decision day.
        lowest_today = {p["id"]: _min_price(p, base["discount"]) for p in fits}
        declined = set(lead.get("declined_quotes", []))
        for quote in _valid_issued(quotes, today):
            # Only a locked quote cheaper than anything legal today is worth re-presenting, and only
            # once: a quote they declined comes back only if a pending decision may have moved the budget.
            if quote["id"] in declined and not provisional:
                continue
            # A quote at today's lowest legal price is a lock held until the decision day, too.
            if (quote["product_id"] in lowest_today and quote["expires_day"] >= ready
                    and quote["monthly_price_cents"] <= lowest_today[quote["product_id"]]):
                kind = "offer" if ready <= today else "later"
                return {"kind": kind, "plan": quote["product_id"], "price": _money(quote["monthly_price_cents"]),
                        "price_cents": quote["monthly_price_cents"], "present_day": ready, "accept_day": ready,
                        "reuse_quote": quote["id"], "why": "re-present the lowest-price quote on the decision day"}
        known = min(c for c in (budget, upper, *caps.values()) if c is not None)
        if lead.get("best_price_offered") and not provisional:
            return {"kind": "closed", "close": "budget",
                    "why": f"declined our lowest legal price; budget {_money(known)}"}
        product = min(fits, key=lambda p: (_best_price_day(p, ready, today, base, changes)[0],
                                           p["monthly_price_cents"]))
        price, day = _best_price_day(product, ready, today, base, changes)
        result = {"plan": product["id"], "price": _money(price), "price_cents": price,
                  "present_day": day, "accept_day": ready, "final": True,
                  "why": f"budget {_money(known)} is below every legal price; one final offer at the lowest "
                         f"legal price {_money(price)}"}
        if day > today:
            result["kind"] = "later"
        elif day < ready:
            result["kind"] = "lock"
        else:
            result["kind"] = "best_price"
        return result

    best = max(options, key=lambda o: (o["profit_cents"], -o["present_day"]))
    fallback = next((o for o in options if o["plan"] == cheapest["id"]), None)
    result = {"plan": best["plan"], "price": _money(best["price_cents"]), "price_cents": best["price_cents"],
              "present_day": best["present_day"], "accept_day": best["accept_day"]}
    if best.get("reuse_quote"):
        result["reuse_quote"] = best["reuse_quote"]
    if fallback and fallback["plan"] != best["plan"]:
        result["upsell_over"] = {"plan": fallback["plan"], "price": _money(fallback["price_cents"]),
                                 "extra_profit": _money(best["profit_cents"] - fallback["profit_cents"])}
    if best["present_day"] > today:
        result["kind"] = "later"
        result["why"] = f"present on day {best['present_day']}"
    elif best["accept_day"] == today:
        result["kind"] = "offer"       # decision day: offer directly, even if the scope was provisional
    else:
        result["kind"] = "lock"
        result["why"] = f"lock the price today; the quote stays valid through decision day {best['accept_day']}"
    return result


def _fitting_plans(lead, products):
    """Plans covering every must-have feature and integration, cheapest first."""
    needs, integrations = _lower(lead.get("needs", [])), _lower(lead.get("integrations", []))
    fits = [p for p in products
            if needs <= {f.lower() for f in p["features"]} and integrations <= {i.lower() for i in p["integrations"]}]
    return sorted(fits, key=lambda p: p["monthly_price_cents"])


def _assess_customer(ctx, customer_id, catalog=None, status=None, account=None):
    status = status or ctx.get_status()
    catalog = catalog or ctx.get_catalog()
    lead = ctx.state.get("leads", {}).get(customer_id)
    if lead is None or lead.get("dropped") and not lead.get("needs"):
        return None
    account = account or ctx.read_customer(customer_id=customer_id)
    base, changes = _policies(ctx, catalog, status["day"])
    return _assess(lead, catalog["products"], status["day"], status["campaign_days"], base, changes, account["quotes"])


def _summary(assessment):
    """A compact recommendation for the model, never ending on a stop word."""
    if not assessment:
        return None
    keep = ("plan", "price", "present_day")     # short: every result is re-read on every later call
    out = {k: assessment[k] for k in keep if k in assessment}
    out["status"] = {"closed": "closed: no sale possible", "later": "nothing to do for this account today",
                     "reask": "decision day: re-ask needs and budget", "need_budget": "offer today (budget unknown)",
                     "offer": "offer today", "lock": "offer today to lock the price",
                     "best_price": "one final offer today at the lowest legal price"}[assessment["kind"]]
    return out


# --------------------------------------------------------------------------- the work queue

DAY_WORDS = re.compile(r"\bday (\d+)\b", re.I)
SOON = ("this week", "today", "minutes", "urgent", "asap", "now")
LATE = ("end of the month", "end of month", "next month", "late this month", "last day of the month",
        "next term", "next year", "later this", "contract ends")


def _urgency(background):
    """Order for first contact: near deadlines first, far ones last."""
    text = background.lower()
    days = [int(d) for d in DAY_WORDS.findall(text)]
    if any(word in text for word in SOON):
        return 0
    if days:
        return min(days)
    if any(word in text for word in LATE):
        return 99
    return 50


def _touch(ctx, customer_id, day):
    touches = ctx.state.setdefault("touches", {})
    if touches.get("day") != day:
        touches.clear()
        touches["day"] = day
    touches[customer_id] = touches.get(customer_id, 0) + 1


LONG_SHOT_TIER = 5     # final offers to fixed budgets below every legal price, and the extra contact


def _upgrade_lock(lead, account, products, today, base, changes):
    """A provisional customer whose pending decision may add a feature their current plan lacks
    (record_lead's pending_features, judged by the model from what they said):
    on the last day a quote presented today still covers their decision day at a price that will
    be gone by then, lock the plan with that feature at today's lowest legal price (Oakbridge:
    Premium $85 on day 6 for a day-9 review that added location reporting). Returns
    (product, price, feature) or None."""
    ready = lead.get("ready_day")
    if lead.get("scope_final", True) or ready is None or ready <= today or lead.get("unsellable_reason"):
        return None
    fits = _fitting_plans(lead, products)
    if not fits:
        return None
    current = set(fits[0]["features"])
    possible = set(lead.get("pending_features", []))
    now, tomorrow = _policy_on(today, base, changes), _policy_on(today + 1, base, changes)
    if today + now["valid"] < ready:
        return None                                  # a quote presented today would expire too early
    for product in sorted(products, key=lambda p: p["monthly_price_cents"]):
        added = set(product["features"]) - current
        if (not added or not current <= set(product["features"])
                or not set(i.lower() for i in lead.get("integrations", [])) <= {i.lower() for i in product["integrations"]}):
            continue
        feature = next((f for f in sorted(added) if f in possible), None)
        if not feature:
            continue
        price = _min_price(product, now["discount"])
        later_price = _min_price(product, _policy_on(ready, base, changes)["discount"])
        still_tomorrow = (today + 1 + tomorrow["valid"] >= ready
                          and _min_price(product, tomorrow["discount"]) <= price)
        held = any(q["product_id"] == product["id"] and q["issued_day"] is not None
                   and q["expires_day"] >= ready for q in account["quotes"])
        if later_price > price and not still_tomorrow and not held:
            return product, price, feature
    return None


def _build_queue(ctx):
    """Today's tasks in priority order: 0 record replies, 1 time-critical, 2 first contacts,
    3 (once every account is reached) answers and persuasion for customers not yet converted."""
    status, catalog = ctx.get_status(), ctx.get_catalog()
    today, last_day = status["day"], status["campaign_days"]
    leads = ctx.state.get("leads", {})
    touches = ctx.state.get("touches", {})
    touched_today = touches if touches.get("day") == today else {}
    customers = ctx.list_customers()
    prospects = [c for c in customers if c["status"] == "prospect" and not leads.get(c["id"], {}).get("dropped")]
    uncontacted = [c for c in prospects if c["contact"]["last_contact_day"] is None]
    everyone_reached = not uncontacted
    touch_limit = MAX_TOUCHES_AFTER if everyone_reached else MAX_TOUCHES_DISCOVERY
    base, changes = _policies(ctx, catalog, today)
    items, later = [], []

    for customer in prospects:
        cid, contact = customer["id"], customer["contact"]
        if contact["last_contact_day"] is None:
            continue
        if contact["reason"] == "call_ended" or touched_today.get(cid, 0) >= touch_limit:
            later.append((today + 1, cid))
            continue
        account = ctx.read_customer(customer_id=cid)
        due = [cb for cb in account["pending_callbacks"] if cb["day"] <= today]
        tag = f"{cid} ({customer['business']})"
        if cid not in leads:
            last = account["last_message"]
            if last and last["role"] == "customer":
                items.append({"p": 0, "kind": "record", "cid": cid,
                              "text": f"RECORD {tag}: last reply not recorded yet",
                              "do": f"record_lead for {cid} from their last reply"})
            continue
        lead = leads[cid]
        a = _assess(lead, catalog["products"], today, last_day, base, changes, account["quotes"])
        kind = a["kind"]
        question = lead.get("open_question", "")
        # A pending question waits for the decision day: the decision-day offer answers it first.
        # Before that day the customer is contacted only to lock a price.
        if kind in ("offer", "lock", "best_price", "need_budget"):
            label = {"offer": "CALLBACK DUE, OFFER" if due else "OFFER", "lock": "LOCK PRICE",
                     "best_price": "FINAL OFFER (lowest legal price)", "need_budget": "OFFER (budget unknown)"}[kind]
            reuse = f" (re-presents {a['reuse_quote']})" if a.get("reuse_quote") else ""
            # Urgent: a callback or decision day today, a price lock, or a price that waiting would lose.
            # Otherwise the offer can wait behind today's reserved first contacts.
            ready_day = lead.get("ready_day")
            tomorrow = (_assess(lead, catalog["products"], today + 1, last_day, base, changes, account["quotes"])
                        if today < last_day else {})
            can_wait = (not due and kind != "lock" and (ready_day is None or ready_day < today)
                        and tomorrow.get("plan") == a.get("plan") and tomorrow.get("price_cents") == a.get("price_cents"))
            # Order: due callback; then last chances (the last day a lock holds its price, or a sale
            # affordable today but not tomorrow); then decision-day offers by profit; then the rest.
            # A final offer to a fixed budget below every legal price is a long shot: it goes last.
            provisional = not lead.get("scope_final", True)
            long_shot = bool(a.get("final")) and not provisional
            last_lock = kind == "lock" and (tomorrow.get("kind") not in ("lock", "offer")
                                            or tomorrow.get("price_cents") != a.get("price_cents"))
            lost_tomorrow = (today < last_day and not long_shot and kind in ("offer", "need_budget")
                             and (tomorrow.get("plan"), tomorrow.get("price_cents"))
                             != (a.get("plan"), a.get("price_cents")))
            decision_day = kind in ("offer", "best_price") and (ready_day is None or ready_day <= today)
            rank = 0 if due else 1 if (last_lock or lost_tomorrow) else 2 if decision_day else 3
            tier = 1 if due else LONG_SHOT_TIER if long_shot else 3 if can_wait else 1
            product = next(p for p in catalog["products"] if p["id"] == a["plan"])
            # A locked price on the decision day of a customer whose budget hung on a pending decision:
            # ask first, so a budget that rose above today's legal price is quoted higher, not at the lock.
            locked = next((q for q in account["quotes"] if q["id"] == a.get("reuse_quote")
                           and q["issued_day"] is not None and q["issued_day"] < today), None)
            if (locked and decision_day and not lead.get("scope_final", True)
                    and touched_today.get(cid, 0) == 0):
                items.append({"p": 1, "kind": "confirm", "cid": cid, "due": bool(due), "rank": rank,
                              "profit": a["price_cents"] - product["monthly_service_cost_cents"],
                              "text": f"CONFIRM {tag}: decision day; locked {a['plan']} at {a['price']} "
                                      f"({locked['id']}); ask what changed before quoting",
                              "do": (f"send_message {cid}: their decision is due today. "
                                     + (f"First answer \"{question[:120]}\" by {_answer_for(ctx, question, lead)} "
                                        if question else "")
                                     + "Ask in one message whether the decision changed their must-haves or "
                                       "budget, and what their approved monthly limit is now. Do not quote a "
                                       "price in this message. Then record_lead with the answer.")})
                continue
            items.append({"p": tier, "kind": kind, "cid": cid, "due": bool(due), "rank": rank,
                          "profit": a["price_cents"] - product["monthly_service_cost_cents"],
                          "text": f"{label} {tag}: {a['plan']} at {a['price']}{reuse}",
                          "do": _offer_instruction(ctx, cid, a, question, reuse, lead)})
        elif kind == "reask":
            # A decision day is a promised callback: same ranking as one (Spruce missed its day 6).
            items.append({"p": 1, "kind": kind, "cid": cid, "due": True, "rank": 0,
                          "text": f"DECISION DAY {tag}: a must-have was unavailable but their decision was pending",
                          "do": f"send_message {cid}: ask in one message whether the decision changed their must-haves "
                                f"and budget"})
        elif due:
            items.append({"p": 1, "kind": "callback", "cid": cid, "due": True, "rank": 0,
                          "text": f"CALLBACK DUE {tag}: {due[0]['reason'][:70]}",
                          "do": f"send_message {cid}: keep the promised follow-up ({due[0]['reason'][:60]})"})
        elif kind == "later":
            when = a.get("present_day") or a.get("ready_day") or today + 1
            later.append((when, cid))
        if not any(item["cid"] == cid for item in items) and touched_today.get(cid, 0) == 0:
            upgrade = _upgrade_lock(lead, account, catalog["products"], today, base, changes)
            if upgrade:
                product, price, feature = upgrade
                ready_day = lead["ready_day"]
                items.append({"p": 1, "kind": "lock_upgrade", "cid": cid, "rank": 1,
                              "profit": price - product["monthly_service_cost_cents"],
                              "text": f"LOCK UPGRADE {tag}: {product['id']} at {_money(price)} in case their "
                                      f"day-{ready_day} decision adds {feature.replace('_', ' ')}",
                              "do": (f"offer {cid} (product_id='{product['id']}', price_cents={price}). Message: "
                                     f"their decision on day {ready_day} may add {feature.replace('_', ' ')}. Say "
                                     f"this {product['name']} price is held through day {ready_day} only in case "
                                     "the decision adds it, that they need not decide now, and that the plan "
                                     "fitting their current needs stays available. No pressure.")})

    # First contacts: a daily quota, fixed at the day's first queue, spreads the remaining accounts
    # over the days left before the target (the day before a worse price change). The quota ranks
    # above offers that can wait; extra first contacts come after them.
    target = min(DISCOVERY_TARGET_DAY, last_day)
    if changes:
        target = min(target, changes[0]["effective_day"] - 1)
    plan = ctx.state.get("first_contact_plan")
    if not plan or plan.get("day") != today:
        days_left = target - today + 1
        quota = -(-len(uncontacted) // days_left) if days_left > 0 else len(uncontacted)
        plan = ctx.state["first_contact_plan"] = {"day": today, "quota": quota, "sent": 0}
    reserved = max(0, plan["quota"] - plan["sent"])
    for n, customer in enumerate(sorted(uncontacted, key=lambda c: _urgency(c["background"]))):
        hint = customer["background"].split("Latest contact: ")[-1][:70]
        items.append({"p": 2 if n < reserved else 4, "kind": "first", "cid": customer["id"],
                      "text": f"FIRST CONTACT {customer['id']} (can buy today): {hint}",
                      "do": f"first_contact {customer['id']}"})

    if everyone_reached:
        # Spare time, lowest priority: one more contact to customers closed only on budget, never on the
        # day they declined; a full day moves it to the next day with room.
        by_id = {c["id"]: c for c in customers}
        for cid, lead in leads.items():
            customer = by_id.get(cid)
            if (not customer or customer["status"] != "prospect" or lead.get("close") != "budget"
                    or not customer["contact"]["eligible"] or touched_today.get(cid, 0) > 0
                    or lead.get("persuasions", 0) >= MAX_PERSUASIONS or lead.get("last_persuade_day") == today):
                continue
            fits = _fitting_plans(lead, catalog["products"])
            if not fits:
                continue
            plan = min(fits, key=lambda p: _min_price(p, base["discount"]))
            pitch = {"plan": plan["id"], "price_cents": _min_price(plan, base["discount"])}
            items.append({"p": LONG_SHOT_TIER, "kind": "persuade", "cid": cid,
                          "text": f"PERSUADE {cid} ({customer['business']}): {plan['id']} at "
                                  f"{_money(pitch['price_cents'])}, lowest legal price "
                                  f"(attempt {lead.get('persuasions', 0) + 1} of {MAX_PERSUASIONS})",
                          "do": _persuade_instruction(ctx, cid, pitch, lead)})
    items.sort(key=lambda item: (item["p"], item.get("rank", 3), -item.get("profit", 0)))

    messages_left = status["minutes_remaining_today"] // status["minutes_per_exchange"]
    if not uncontacted:
        goal = "done"
    elif today <= target:
        goal = (f"reach every account by day {target}; today's quota {plan['quota']}, "
                f"{min(reserved, messages_left)} still reserved")
    else:
        goal = "target day passed: contact as many as the minutes allow"
    later.sort()
    if messages_left == 0:
        do_next = "No messages left today: end the day in one line."
    elif items:
        do_next = items[0]["do"]
    else:
        do_next = "Every account is handled for today: end the day in one line."
    return {"day": today, "of": last_day, "messages_left_today": messages_left,
            "first_contacts_left": len(uncontacted), "first_contacts_goal": goal,
            "items": items, "later": [f"{cid} day {day}" for day, cid in later],
            "sold": sum(c["status"] == "purchased" for c in customers), "do_next": do_next}


QUOTE_CLOSE = 0.9      # a second document is quoted too when it scores this close to the best one
QUOTE_WIDE = 0.3       # after a repeated question: every document this close to the best one, up to 3


def _same_question(a, b):
    return bool(a and b) and set(_tokens(a)) == set(_tokens(b))


def _answer_docs(ctx, question, repeat=False):
    """The documents answering a question, in full, to be quoted word for word. Company documents are
    short, and a whole document keeps facts that sentence picking drops (Driftwood's support hours).
    After a repeated question the answer widens to the next-best documents."""
    ranked = [(s, d) for s, d in _search_docs(ctx, question, exclude=NOT_ANSWERS)[:3] if s >= 0.1]
    if not ranked:
        return []
    best = ranked[0][0]
    return [d for s, d in ranked if s >= best * (QUOTE_WIDE if repeat else QUOTE_CLOSE)]


def _quotes_for(ctx, question, repeat=False):
    docs = _answer_docs(ctx, question, repeat)
    if not docs:
        return "(no document matched: use find_docs)"
    return " ".join(f"\"{d['text']}\" [{d['id']}]" for d in docs)


def _tool_quotes(question, lead):
    """True when the offer tool itself puts the answer documents first (any first-time question).
    A repeated question goes back to the model, which reads the documents and answers every part."""
    return bool(question) and not _same_question(question, (lead or {}).get("last_answered", ""))


def _answer_for(ctx, question, lead, by_offer=False):
    """The quoting instruction for an open question; a repeat gets more documents and find_docs.
    by_offer: the offer tool inserts the document text itself, so the text stays out of the history."""
    repeat = _same_question(question, (lead or {}).get("last_answered", ""))
    if by_offer and not repeat:
        ids = [d["id"] for d in _answer_docs(ctx, question)]
        if ids:
            return (f"the documents {ids}: the offer tool puts their exact text at the start of your "
                    "message, so do not retype or paraphrase them.")
    text = f"quoting these documents word for word: {_quotes_for(ctx, question, repeat)}"
    if repeat:
        text += (" They asked this again, so the last answer missed what they need: call find_docs with "
                 "their question first and answer every part of it (when, how, which plan) from the documents.")
    return text


def _offer_instruction(ctx, customer_id, assessment, question, reuse, lead=None):
    """The exact offer call, and what its message must contain: answer first, then the plan."""
    call = (f"offer {customer_id} (product_id='{assessment['plan']}', price_cents={assessment['price_cents']})"
            f"{reuse}.")
    if assessment.get("final") and lead is not None:
        # Budget below every legal price: the final offer, asking whether anything changed.
        pitch = {"plan": assessment["plan"], "price_cents": assessment["price_cents"]}
        return (_persuade_instruction(ctx, customer_id, pitch, lead)
                + " Ask them to tell you if their budget or requirements have changed.")
    if question:
        return (f"{call} First answer \"{question[:120]}\" by {_answer_for(ctx, question, lead, by_offer=True)} "
                "Then ask if needs or budget changed, then plan and price.")
    return call


def _answer_instruction(ctx, customer_id, question):
    """A spare-time answer to a customer who decides later."""
    return (f"send_message {customer_id}: answer \"{question[:120]}\" by quoting these document sentences "
            f"word for word: {_quotes_for(ctx, question)} Then remind them of the plan you will offer on their "
            "decision day. No small talk.")


def _persuade_instruction(ctx, customer_id, pitch, lead):
    """One persuasion offer at the lowest legal price, built on what the customer said."""
    risk = _key_sentences(ctx, "trial cancellation refund first month price", limit=1)
    risk_line = f" Lower the risk by quoting: \"{risk[0][1]}\"" if risk else ""
    reasons = ", ".join(lead.get("needs", []) + lead.get("nice_to_haves", [])) or "their stated needs"
    question = lead.get("open_question", "")
    answer = (f" First answer \"{question[:100]}\" by {_answer_for(ctx, question, lead, by_offer=True)}"
              if question else "")
    return (f"offer {customer_id} (product_id='{pitch['plan']}', price_cents={pitch['price_cents']}). "
            f"Message:{answer} Persuade briefly: the plan solves {reasons}; this is the lowest price the policy "
            f"allows.{risk_line} Ask whether their budget or requirements have changed. One short, honest "
            "message; no pressure, no promises beyond the documents.")


# --------------------------------------------------------------------------- document retrieval (cosine)

def _stem(word):
    for suffix in ("ations", "ation", "ings", "ing", "ions", "ion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _tokens(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2]


def _documents(ctx):
    """Every document available today, read once per day and cached."""
    today = ctx.get_status()["day"]
    cache = ctx.state.get("doc_cache")
    if not cache or cache.get("day") != today:
        docs = []
        for listed in ctx.list_documents():
            document = ctx.read_document(document_id=listed["id"])
            docs.append({"id": document["id"], "title": document["title"], "text": document["text"]})
        cache = ctx.state["doc_cache"] = {"day": today, "docs": docs}
    return cache["docs"]


def _index(docs):
    """TF-IDF vectors; title and id words count three times."""
    counts = []
    for d in docs:
        bag = {}
        for w in _tokens(d["text"]) + _tokens(d["id"] + " " + d["title"]) * 3:
            bag[w] = bag.get(w, 0) + 1
        counts.append(bag)
    df = {}
    for bag in counts:
        for w in bag:
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
    return idf, [_weigh(bag, idf) for bag in counts]


def _weigh(bag, idf):
    return {w: c * idf.get(w, 1.0) for w, c in bag.items()}


def _bag(text):
    bag = {}
    for w in _tokens(text):
        bag[w] = bag.get(w, 0) + 1
    return bag


def _cosine(a, b):
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _search_docs(ctx, query, exclude=()):
    """Documents ranked by cosine similarity to the query (archived ones last)."""
    docs = _documents(ctx)
    idf, vectors = _index(docs)
    q = _weigh(_bag(query), idf)
    scored = []
    for d, v in zip(docs, vectors):
        if d["id"].startswith(exclude):
            continue
        score = _cosine(q, v) - (1.0 if "archive" in d["id"].lower() else 0.0)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda item: -item[0])
    return scored


SENTENCE_END = re.compile(r"(?<![ap]\.m\.)(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?])\s+(?=[A-Z\"(])")   # keeps "8 a.m. to 6 p.m." whole


def _key_sentences(ctx, question, limit=3):
    """The sentences from the best documents that answer a question, to be quoted word for word."""
    ranked = [(s, d) for s, d in _search_docs(ctx, question, exclude=NOT_ANSWERS)[:3] if s >= 0.1]
    if not ranked:
        return []
    idf, _ = _index(_documents(ctx))
    q = _weigh(_bag(question), idf)
    scored = []
    for _, d in ranked:
        for sentence in SENTENCE_END.split(d["text"]):
            score = _cosine(q, _weigh(_bag(sentence), idf))
            if score >= SENTENCE_MATCH:
                scored.append((score, d["id"], sentence.strip()))
    scored.sort(key=lambda item: -item[0])
    return [(doc_id, sentence) for _, doc_id, sentence in scored[:limit]]


# --------------------------------------------------------------------------- first contact

def _first_message(ctx, customer_id, catalog):
    """Discovery message: plans, the documents answering the concern in the CRM record
    (a bundle per concern, so follow-up questions are covered too), the questions."""
    history = ctx.read_customer_history(customer_id=customer_id)
    name = history["customer"]["name"].split()[0]
    texts = [history["customer"].get("background", "")]
    for record in history["record_index"]:
        if record["kind"] == "interaction":
            data = ctx.read_record(record_id=record["id"])["data"]
            texts += [m["text"] for m in data.get("messages", []) if m.get("role") != "salesperson"]
    concern = " ".join(texts).lower()
    answers, used = [], set()
    for triggers, searches in TOPICS:
        if not any(trigger in concern for trigger in triggers):
            continue
        for search in searches:
            if len(used) >= MAX_FIRST_CONTACT_DOCS:
                break
            ranked = _search_docs(ctx, search, exclude=NOT_ANSWERS)
            if ranked and ranked[0][0] >= DOC_MATCH and ranked[0][1]["id"] not in used:
                used.add(ranked[0][1]["id"])
                answers.append(ranked[0][1]["text"])
    plans = "; ".join(
        f"{p['name']} {_money(p['monthly_price_cents'])}/month ({', '.join(f.replace('_', ' ') for f in p['features'])}; "
        f"works with {', '.join(p['integrations'])})" for p in catalog["products"])
    message = (f"Hi {name}, thanks for considering our scheduling software. Each plan is one flat monthly "
               f"price for the whole account: {plans}.")
    if answers:
        message += " On what you raised: " + " ".join(answers)
    message += (" So I can recommend the right plan and price: which features and live integrations must you "
                "have, what is the most you would willingly pay per month, what is the earliest day you can "
                "approve a purchase, and do your needs or budget depend on an upcoming meeting or decision?")
    return message, sorted(used)


def _send_first_contact(ctx, customer_id, catalog, status):
    """Compose and send a first contact; the send is the last operation."""
    message, sources = _first_message(ctx, customer_id, catalog)
    _touch(ctx, customer_id, status["day"])
    plan = ctx.state.get("first_contact_plan")
    if plan and plan.get("day") == status["day"]:
        plan["sent"] += 1                      # counts against today's first-contact quota
    messages_left = status["minutes_remaining_today"] // status["minutes_per_exchange"] - 1
    result = ctx.talk_to_customer(customer_id=customer_id, message=message)
    out = {"first_contact_sent_to": customer_id, "reply": result["reply"]["text"],
           "decision": result["decision"], "messages_left_today": messages_left}
    if result["decision"] == "opt_out":
        ctx.state.setdefault("leads", {})[customer_id] = {"dropped": True}
        out["do_next"] = "Opted out: never contact again. work_queue"
    else:
        out["do_next"] = f"record_lead {customer_id}"
    return out


# --------------------------------------------------------------------------- planning tools

@tool
def campaign_brief(ctx) -> dict:
    """Call on day 1: plans, today's pricing, and documents announcing price changes."""
    status, catalog = ctx.get_status(), ctx.get_catalog()
    policy = catalog["current_pricing_policy"]
    plans = [{"id": p["id"], "features": p["features"], "integrations": p["integrations"],
              "list": _money(p["monthly_price_cents"]),
              "lowest_legal_today": _money(_min_price(p, policy["max_discount_percent"])),
              "service_cost": _money(p["monthly_service_cost_cents"])} for p in catalog["products"]]
    documents = ctx.list_documents()
    seen = set(ctx.state.get("seen_documents", []))
    new = [d for d in documents if d["id"] not in seen]
    ctx.state["seen_documents"] = sorted(seen | {d["id"] for d in documents})
    notices = []
    for listed in documents:
        if listed["id"].startswith(("product:", "policy:", "archive")):
            continue
        text = ctx.read_document(document_id=listed["id"])["text"]
        lowered = (listed["title"] + " " + text).lower()
        if re.search(r"pric|discount", lowered) and re.search(r"\bday \d+|from day|effective", lowered):
            notices.append({"id": listed["id"], "title": listed["title"], "text": text})
    result = {"day": status["day"], "campaign_days": status["campaign_days"],
              "minutes_left_today": status["minutes_remaining_today"],
              "pricing_today": {"policy": policy["id"], "max_discount_percent": policy["max_discount_percent"],
                                "quote_presented_today_valid_through_day": status["day"] + policy["quote_valid_days"]},
              "plans": plans,
              "registered_price_changes": ctx.state.get("price_changes", []),
              "pricing_notices": notices}
    if seen and new:
        result["new_documents"] = [{"id": d["id"], "title": d["title"]} for d in new]
    if notices and not ctx.state.get("price_changes"):
        result["do_next"] = "register_price_change for the announced policy, then work_queue"
    else:
        result["do_next"] = "work_queue"
    return result


@tool
def register_price_change(ctx, effective_day: int, max_discount_percent: int, quote_valid_days: int) -> dict:
    """Record an announced future pricing policy, e.g. from day 7, 5% off, quotes valid 2 days
    -> (7, 5, 2). Once per change."""
    changes = [c for c in ctx.state.get("price_changes", []) if c["effective_day"] != effective_day]
    changes.append({"effective_day": effective_day, "discount": max_discount_percent, "valid": quote_valid_days})
    ctx.state["price_changes"] = sorted(changes, key=lambda c: c["effective_day"])
    return {"registered": ctx.state["price_changes"], "do_next": "work_queue"}


@tool
def work_queue(ctx) -> dict:
    """The next task (do_next). Call at the start of every day. If the next task is a first
    contact, it SENDS it and returns the reply: record it with record_lead."""
    queue = _build_queue(ctx)
    result = {"day": queue["day"], "messages_left_today": queue["messages_left_today"],
              "todo": [item["text"] for item in queue["items"][:1]]}     # only the next task: short output
    top = queue["items"][0] if queue["items"] else None
    if top and top["kind"] == "first" and queue["messages_left_today"] > 0:
        status, catalog = ctx.get_status(), ctx.get_catalog()
        result.update(_send_first_contact(ctx, top["cid"], catalog, status))
        return result
    result["do_next"] = queue["do_next"]
    return result


@tool
def check_time(ctx) -> dict:
    """Call BEFORE ending a day: says whether you may end it (messages left, work left)."""
    status = ctx.get_status()
    messages_left = status["minutes_remaining_today"] // status["minutes_per_exchange"]
    result = {"messages_left_today": messages_left}
    if messages_left == 0:
        result["may_end_day"] = True
        result["do_next"] = "End the day in one line."
        return result
    queue = _build_queue(ctx)
    if not queue["items"]:
        result["may_end_day"] = True
        result["do_next"] = queue["do_next"]
    else:
        result["may_end_day"] = False
        result["do_next"] = f"Do NOT end the day. {queue['items'][0]['do']}"
    return result


@tool
def read_account(ctx, customer_id: str) -> dict:
    """One account in detail (history, quotes, callbacks, recorded lead). Rarely needed."""
    history = ctx.read_customer_history(customer_id=customer_id)
    status = ctx.get_status()
    today = status["day"]
    crm = []
    for record in history["record_index"]:
        if record["kind"] in ("interaction", "sale", "refund", "cancellation"):
            data = ctx.read_record(record_id=record["id"])["data"]
            texts = [m["text"] for m in data.get("messages", [])] or [str(data)[:200]]
            crm.append(f"{record['date']} {record['kind']}: " + " / ".join(texts))
    conversation = []
    for message in history["history"]:
        who = "you" if message["role"] == "salesperson" else f"customer[{message['decision']}]"
        text = message["text"] if message["role"] != "salesperson" else message["text"][:140]
        quote = f" (quote {message['quote_id']})" if message["quote_id"] else ""
        conversation.append(f"day {message['day']} {who}{quote}: {text}")
    result = {"customer": history["customer"]["name"], "business": history["customer"]["business"],
              "status": history["status"], "contact": history["contact"],
              "crm_background": history["customer"].get("background"),
              "crm_history": crm, "conversation": conversation,
              "quotes": [_quote_view(q, today, history["orders"]) for q in history["quotes"]],
              "pending_callbacks": [{"day": cb["day"], "reason": cb["reason"]} for cb in history["pending_callbacks"]]}
    lead = ctx.state.get("leads", {}).get(customer_id)
    if lead and lead.get("needs") is not None:
        result["recorded"] = {k: v for k, v in lead.items() if k != "dropped"}
        result["recommendation"] = _summary(_assess_customer(ctx, customer_id, status=status, account=history))
    return result


@tool
def record_lead(ctx, customer_id: str, needs: list[str], nice_to_haves: list[str], integrations: list[str],
                budget_cents: int | None, max_budget_cents: int | None, ready_day: int | None,
                scope_final: bool, pending_features: list[str], open_question: str, unsellable_reason: str,
                notes: str) -> dict:
    """Save a customer's reply (call after EVERY reply; latest replaces older). Returns the plan,
    price and do_next; may SEND the next first contact and return its reply.
    needs: must-have feature ids (also ones no plan offers). nice_to_haves: feature ids valued
    but not required. integrations: must-have integration names.
    budget_cents: what they willingly pay per month; null if no number (never guess).
    max_budget_cents: their stated approved limit, else null.
    ready_day: earliest approval day (a day after the campaign if they cannot approve within
    it); null if now. scope_final: false if a pending decision may change needs or budget.
    pending_features: feature ids that decision could add; [] if none.
    open_question: their unanswered question, verbatim; "" if none.
    unsellable_reason: only if a must-have is confirmed unavailable (check find_docs) and
    nothing pending could change it; else "". notes: "" unless essential (under 10 words)."""
    leads = ctx.state.setdefault("leads", {})
    previous = leads.get(customer_id, {})
    new_budget = budget_cents if budget_cents is not None else previous.get("budget_cents")
    new_upper = max_budget_cents if max_budget_cents is not None else previous.get("max_budget_cents")
    learned = set(previous.get("plan_caps", {}).values())
    budget_changed = ((new_budget, new_upper) != (previous.get("budget_cents"), previous.get("max_budget_cents"))
                      and new_budget not in learned)   # copying an offer's "at most $X" is not a new budget
    lead = {"needs": needs, "nice_to_haves": nice_to_haves, "integrations": integrations,
            "budget_cents": new_budget, "max_budget_cents": new_upper,
            "ready_day": ready_day, "scope_final": scope_final, "open_question": open_question.strip(),
            "pending_features": [] if scope_final else [str(f).strip().lower() for f in pending_features
                                                        if str(f).strip()],
            "last_answered": previous.get("last_answered", ""),
            "unsellable_reason": unsellable_reason.strip(), "notes": notes[:300],
            # Prices learned from offer replies, and the one-time final offer, hold until the budget changes.
            "plan_caps": {} if budget_changed else previous.get("plan_caps", {}),
            "best_price_offered": False if budget_changed else previous.get("best_price_offered", False),
            "declined_quotes": [] if budget_changed else previous.get("declined_quotes", []),
            "persuasions": previous.get("persuasions", 0), "last_persuade_day": previous.get("last_persuade_day")}
    leads[customer_id] = lead
    assessment = _assess_customer(ctx, customer_id)
    if assessment and assessment["kind"] == "closed":
        lead["dropped"] = True
        lead["close"] = assessment.get("close", "budget")   # window / capability: never again; budget: persuade later
    out = {"saved": customer_id, "recommendation": _summary(assessment)}
    if assessment and assessment.get("close") == "window":
        out["abandoned"] = "they cannot approve within the campaign; no further contact"

    # A customer who can only decide later gets a callback on that day, so it is never missed.
    today = ctx.get_status()["day"]
    if assessment and assessment["kind"] != "closed" and ready_day is not None and ready_day > today:
        account = ctx.read_customer(customer_id=customer_id)
        if not account["pending_callbacks"] and account["status"] == "prospect":
            booked = ctx.call("schedule_callback", customer_id=customer_id, day=ready_day,
                              reason=f"Customer can decide from day {ready_day}: confirm needs and offer.")
            if booked["ok"]:
                out["callback_booked"] = f"day {ready_day}"

    queue = _build_queue(ctx)
    own = [item for item in queue["items"] if item["cid"] == customer_id and item["p"] <= 1]
    out["messages_left_today"] = queue["messages_left_today"]
    if own and queue["messages_left_today"] > 0:
        out["do_next"] = own[0]["do"]           # this customer can buy or decide now: finish with them first
        return out
    top = queue["items"][0] if queue["items"] else None
    if top and top["kind"] == "first" and queue["messages_left_today"] > 0:
        status, catalog = ctx.get_status(), ctx.get_catalog()
        out.update(_send_first_contact(ctx, top["cid"], catalog, status))   # chain to the next new account
        return out
    out["do_next"] = queue["do_next"]
    return out


@tool
def find_docs(ctx, question: str) -> list:
    """The company documents that answer a question (best first, full text): quote them word
    for word. Archived documents are outdated."""
    ranked = _search_docs(ctx, question)[:3]
    best = ranked[0][0] if ranked else 0
    # Only documents close to the best match: each result is re-read on every later call.
    return [{"id": d["id"], "text": d["text"]} for s, d in ranked if s >= best * 0.5]


# --------------------------------------------------------------------------- contact tools

def _contact_check(ctx, customer_id):
    account = ctx.read_customer(customer_id=customer_id)
    if account["status"] != "prospect":
        return account, {"error": f"Do not contact: status is {account['status']}.", "do_next": "work_queue"}
    if not account["contact"]["eligible"]:
        return account, {"error": f"Cannot contact today: {account['contact']['reason']}.", "do_next": "work_queue"}
    return account, None


@tool
def first_contact(ctx, customer_id: str) -> dict:
    """Send the first message to a never-contacted account (usually sent for you)."""
    account, problem = _contact_check(ctx, customer_id)
    if problem:
        return problem
    return _send_first_contact(ctx, customer_id, ctx.get_catalog(), ctx.get_status())


@tool
def send_message(ctx, customer_id: str, message: str) -> dict:
    """Send one message and get the reply. Only when do_next says send_message."""
    account, problem = _contact_check(ctx, customer_id)
    if problem:
        return problem
    status = ctx.get_status()
    _touch(ctx, customer_id, status["day"])
    has_valid_quote = bool(_valid_issued(account["quotes"], status["day"]))
    lead = ctx.state.get("leads", {}).get(customer_id)
    if lead and lead.get("open_question"):
        lead["last_answered"] = lead["open_question"]
        lead["open_question"] = ""             # this message answers it
    result = ctx.talk_to_customer(customer_id=customer_id, message=message)
    if result["decision"] == "opt_out":
        ctx.state.setdefault("leads", {})[customer_id] = {"dropped": True}
    output = {"reply": result["reply"]["text"], "decision": result["decision"],
              "messages_left_today": status["minutes_remaining_today"] // status["minutes_per_exchange"] - 1,
              "do_next": f"record_lead {customer_id}"}
    if has_valid_quote and result["decision"] == "continue":
        output["reminder"] = "Valid quote: re-present it with offer."
    return output


@tool
def offer(ctx, customer_id: str, product_id: str, price_cents: int, message: str) -> dict:
    """Present a formal offer (one message) and get the decision. Use the product_id and
    price_cents do_next gives; reuses a valid quote; refuses illegal prices. If the customer
    has an open question, the tool puts the answering documents' exact text first: write only
    the rest of the message."""
    account, problem = _contact_check(ctx, customer_id)
    if problem:
        return problem
    recorded = ctx.state.get("leads", {}).get(customer_id)
    if not recorded or not recorded.get("needs"):
        # No discovery yet: an offer would guess the plan (Copper Bicycle needed OtherSuite).
        return {"error": f"No needs recorded for {customer_id}: discover them before offering.",
                "do_next": (f"first_contact {customer_id}" if account["contact"]["last_contact_day"] is None
                            else f"record_lead for {customer_id} from their last reply")}
    status, catalog = ctx.get_status(), ctx.get_catalog()
    today = status["day"]
    product = next((p for p in catalog["products"] if p["id"] == product_id), None)
    if product is None:
        return {"error": f"Unknown plan {product_id!r}. Plans: {[p['id'] for p in catalog['products']]}"}
    # Re-present a held quote only at its own price, or when the asked price cannot be quoted fresh
    # today: a higher legal price is never swapped for a cheaper lock.
    fresh_legal = min(price_cents, product["monthly_price_cents"]) >= _min_price(
        product, catalog["current_pricing_policy"]["max_discount_percent"])
    reuse = [q for q in _valid_issued(account["quotes"], today)
             if q["product_id"] == product_id and (q["monthly_price_cents"] == price_cents
                                                   or (not fresh_legal and q["monthly_price_cents"] <= price_cents))]
    if reuse:
        quote = max(reuse, key=lambda q: q["monthly_price_cents"])
        expires = quote["expires_day"]
    else:
        policy = catalog["current_pricing_policy"]
        low, high = _min_price(product, policy["max_discount_percent"]), product["monthly_price_cents"]
        price = min(price_cents, high)
        if price < low:
            return {"error": f"{_money(price)} is below today's legal minimum for {product_id}.",
                    "legal_range_today": f"{_money(low)} to {_money(high)}",
                    "do_next": f"record_lead for {customer_id} (the account may be closed), then continue"}
        quote = ctx.create_quote(customer_id=customer_id, product_id=product_id, monthly_price_cents=price)
        expires = today + policy["quote_valid_days"]
    _touch(ctx, customer_id, today)
    lead = ctx.state.get("leads", {}).get(customer_id)
    # An open question is answered by the documents' exact text, put first by the tool (the model
    # writes only the plan and price), so the document text never enters the conversation history.
    question = (lead or {}).get("open_question", "")
    if _tool_quotes(question, lead):
        docs = [d for d in _answer_docs(ctx, question) if d["text"][:80] not in message]
        if docs:
            message = " ".join(f"\"{d['text']}\" [{d['id']}]" for d in docs) + " " + message.strip()
    if lead is not None and lead.get("needs") is not None:
        if lead.get("open_question"):
            lead["last_answered"] = lead["open_question"]
        lead["open_question"] = ""             # the offer message answers it
        if lead.get("close") == "budget":      # a persuasion attempt on an account closed on budget
            lead["persuasions"] = lead.get("persuasions", 0) + 1
            lead["last_persuade_day"] = today
        known = [c for c in (lead.get("plan_caps", {}).get(product_id), lead.get("budget_cents"),
                             lead.get("max_budget_cents")) if c is not None]
        if known and quote["monthly_price_cents"] > max(known):
            lead["best_price_offered"] = True  # offered above everything they said they would pay
    result = ctx.present_quote(quote_id=quote["id"], message=message)
    # "For this offer, I would pay at most $X": remember it as this plan's price cap.
    reply_text = result["reply"]["text"] if result.get("reply") else ""
    at_most = re.search(r"at most \$([0-9][0-9,]*(?:\.[0-9]{2})?)", reply_text)
    if at_most and lead is not None and lead.get("needs") is not None and not result.get("order"):
        stated = round(float(at_most.group(1).replace(",", "")) * 100)
        lead.setdefault("plan_caps", {})[product_id] = stated
        if stated < _min_price(product, catalog["current_pricing_policy"]["max_discount_percent"]):
            lead["best_price_offered"] = True  # they turned down our lowest legal price: the final offer is made
    if (lead is not None and lead.get("needs") is not None and not result.get("order")
            and result["decision"] in ("decline", "end_call")):
        declined = lead.setdefault("declined_quotes", [])
        if quote["id"] not in declined:
            declined.append(quote["id"])       # never re-present it unless their budget or scope changes
    output = {"decision": result["decision"],
              "reply": result["reply"]["text"] if result.get("reply") else None,
              "messages_left_today": status["minutes_remaining_today"] // status["minutes_per_exchange"] - 1}
    if result.get("order"):
        output["order"] = "ACCEPTED: never contact again"
        output["do_next"] = "work_queue"
    else:
        output["do_next"] = f"record_lead {customer_id}"
    return output


@tool
def schedule_callback(ctx, customer_id: str, day: int, reason: str) -> dict:
    """Book a follow-up on a day you promised (decision-day callbacks are booked for you)."""
    callback = ctx.schedule_callback(customer_id=customer_id, day=day, reason=reason)
    return {"booked": callback["id"], "day": callback["day"], "do_next": "work_queue"}


@tool
def read_document(ctx, document_id: str) -> dict:
    """Read one company document by id (ids come from find_docs or campaign_brief)."""
    return ctx.read_document(document_id=document_id)
