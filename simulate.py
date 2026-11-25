"""Free offline run of the 40-account campaign with your tools.py.

Uses the course's real campaign data, environment, pricing, quotes, callbacks and settlement.
Two model calls are replaced by deterministic stand-ins, so nothing is paid:

  * the customer's interpretation call (sales_sim.discovery_customers._interpret): a seller
    message that asks anything discloses every current fact; a purchase concern counts as
    answered when a seller message quotes one of the concern's documents in full;
  * the seller model: it follows do_next, reads customer replies with simple parsing (the
    customers' facts are fixed sentences) and calls check_time before ending a day.

What it measures: profit, orders, missed callbacks, unsold servable accounts (audit.json),
and an estimate of input tokens (every model call resends the whole history, as agent.py does).
What it cannot measure: how the real model behaves, and customers' free-form judgement.
Compare versions of tools.py with it; confirm a final version with one paid run.

    python simulate.py                 # summary
    python simulate.py --accounts      # plus one line per account
    python simulate.py --json out.json # save the full result
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sales_sim import notebook  # noqa: E402
from sales_sim.crm import load_dataset  # noqa: E402
from sales_sim.discovery_customers import DiscoveryCustomers  # noqa: E402
from sales_sim.environment import SalesEnvironment  # noqa: E402
from sales_sim.harness import World  # noqa: E402
from sales_sim.student_submission import load_submission  # noqa: E402
from sales_sim.tools import LocalToolClient  # noqa: E402

MAX_TURNS_PER_DAY = 100          # agent.py's limit
CHARS_PER_TOKEN = 4              # rough estimate for English and JSON

FEATURE_PHRASES = {              # how the campaign's customers name catalog features
    "appointment scheduling": "scheduling",
    "automatic reminders": "reminders",
    "separate and consolidated site reporting": "location_reporting",
}
PENDING_HINTS = {                # what a pending decision may add, as the model would read it
    "location_reporting": ("site", "venue", "location", "report", "dashboard", "compare"),
    "reminders": ("reminder",),
}
PROVISIONAL = ("review", "decide", "decision", "provisional", "pending", "may ", "meeting", "vote",
               "approves", "funding", "will be selected", "replace")


# --------------------------------------------------------------------------- the customer stand-in

class _NoModel:
    """Stands in for the API client; any use means a paid call slipped through."""

    def __getattr__(self, name):
        raise RuntimeError(f"simulate.py must not call a model (tried client.{name})")


class ScriptedCustomers(DiscoveryCustomers):
    """The real discovery customers with the interpretation model call replaced."""

    def _interpret(self, turn, profile, state, documents):
        message = turn.salesperson_message
        asks = "?" in message
        facts = [fact["id"] for fact in profile["facts"]] if asks else []
        sellers = {"current": message, **{m.id: m.text for m in turn.history if m.role == "salesperson"}}
        checks = {}
        for concern in profile.get("purchase_concerns", []):
            if concern["id"] in state["resolved_concerns"]:
                continue
            hit = next(((mid, key) for key in concern["document_ids"] if key in documents
                        for mid, text in sellers.items() if documents[key].text.strip() in text), None)
            checks[concern["id"]] = {
                r["id"]: ({"status": "answered", "message_id": hit[0], "document_id": hit[1]} if hit
                          else {"status": "not_addressed", "message_id": "", "document_id": ""})
                for r in self._requirements(concern)}
        payload = {"facts": facts, "concern_checks": checks,
                   "friction": {"kind": "none", "prior_message_id": ""}, "closure": {"kind": "none"}}
        return self._decode_events(payload, turn, profile, state, documents)


# --------------------------------------------------------------------------- the seller stand-in

def money(text):
    return round(float(text.replace(",", "")) * 100)


def read_reply(text, catalog_features, previous):
    """What the model would record from a customer's reply (fixed fact sentences)."""
    lead = dict(previous)
    lowered = text.lower()
    need = re.search(r"we need (appointment scheduling[^.]*)\.", lowered)   # the scope fact, not the opening
    if need:
        listed = need.group(1)
        needs = []
        for phrase, feature in FEATURE_PHRASES.items():     # known phrases first: they contain "and"
            if phrase in listed:
                needs.append(feature)
                listed = listed.replace(phrase, "")
        unknown = [part.strip() for part in re.split(r",|and", listed) if part.strip()]
        lead["needs"] = needs + [u.replace(" ", "_") for u in unknown]
        lead["unsellable_reason"] = (f"No plan offers: {', '.join(unknown)}." if unknown else "")
    integration = re.search(r"a live (\w+) connection is required", lowered)
    if integration:
        lead["integrations"] = [re.search(r"a live (\w+) connection", text, re.I).group(1)]
    limit = re.search(r"(?:approved monthly limit is|approved account limit is now) \$([\d,.]+)", text)
    willing = re.search(r"willingly spend \$([\d,.]+)", text)
    if limit:
        lead["max_budget_cents"] = money(limit.group(1))
        lead["budget_cents"] = money(willing.group(1)) if willing else money(limit.group(1))
    elif willing:
        lead["budget_cents"] = money(willing.group(1))
    day = re.search(r"(?:from campaign day|from day) (\d+)", lowered)
    if day:
        lead["ready_day"] = int(day.group(1))
    if "decision is complete" in lowered or "confirmed scope has changed" in lowered:
        lead["scope_final"] = True
    elif any(word in lowered for word in PROVISIONAL):
        lead["scope_final"] = False
    lead.setdefault("scope_final", True)
    current = set(lead.get("needs", []))
    lead["pending_features"] = [] if lead["scope_final"] else sorted(
        f for f, words in PENDING_HINTS.items()
        if f in catalog_features and f not in current and any(w in lowered for w in words))
    questions = [s.strip() for s in re.findall(r"[^.?!]*\?", text)]
    lead["open_question"] = questions[-1] if questions else ""
    return lead


class Seller:
    """Follows do_next the way the model does; one tool call per model turn."""

    def __init__(self, world, toolbox, base_tokens, profiles=None):
        self.world, self.toolbox = world, toolbox
        self.profiles = profiles or {}
        self.history_chars = 0
        self.base = base_tokens
        self.calls = []                       # per model call: estimated input tokens
        self.leads = {}
        self.trace = []
        self.unhandled = []           # do_next texts this stand-in could not act on
        self.features = set()
        self.ids = 0

    # one model turn: the model's call, then the tool output joins the history
    def turn(self, name, arguments):
        self.calls.append(self.base + self.history_chars // CHARS_PER_TOKEN)
        self.ids += 1
        call_text = json.dumps({"name": name, "arguments": arguments})
        output = self.toolbox.run(name, json.dumps(arguments), self.world, f"sim:{self.ids}")
        self.history_chars += len(call_text) + len(output)
        result = json.loads(output)
        data = result.get("data", result) if isinstance(result, dict) else result
        self.trace.append((self.world.day, name, arguments.get("customer_id", ""),
                           (data.get("decision") if isinstance(data, dict) else None),
                           (str(data.get("do_next"))[:90] if isinstance(data, dict) else "")))
        return data

    def end_day(self):
        self.calls.append(self.base + self.history_chars // CHARS_PER_TOKEN)   # the reply that ends it
        note = "Ending the day."
        self.history_chars += len(note)
        self.world.end_day(note)
        if not self.world.finished:
            self.history_chars += len(self.world.morning_note())

    def record(self, cid, reply):
        lead = read_reply(reply or "", self.features, self.leads.get(cid, {}))
        profile = self.profiles.get(cid)
        if profile is not None:
            # A careful reader: the scope is open only while the customer's needs or budget are
            # really still pending (a later update exists); a board vote on a fixed budget is not.
            pending = [u for u in profile.get("updates", []) if u["day"] > self.world.day]
            lead["scope_final"] = not pending
            current = set(lead.get("needs", []))
            added = {f for u in pending for f in u.get("required_features", [])} - current
            lead["pending_features"] = sorted(added & self.features)
        self.leads[cid] = lead
        args = {"customer_id": cid, "needs": lead.get("needs", []), "nice_to_haves": [],
                "integrations": lead.get("integrations", []), "budget_cents": lead.get("budget_cents"),
                "max_budget_cents": lead.get("max_budget_cents"), "ready_day": lead.get("ready_day"),
                "scope_final": lead.get("scope_final", True), "pending_features": lead.get("pending_features", []),
                "open_question": lead.get("open_question", ""),
                "unsellable_reason": lead.get("unsellable_reason", ""), "notes": ""}
        return self.turn("record_lead", args)

    @staticmethod
    def compose(instruction):
        """The message a do_next asks for: quoted documents verbatim, the question, the plan."""
        quotes = re.findall(r'"([^"]{40,})" \[[^\]]+\]', instruction)
        parts = list(dict.fromkeys(quotes))
        if "changed" in instruction or "CONFIRM" in instruction or "decision" in instruction:
            parts.append("Has anything changed in your needs or budget? What is your approved monthly limit now?")
        parts.append("This plan covers what you told us you need.")
        return " ".join(parts)

    def follow(self, data):
        """Act on one tool result; returns the next tool result, or None when the day should end."""
        if not isinstance(data, dict):
            return self.turn("work_queue", {})
        if data.get("reply") is not None and data.get("first_contact_sent_to"):
            cid = str(data["first_contact_sent_to"]).split()[0]
            self.last_reply[cid] = data["reply"]
        do = str(data.get("do_next") or "").strip()
        low = do.lower()
        match = re.match(r"offer (\S+) \(product_id='(\w+)', price_cents=(\d+)\)", do)
        if match:
            cid = match.group(1)
            out = self.turn("offer", {"customer_id": cid, "product_id": match.group(2),
                                      "price_cents": int(match.group(3)), "message": self.compose(do)})
            self.last_reply[cid] = out.get("reply") or ""
            return out
        match = re.match(r"record_lead (?:for )?(\S+?)[:,.]?(?:\s|$)", do)
        if match:
            cid = match.group(1)
            return self.record(cid, data.get("reply") or self.last_reply.get(cid, ""))
        match = re.match(r"send_message (\S+?):", do)
        if match:
            cid = match.group(1)
            out = self.turn("send_message", {"customer_id": cid, "message": self.compose(do)})
            self.last_reply[cid] = out.get("reply") or ""
            return out
        match = re.match(r"first_contact (\S+)", do)
        if match:
            return self.turn("first_contact", {"customer_id": match.group(1)})
        if low.startswith("register_price_change"):
            return self.register(data)
        if "do not end the day" in low:
            return self.follow({"do_next": do.split("Next: ", 1)[-1]})
        if "end the day" in low or "no messages left" in low or "no contact minutes" in low:
            if data.get("may_end_day") is not None:          # this is check_time's own answer
                return None if data["may_end_day"] else self.follow({"do_next": do.split("Next: ", 1)[-1]})
            check = self.turn("check_time", {})
            return None if check.get("may_end_day", True) else self.follow(check)
        if "work_queue" in low:
            return self.turn("work_queue", {})
        self.unhandled.append((self.world.day, do[:100]))
        return None

    def register(self, brief):
        for notice in brief.get("pricing_notices", []):
            text = notice["text"]
            start = re.search(r"from day (\d+)", text, re.I)         # the policy that takes effect later
            if not start:
                continue
            after = text[start.start():]
            discount = re.search(r"(\d+)\s*(?:%|percent)", after)
            valid = re.search(r"valid for (\d+) days", after)
            if discount and valid:
                self.turn("register_price_change", {"effective_day": int(start.group(1)),
                                                    "max_discount_percent": int(discount.group(1)),
                                                    "quote_valid_days": int(valid.group(1))})
        return self.turn("work_queue", {})

    def run(self):
        self.last_reply = {}
        brief = self.turn("campaign_brief", {})
        self.features = {f for plan in brief.get("plans", []) for f in plan["features"]}
        data = self.follow(brief)
        while not self.world.finished:
            turns = 0
            while data is not None and turns < MAX_TURNS_PER_DAY:
                data = self.follow(data)
                turns += 1
            self.end_day()
            if not self.world.finished:
                data = self.turn("work_queue", {})


# --------------------------------------------------------------------------- the campaign

def simulate(root=ROOT, agent_dir=None):
    """agent_dir: a folder holding the tools.py and prompt.py to test (default: this project)."""
    root = Path(root)
    agent = Path(agent_dir) if agent_dir else root
    selected = notebook._case(notebook.CAMPAIGN, root)
    dataset = load_dataset(selected.dataset_path)
    config = json.loads(selected.customer_config_path.read_text(encoding="utf-8"))
    customers = ScriptedCustomers(config, dataset.scenario.products, client=_NoModel(),
                                  documents=dataset.scenario.documents)
    world = World(LocalToolClient(SalesEnvironment(dataset.scenario, customers, dataset.crm)))
    toolbox = notebook.Toolbox(notebook.load_tools((agent / "tools.py").read_text(encoding="utf-8"), agent))
    prompt = _prompt(agent)
    base = (len(prompt) + len(json.dumps(toolbox.schemas())) + len(world.morning_note())) // CHARS_PER_TOKEN
    profiles = {p["customer_id"]: p for p in config["profiles"]}
    seller = Seller(world, toolbox, base, profiles)
    seller.run()
    outcome = world.outcome()

    audit = json.loads((root / "development/full_campaign/audit.json").read_text(encoding="utf-8"))["accounts"]
    status = {c["id"]: c["status"] for c in world._read("list_customers")}
    unsold = sorted(cid for cid, ref in audit.items() if ref["servable"] and status.get(cid) != "purchased")
    wrong = sorted(cid for cid, ref in audit.items() if not ref["servable"] and status.get(cid) == "purchased")
    return {"outcome": outcome, "status": status, "servable_unsold": unsold, "sold_unservable": wrong,
            "model_calls": len(seller.calls), "input_tokens_estimate": sum(seller.calls),
            "last_call_tokens_estimate": seller.calls[-1] if seller.calls else 0,
            "base_tokens_estimate": base, "world": world, "audit": audit, "seller": seller}


def _prompt(agent):
    namespace = {}
    exec(compile((agent / "prompt.py").read_text(encoding="utf-8"), str(agent / "prompt.py"), "exec"), namespace)
    return namespace["SYSTEM_PROMPT"]


def account_lines(result):
    lines = []
    for cid, ref in sorted(result["audit"].items()):
        lead = result["seller"].leads.get(cid, {})
        lines.append(f"{cid} {ref['business'][:26]:26} servable={str(ref['servable']):5} "
                     f"status={result['status'].get(cid, '?'):9} needs={','.join(lead.get('needs', []))} "
                     f"budget={lead.get('budget_cents')} ready={lead.get('ready_day')} final={lead.get('scope_final')}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accounts", action="store_true", help="one line per account")
    parser.add_argument("--json", help="save the summary as JSON")
    parser.add_argument("--agent", help="folder with another tools.py and prompt.py to test")
    args = parser.parse_args()
    result = simulate(agent_dir=args.agent)
    o = result["outcome"]
    summary = {k: o.get(k) for k in ("profit_cents", "accepted_orders", "retained_orders",
                                      "refunded_revenue_cents", "missed_callbacks", "booked_revenue_cents",
                                      "service_cost_cents", "contact_cost_cents")}
    summary.update({k: result[k] for k in ("servable_unsold", "sold_unservable", "model_calls",
                                           "input_tokens_estimate", "last_call_tokens_estimate",
                                           "base_tokens_estimate")})
    if result["seller"].unhandled:
        summary["stand_in_could_not_follow"] = result["seller"].unhandled[:5]
    print(json.dumps(summary, indent=1))
    if args.accounts:
        print("\n".join(account_lines(result)))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
