"""Answers to customer questions, and the decision-day re-ask ranking (run tools-20261124-231414).

Driftwood (lead702, Premium $98) asked "When can we reach support, and how?" on day 5. On day 11
the offer quoted "There is no paid support tier..." plus two OtherSuite sentences, three times;
the customer ended the call and never bought. Causes: the sentence splitter cut "8 a.m. to 6 p.m.",
and "supported integrations" outranked the support hours. Answers now quote the best document in
full, and a repeated question widens to more documents and asks for find_docs.

Spruce (lead599) had a DECISION DAY re-ask due on day 6 that ranked behind locks and final offers;
the minutes ran out and its callback was missed.
"""

import json
import unittest

from sales_sim import tool
from sales_sim.notebook import try_tool

from test_failures import ANCHOR_LIKE, FakeCtx, JUNIPER_CLOSED, ROOT, items_for, lead, queue, tools
from test_lifecycle import Campaign, Customer

SUPPORT_Q = "When can we reach support, and how?"


@tool
def answer_probe(ctx, question: str, last_answered: str) -> dict:
    """Test helper: the quoting instruction tools.py builds for a question."""
    return {"answer": tools._answer_for(ctx, question, {"last_answered": last_answered}),
            "sentences": [s for _, s in tools._key_sentences(ctx, question)]}


def probe(question, last_answered=""):
    """Runs against the real 40-account campaign's documents (reading is free)."""
    out = try_tool("answer_probe", {"question": question, "last_answered": last_answered},
                   case="full_campaign", root=ROOT, tools=[answer_probe])
    return out["result"]


class DocumentAnswers(unittest.TestCase):

    def test_support_answer_has_hours_and_channels(self):
        answer = probe(SUPPORT_Q)["answer"]
        for fact in ("8 a.m. to 6 p.m. Eastern time", "reply within one business day",
                     "Phone support is included only with Premium"):
            self.assertIn(fact, answer)
        self.assertIn("[support:hours]", answer)
        self.assertNotIn("OtherSuite", answer, "the integrations document does not answer a support question")

    def test_whole_document_is_quoted(self):
        answer = probe(SUPPORT_Q)["answer"]
        self.assertIn("Outside these hours, the help center remains available.", answer)

    def test_other_questions_get_their_own_document(self):
        cases = {
            "Can student contacts and old reminder delivery logs be imported?": ("Historical reminder delivery "
                                                                                 "logs are not imported", None),
            "When can we export records relative to the trial and cancellation?": (
                "Exports are available during the 30-day trial", "[data:exports]"),
        }
        for question, (fact, source) in cases.items():
            with self.subTest(question=question):
                answer = probe(question)["answer"]
                self.assertIn(fact, answer)
                if source:
                    self.assertIn(source, answer)

    def test_first_answer_does_not_say_repeat(self):
        self.assertNotIn("asked this again", probe(SUPPORT_Q)["answer"])

    def test_repeated_question_widens_and_asks_for_find_docs(self):
        first, again = probe(SUPPORT_Q)["answer"], probe(SUPPORT_Q, last_answered=SUPPORT_Q)["answer"]
        self.assertIn("asked this again", again)
        self.assertIn("find_docs", again)
        self.assertGreater(again.count("["), first.count("["), "a repeat quotes more documents")
        self.assertIn("[support:holidays]", again)

    def test_a_reworded_question_is_not_a_repeat(self):
        answer = probe(SUPPORT_Q, last_answered="Which records can we export?")["answer"]
        self.assertNotIn("asked this again", answer)

    def test_same_question_ignores_case_and_punctuation(self):
        self.assertTrue(tools._same_question("When can we reach support, and how?",
                                             "when can we reach SUPPORT and how"))
        self.assertFalse(tools._same_question("", ""))


class SentenceSplitting(unittest.TestCase):

    def split(self, text):
        return tools.SENTENCE_END.split(text)

    def test_times_stay_in_one_sentence(self):
        text = ("Email support is available on every plan on weekdays from 8 a.m. to 6 p.m. Eastern time, "
                "with a reply within one business day. Phone support is included only with Premium.")
        self.assertEqual(len(self.split(text)), 2)
        self.assertIn("6 p.m. Eastern time", self.split(text)[0])

    def test_abbreviations_stay_in_one_sentence(self):
        self.assertEqual(len(self.split("Use a file, e.g. CSV. Then import it, i.e. Once.")), 2)

    def test_ordinary_sentences_split(self):
        self.assertEqual(self.split("One. Two! Three? \"Four.\""), ["One.", "Two!", "Three?", "\"Four.\""])

    def test_no_key_sentence_is_cut_at_a_time(self):
        for question in ("support hours email", SUPPORT_Q):
            for sentence in probe(question)["sentences"]:
                self.assertFalse(sentence.endswith(("a.m.", "p.m.")), sentence)


class QuestionMemory(unittest.TestCase):
    """last_answered: what a message answered, kept by record_lead, so a repeat can be spotted."""

    def campaign(self):
        c = Campaign([Customer("lead702", ["scheduling", "reminders", "location_reporting"], {1: 9800},
                               ready=11, approved=10000)], day=11)
        c.discover("lead702")
        c.state["leads"]["lead702"]["open_question"] = SUPPORT_Q
        return c

    def test_offer_moves_the_question_to_last_answered(self):
        c = self.campaign()
        c.customers["lead702"].caps[11] = 9000          # declines, so the conversation goes on
        tools.offer(c, customer_id="lead702", product_id="premium", price_cents=9800, message="Premium.")
        lead = c.state["leads"]["lead702"]
        self.assertEqual((lead["open_question"], lead["last_answered"]), ("", SUPPORT_Q))

    def test_send_message_moves_the_question_to_last_answered(self):
        c = self.campaign()
        tools.send_message(c, customer_id="lead702", message="Support is...")
        self.assertEqual(c.state["leads"]["lead702"]["last_answered"], SUPPORT_Q)

    def test_record_lead_keeps_last_answered(self):
        c = self.campaign()
        tools.send_message(c, customer_id="lead702", message="Support is...")
        c.record("lead702")
        self.assertEqual(c.state["leads"]["lead702"]["last_answered"], SUPPORT_Q)

    def test_the_next_offer_after_a_repeat_says_so(self):
        """Driftwood on day 11: the same question comes back after a message answered it."""
        c = self.campaign()
        tools.send_message(c, customer_id="lead702", message="Support is...")
        tools.record_lead(c, customer_id="lead702", needs=["scheduling", "reminders", "location_reporting"],
                          nice_to_haves=[], integrations=["CalendarLink"], budget_cents=9800,
                          max_budget_cents=10000, ready_day=11, scope_final=True, pending_features=[], open_question=SUPPORT_Q,
                          unsellable_reason="", notes="")
        own = items_for(queue(c), "lead702")
        self.assertEqual(len(own), 1)
        self.assertTrue(own[0]["do"].startswith("offer lead702 (product_id='premium', price_cents=9800)"))
        self.assertIn("asked this again", own[0]["do"])
        self.assertIn("find_docs", own[0]["do"])

    def test_a_new_question_after_an_answer_is_not_a_repeat(self):
        c = self.campaign()
        tools.send_message(c, customer_id="lead702", message="Support is...")
        tools.record_lead(c, customer_id="lead702", needs=["scheduling", "reminders", "location_reporting"],
                          nice_to_haves=[], integrations=["CalendarLink"], budget_cents=9800,
                          max_budget_cents=10000, ready_day=11, scope_final=True, pending_features=[],
                          open_question="Which records can we export?", unsellable_reason="", notes="")
        own = items_for(queue(c), "lead702")
        self.assertNotIn("asked this again", own[0]["do"])


class ReaskRanking(unittest.TestCase):
    """Spruce: a DECISION DAY re-ask is a due callback and ranks with them."""

    SPRUCE = lead(needs=["scheduling", "reminders"], integrations=["OtherSuite"], budget_cents=6000,
                  max_budget_cents=6000, ready_day=6, scope_final=False)

    def test_reask_is_listed_with_due_callbacks(self):
        ctx = FakeCtx(day=6)
        for i in range(3):
            ctx.add(f"lock{i}", ANCHOR_LIKE, contacted=2)
        ctx.add("final", lead(needs=["scheduling", "reminders"], budget_cents=3500, max_budget_cents=3500,
                              ready_day=6), contacted=4)
        ctx.add("lead599", self.SPRUCE, contacted=1, callback_day=6)
        items = queue(ctx)["items"]
        self.assertEqual(items[0]["cid"], "lead599", [i["text"] for i in items[:5]])
        self.assertEqual(items[0]["kind"], "reask")

    def test_reask_fits_in_a_short_day(self):
        ctx = FakeCtx(day=6, minutes=10)
        for i in range(5):
            ctx.add(f"lock{i}", ANCHOR_LIKE, contacted=2)
        ctx.add("lead599", self.SPRUCE, contacted=1, callback_day=6)
        self.assertIn("lead599", [i["cid"] for i in queue(ctx)["items"][:2]])

    def test_reask_still_comes_after_recording_a_reply(self):
        ctx = FakeCtx(day=6)
        ctx.add("lead599", self.SPRUCE, contacted=1, callback_day=6)
        ctx.add("reply", contacted=6, last_role="customer")             # an unrecorded reply: priority 0
        self.assertEqual(queue(ctx)["items"][0]["kind"], "record")

    def test_budget_blocked_extra_contact_stays_last(self):
        ctx = FakeCtx(day=11)
        ctx.add("lead956", JUNIPER_CLOSED, contacted=10)
        ctx.add("lead599", self.SPRUCE | {"ready_day": 11}, contacted=1, callback_day=11)
        self.assertEqual([i["cid"] for i in queue(ctx)["items"]], ["lead599", "lead956"])


if __name__ == "__main__":
    unittest.main()
