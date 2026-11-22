# Business operations

Your tools reach the company only through these 14 operations. The supplied
`tools.py` has one basic tool for each, with the same name and arguments.

Inside a tool, `ctx.<operation>(...)` returns the data shown below. If the
company's rules refuse, it raises an error that reaches the model as
`{"error": {"code": ..., "message": ...}}`. To handle a refusal inside your tool,
use `ctx.call("<operation>", ...)`: it returns `{"ok": true, "data": ...}` or
`{"ok": false, "error": ...}`.

- Money is in integer US cents: `6000` is $60.00.
- Reads are free. `talk_to_customer` and `present_quote` each use one message of
  contact time and cost $0.50.
- Use the IDs the operations return; don't build them yourself.
- There is no `advance_day`: the day ends when the model replies without calling
  a tool.

The examples are real outputs from the practice case `requirements_update`
(customer `nimbus`), in the order shown.

| Operation | Arguments |
| --- | --- |
| [`get_status`](#get_status) | — |
| [`list_customers`](#list_customers) | — |
| [`read_customer`](#read_customer) | customer_id: str |
| [`read_customer_history`](#read_customer_history) | customer_id: str |
| [`search_customer_history`](#search_customer_history) | customer_id: str, query: str, kind: str |
| [`read_record`](#read_record) | record_id: str |
| [`get_catalog`](#get_catalog) | — |
| [`list_documents`](#list_documents) | — |
| [`read_document`](#read_document) | document_id: str |
| [`save_note`](#save_note) | customer_id: str, text: str, source_ids: list[str] |
| [`schedule_callback`](#schedule_callback) | customer_id: str, day: int, reason: str |
| [`talk_to_customer`](#talk_to_customer) | customer_id: str, message: str |
| [`create_quote`](#create_quote) | customer_id: str, product_id: str, monthly_price_cents: int |
| [`present_quote`](#present_quote) | quote_id: str, message: str |

## get_status

Read the campaign's current day, remaining time, and progress.

`ctx.get_status()` · Data excerpt before any contact:

```json
{
  "day": 1,
  "date": "2026-10-01",
  "campaign_days": 1,
  "finished": false,
  "minutes_remaining_today": 120,
  "minutes_per_exchange": 5,
  "cost_per_exchange_cents": 50,
  "contact_cost_cents": 0,
  "booked_revenue_cents": 0,
  "orders": 0
}
```

`orders` is a count; booked revenue is not settled profit. Callbacks are listed on
each account (`read_customer`), not here. `outcome` appears only when `finished` is
true (see the end of this page).

## list_customers

List public customers, contact eligibility, and last decisions.

`ctx.list_customers()` · The first entry:

```json
{
  "id": "nimbus",
  "name": "Mina Park",
  "business": "Nimbus Music School",
  "background": "A music school that revised its booking needs after changing software.",
  "status": "prospect",
  "contact": {"eligible": true, "reason": null, "available_day": 1, "last_contact_day": null, "last_decision": null}
}
```

Pass `id` as `customer_id`. `status` is `prospect`, `purchased` or `opted_out`.
`contact.eligible` says whether contact is allowed now; `reason` is null or one of
`campaign_finished`, `already_purchased`, `opted_out`, `call_ended`, `day_full`.
`available_day` is the earliest campaign day a contact is allowed, or null.

## read_customer

Read compact account state: contact eligibility, profile reference, latest message/note, pending callbacks, quotes, and orders. Use search_customer_history and read_record for sources, or read_customer_history for the complete history.

`ctx.read_customer(customer_id='nimbus')` · Data excerpt before any action (`customer` and `contact` omitted):

```json
{
  "profile_id": "crm:profile:nimbus",
  "record_counts": {"interaction": 2, "profile": 1},
  "status": "prospect",
  "last_message": null,
  "latest_note": null,
  "pending_callbacks": [],
  "quotes": [],
  "orders": []
}
```

`profile_id` is an ID for `read_record`, or null when the account has no profile.
`record_counts` counts records by kind. The lists hold callbacks, quotes and orders in
the shapes shown further down. A quote has no status field: a draft has
`issued_day: null`; a presented quote is valid through `expires_day`; an accepted one
also appears in `orders`. Unknown customer: `not_found`.

## read_customer_history

Read an account's full current conversation, notes, callbacks, profile, and record index. No history is truncated.

`ctx.read_customer_history(customer_id='nimbus')` · Data excerpt before any action:

```json
{
  "history": [],
  "notes": [],
  "callbacks": [],
  "record_index": [{"id": "crm:interaction:nimbus:earlier", "customer_id": "nimbus", "kind": "interaction", "date": "2026-08-10", "source": "Synthetic public practice CRM", "source_ids": []}, "… one entry per record"]
}
```

Everything in `read_customer`, plus `profile`, `record_index`, `history` (this
campaign's messages), `notes` (your saved notes) and `callbacks` (all statuses).
Empty lists do not mean there are no earlier CRM records: `record_index` lists them;
read each with `read_record`. Nothing is truncated, so this output can be long.

## search_customer_history

Search one customer's dated CRM records and agent notes. An empty query browses all records. Search matches words in IDs and content; exact IDs and matches to all words rank first, followed by partial matches. Set kind to all, profile, sale, refund, cancellation, interaction, quote, callback, or note. Read full records with read_record before relying on an excerpt.

`ctx.search_customer_history(customer_id='nimbus', query='revision', kind='interaction')` · Complete data:

```json
[
  {"id": "crm:interaction:nimbus:revision", "customer_id": "nimbus", "kind": "interaction", "date": "2026-09-29", "source": "Synthetic public practice CRM", "source_ids": [], "excerpt": "call\ncustomer\nOur earlier requirements are obsolete. We moved to BookFlow and need automated reminders. We can spend $60 per month.", "excerpt_is_partial": false}
]
```

`kind` is one of `all`, `profile`, `sale`, `refund`, `cancellation`, `interaction`,
`quote`, `callback`, `note`; `query=""` browses. No match returns `[]`; try fewer
words. An excerpt is a preview: read the record before relying on it.
Unknown customer: `not_found`; unsupported kind: `invalid_kind`.

## read_record

Read a complete CRM record by its returned canonical ID (crm:... or run:...). Includes date, authorship, and source references. Unambiguous legacy IDs also work.

`ctx.read_record(record_id='crm:profile:nimbus')` · Complete data:

```json
{
  "id": "crm:profile:nimbus",
  "customer_id": "nimbus",
  "kind": "profile",
  "date": "2026-09-30",
  "source": "Synthetic public practice CRM",
  "source_ids": [],
  "data": {"name": "Mina Park", "business": "Nimbus Music School", "background": "A music school that revised its booking needs after changing software."}
}
```

Every record has these seven outer fields; `data` depends on `kind`. `date` and
`source` give provenance. Notes saved by the agent are interpretations
(`authoritative: false`). Unknown record: `not_found`; ambiguous legacy ID:
`ambiguous_reference`.

## get_catalog

Read the products (features, integrations, list price and service cost in integer cents) and today's pricing policy, with source references. Legal price ranges are not precomputed: the policy states the rule.

`ctx.get_catalog()` · Data excerpt: today's policy and the Pro product (`products[1]`):

```json
{
  "day": 1,
  "current_pricing_policy": {"id": "launch", "effective_day": 1, "max_discount_percent": 15, "quote_valid_days": 3, "rule": "A quote's monthly price is a whole number of cents, at most the product's list price and at least the list price less the maximum discount (rounded up to the next cent), and never below the product's monthly service cost.", "source_id": "policy:launch"},
  "products": [{"id": "pro", "name": "Pro", "monthly_price_cents": 6000, "monthly_service_cost_cents": 2000, "features": ["scheduling", "reminders"], "integrations": ["CalendarLink", "BookFlow"], "source_id": "product:pro"}, "… one entry per product"]
}
```

`monthly_price_cents` is the list price, the highest legal price. The lowest is not
listed: `current_pricing_policy.rule` states how to compute it. With 15% off, a
6000-cent list price allows 5100 to 6000 cents, unless the service cost is higher. The
range changes when the pricing policy changes. Service cost is the company's cost for
every accepted order, refunded or not. Every `source_id` can be read with
`read_document`.

## list_documents

List the ID and title of every company document available today. Read a document's full text with read_document.

`ctx.list_documents()` · Complete data:

```json
[
  {"id": "migration:contacts", "title": "Importing contacts and history", "available_day": 1},
  {"id": "product:basic", "title": "Basic", "available_day": 1},
  {"id": "product:pro", "title": "Pro", "available_day": 1},
  {"id": "policy:launch", "title": "Pricing policy launch", "available_day": 1}
]
```

Every document available today, including one per product (`product:<id>`) and
one per pricing policy (`policy:<id>`). There is no search: read what you need with
`read_document`. Documents that become available later appear on their day.

## read_document

Read a product or policy document by its ID.

`ctx.read_document(document_id='migration:contacts')` · Complete data:

```json
{
  "id": "migration:contacts",
  "title": "Importing contacts and history",
  "text": "Customer contact records can be imported from CSV. Historical reminder delivery logs cannot be imported. This limit applies to Basic and Pro.",
  "available_day": 1
}
```

`text` is the whole document. Product and pricing sources from `get_catalog` work
too. Unknown or not yet available: `not_found`.

## save_note

Save a note for one customer, with IDs of supporting records. Use an empty source_ids list when there are no source references. Notes are the seller's interpretation, not authoritative quotes or orders.

`ctx.save_note(customer_id='nimbus', text='Check the revised requirements before quoting.', source_ids=['crm:interaction:nimbus:revision'])` · Complete data:

```json
{
  "id": "run:note:N0001",
  "day": 1,
  "customer_id": "nimbus",
  "text": "Check the revised requirements before quoting.",
  "source_ids": ["crm:interaction:nimbus:revision"],
  "author": "sales_agent"
}
```

Adds a note without contacting anyone; it does not change authoritative facts.
`source_ids` may be empty; each must belong to this customer or be an available
document. Errors: `not_found`, `empty_text`, `invalid_source`.

## schedule_callback

Record a callback for today or a later campaign day (integers from 1). The next contact on that day completes the appointment, not its substantive purpose. Postponing an already-due callback preserves it as missed.

`ctx.schedule_callback(customer_id='nimbus', day=1, reason='Confirm the revised requirements.')` · Complete data:

```json
{
  "id": "run:callback:CB0001",
  "customer_id": "nimbus",
  "created_day": 1,
  "day": 1,
  "reason": "Confirm the revised requirements.",
  "status": "pending",
  "updated_day": 1,
  "contacted_day": null,
  "contact_record_id": null,
  "completion_basis": "contact_only"
}
```

`day` must be today or a later campaign day. Contact on the due day marks it
`completed`, which proves contact, not that a promise was kept. Other statuses:
`missed`, `rescheduled`, `canceled`. A due callback becomes `missed` when its day ends
without contact. Errors: `invalid_day`, `empty_text`, `opted_out`,
`already_purchased`, `not_found`.

## talk_to_customer

Send a message and receive the customer's reply. This consumes contact time. Use present_quote to make a formal offer that can become an order.

`ctx.talk_to_customer(customer_id='nimbus', message='Which requirements should I use for the quote?')` · Complete data:

```json
{
  "reply": {"id": "run:interaction:M0002", "day": 1, "customer_id": "nimbus", "role": "customer", "text": "We now use BookFlow and need reminders, with a $60 monthly budget.", "quote_id": null, "decision": "continue"},
  "decision": "continue",
  "order": null
}
```

Uses one exchange of contact time and cost. Replies `continue`, `decline`,
`end_call` or `opt_out`; only `present_quote` can be accepted. A tool that talks to a
customer must return right after, so the model reads the reply. Errors: `not_found`,
`empty_text`, `opted_out`, `already_purchased`, `call_ended`, `day_full`.

## create_quote

Create a quote under the current pricing rules. Price is an integer number of US cents per month: 5500 means $55.00. Creating a quote does not send it or reserve its price. First presentation checks current prices and starts validity. The price must be legal under today's pricing policy (see get_catalog).

`ctx.create_quote(customer_id='nimbus', product_id='pro', monthly_price_cents=6000)` · Complete data:

```json
{
  "id": "run:quote:Q0001",
  "customer_id": "nimbus",
  "product_id": "pro",
  "monthly_price_cents": 6000,
  "monthly_service_cost_cents": 2000,
  "created_day": 1,
  "expires_day": null,
  "pricing_policy_id": "launch",
  "issued_day": null,
  "updated_day": 1
}
```

A draft: it is not sent and does not reserve the price. `issued_day` and
`expires_day` are set when the quote is first presented. Price outside today's legal
range: `invalid_price` (the message does not say what the range is); unknown product or
customer: `not_found`.

## present_quote

Present a quote using its returned canonical ID with a message and receive the customer's decision. This consumes contact time. Only a confirmed acceptance creates an order. Issued, unexpired offers retain their terms; an unseen draft must meet current prices when first presented.

`ctx.present_quote(quote_id='run:quote:Q0001', message='Pro includes reminders and BookFlow integration at $60 per month.')` · Data excerpt (`reply` and `order.quote` omitted):

```json
{
  "decision": "accept",
  "order": {"id": "run:sale:O0001", "accepted_day": 1, "updated_day": 1, "settlement": null}
}
```

Uses one exchange. The first presentation checks today's prices and starts the
quote's validity; an issued, unexpired quote keeps its terms through pricing changes.
Only an acceptance creates an order (one per customer). The order holds the complete
quote; `settlement` stays null until the campaign settles. Errors include
`stale_draft`, `expired_quote`, `not_found`, `invalid_reference` and the contact
errors of `talk_to_customer`. Return right after, like `talk_to_customer`.

## How a campaign ends

After the last day the harness settles every order: an order is refunded if the product does not
meet the customer's actual requirements, and its service cost is still paid. `get_status` then
shows `finished: true` and this `outcome` (the example campaign above, after one day):

```json
{
  "accepted_orders": 1,
  "retained_orders": 1,
  "booked_revenue_cents": 6000,
  "refunded_revenue_cents": 0,
  "retained_revenue_cents": 6000,
  "service_cost_cents": 2000,
  "contact_cost_cents": 100,
  "profit_cents": 3900,
  "missed_callbacks": 0
}
```

`profit_cents` = retained revenue − service cost of every accepted order − contact costs.
API costs are reported separately by the runner.
