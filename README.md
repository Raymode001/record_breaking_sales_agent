# Building a sales agent

## The task

An AI agent sells scheduling software to 40 small businesses over a 12-day campaign. It misses many sales it could make. Make it earn more, on customers it has never met.

You change two files:

- `prompt.py`: the agent's instructions.
- `tools.py`: the agent's tools. It starts with 14 basic tools, one for each company operation. You build better tools on top of them.

Everything else stays fixed: the loop that runs the agent (`agent.py`), the company and its simulated customers. You hand in the two files and five runs of them. Staff then run your two files on campaigns you have not seen; that decides 40% of your grade.

## The campaign

- 40 customer accounts, 12 business days.
- Each day the agent has 60 minutes of contact time. Each message to a customer uses 5 minutes and costs $0.50. Reading records and documents is free.
- A sale happens only when a customer accepts a quote. Its price must follow the current pricing policy, and quotes expire.
- The agent ends its day by replying without calling a tool. After day 12 the campaign closes and the orders are settled.

**Profit = revenue from the orders customers keep − service cost of every order − contact costs**

An order for a plan that doesn't meet the customer's needs is refunded at the end, and its service cost is still paid.

Customers' budgets, needs and decision days are not in the records: the agent has to ask. The company's documents describe its products and policies, including what changes during the campaign.

## Set up

Unzip the homework and open `explore.ipynb` in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**). Follow the notebook for setup, your Parley API key, editing and saving your work.

Paid commands call models through MIT Parley (graduate students get $30 of API credit a month; a 40-account run costs about $0.55).

## Prompt and tools

### 1. Find what goes wrong

Run the starting agent on the 40-account campaign in notebook section 8 (about $0.55 and 5–10 minutes).

Open `inspect.html` in the folder it saves under `outputs/`. It shows where the 40 accounts ended up, how each day's contact time was used, every conversation and every tool call. Find the failures that cost the most profit: accounts never contacted, offers the customer could not accept, promises not kept, tool errors.

### 2. Improve the prompt

Rewrite `SYSTEM_PROMPT` in `prompt.py`. It is everything the agent is told. Keep it general: the evaluation uses customers your agent has never seen. `python run.py prompt` shows exactly what the model reads.

### 3. Build tools

The basic tools return raw data. The catalog gives list prices and a pricing rule, not the prices you may quote. Documents can be listed and read, not searched. Nothing says which customers need attention today. Whatever your tools don't work out, the model has to work out itself, and that is where it makes mistakes.

Build tools that fix the failures you found. For example, a tool can:

- find the documents that answer a customer's question,
- compute the best legal price within a customer's budget,
- hand the agent the next account to work on, and why,
- gather everything needed before a call in one step.

A tool is a Python function marked `@tool`. The model sees its name, its docstring and its arguments. `ctx` gives it the company's operations, listed with real outputs in `OPERATIONS.md`:

```python
@tool
def company_answer(ctx, topic: str) -> list:
    """What company documents say about a topic, such as 'refund' or 'migration'.
    Returns the full text of up to three documents that mention it."""
    words = topic.lower().split()
    found = []
    for listed in ctx.list_documents():
        document = ctx.read_document(document_id=listed["id"])
        text = " ".join([document["id"], document["title"], document["text"]]).lower()
        if any(word in text for word in words):
            found.append({"title": document["title"], "text": document["text"]})
    return found[:3]
```

Try tools for free in `explore.ipynb` before a paid run: it runs them on the 40 accounts and on practice cases. You may use a coding assistant such as Claude Code, Cursor or Codex; you are responsible for the code you hand in.

## Rules and handing in

### Rules for tools

- Tools reach the company only through `ctx`. They don't read the files in the homework folder (the campaign's `customers.json` holds the simulated customers' private profiles) and don't hard-code facts about specific customers.
- Tools don't call a language model.
- A tool that sends a customer a message must return right after, so the agent reads the reply.

To measure a change, run your version on the campaign in notebook section 9 (about $1).

Profit varies from run to run, so compare more than one run, and check in `inspect.html` that the accounts your change targeted behaved differently.

### 4. Make your final runs

Use the final-run cell in notebook section 10.

This runs your prompt and tools on the 40-account campaign five times, one after another (about an hour and $5–8).

### 5. Hand in

Download your submission ZIP using notebook section 10.

Hand in the ZIP it creates. It holds your `prompt.py` and `tools.py` and the five runs: each run's transcript, result and run report. It refuses runs made with other files or settings, so if you change either file, run `python run.py final` again.

## Grading

| Part | Weight |
| --- | ---: |
| Your five submitted runs of the 40-account, 12-day campaign | 60% |
| Staff evaluation of your submitted prompt and tools on held-out campaigns | 40% |

Each run is scored on:

| Measure | Weight |
| --- | ---: |
| Profit | 80% |
| Token efficiency | 20% |
