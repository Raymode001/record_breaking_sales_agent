"""The agent loop: a language model using tools, one day at a time. Read it; don't change it.

Each day the model is called until it replies without asking for a tool: that reply
ends its day. The harness, not the model, then moves the calendar and opens the next
day with a short note. After the last day the campaign settles by itself.
"""


def run_agent(model, prompt, toolbox, world, max_turns_per_day=100):
    messages = [{"role": "developer", "content": prompt},
                {"role": "user", "content": world.morning_note()}]

    while not world.finished:
        for turn in range(max_turns_per_day):
            reply = model(messages, toolbox.schemas())       # the only call to the LLM
            messages += reply.items                          # what it decided, and why
            if not reply.tool_calls:                         # asked for nothing: its day is over
                break
            for call in reply.tool_calls:
                output = toolbox.run(call.name, call.arguments, world, call.id)   # errors come back as text
                messages.append({"type": "function_call_output",
                                 "call_id": call.id, "output": output})
        world.end_day(reply.text, cut_off=bool(reply.tool_calls))    # the harness owns the calendar
        if not world.finished:
            messages.append({"role": "user", "content": world.morning_note()})

    return world.outcome(), messages
