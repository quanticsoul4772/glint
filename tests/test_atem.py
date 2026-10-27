from glint.atem import parse_turn

TOOLS = [
    {"type": "function", "function": {"name": "find_user_id_by_name_zip", "parameters": {"type": "object", "properties": {
        "first_name": {"type": "string"}, "last_name": {"type": "string"}, "zip": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "return_items", "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
        "quantity": {"type": "integer"}}}}},
]


def call(name, params, end="<|eot|>"):
    body = "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params.items())
    return (f"<|start|>assistant to={name}<|message|><atem:function_calls>\n<atem:invoke name=\"{name}\">\n"
            f"{body}</atem:invoke>\n</atem:function_calls>{end}")


def test_schema_string_stays_string_including_leading_zero():
    text = " to=self<|message|>Look up the user.<|eom|>" + call(
        "find_user_id_by_name_zip", {"first_name": "Chen", "last_name": "Silva", "zip": "02134"})
    turn = parse_turn(text, TOOLS)
    assert turn.reasoning == "Look up the user."
    assert turn.tool_calls[0].arguments == {"first_name": "Chen", "last_name": "Silva", "zip": "02134"}


def test_non_string_types_are_json_decoded():
    turn = parse_turn(call("return_items", {"order_id": "#W123", "item_ids": '["1", "2"]', "quantity": "2"}), TOOLS)
    assert turn.tool_calls[0].arguments == {"order_id": "#W123", "item_ids": ["1", "2"], "quantity": 2}


def test_invoke_echoed_in_reasoning_or_answer_is_not_a_call():
    echo = '<atem:function_calls><atem:invoke name="return_items"></atem:invoke></atem:function_calls>'
    text = f" to=self<|message|>maybe {echo}<|eom|><|start|>assistant to=user<|message|>I won't {echo}<|eot|>"
    turn = parse_turn(text, TOOLS)
    assert turn.tool_calls == []
    assert turn.content.startswith("I won't")


def test_reasoning_missing_eom_does_not_swallow_the_call():
    text = " to=self<|message|>thinking" + call("find_user_id_by_name_zip", {"zip": "80273"})
    turn = parse_turn(text, TOOLS)
    assert turn.reasoning == "thinking"
    assert turn.tool_calls[0].arguments == {"zip": "80273"}


def test_doubled_name_collapses():
    doubled = call("find_user_id_by_name_zip.find_user_id_by_name_zip", {"zip": "1"})
    assert parse_turn(doubled, TOOLS).tool_calls[0].name == "find_user_id_by_name_zip"


def test_unregistered_name_passes_through_for_the_environment_to_reject():
    # 2026-10-27, banking task_038 on Modal: Glimmer called a hidden (discoverable) tool directly by name. Raising
    # ended the run as an infrastructure error; τ² should answer it as an unknown tool instead.
    turn = parse_turn(call("get_user_dispute_history_7291", {"user_id": "890389b165"}), TOOLS)
    # no schema, so values are JSON-decoded where they parse ("7" would become 7) and kept as text otherwise
    assert [(c.name, c.arguments) for c in turn.tool_calls] == [("get_user_dispute_history_7291",
                                                                  {"user_id": "890389b165"})]
    assert parse_turn(call("x_1.x_1", {}), TOOLS).tool_calls[0].name == "x_1"


def test_recorded_deepinfra_turn():
    # verbatim tail of a turn recorded from DeepInfra on 2026-10-26 (task 48); DeepInfra parsed zip as int 80273
    text = (" to=self<|message|> via email or name+zip. I'll call find_user_id_by_name_zip.<|eom|><|start|>assistant "
            "to=find_user_id_by_name_zip<|message|><atem:function_calls>\n<atem:invoke name=\"find_user_id_by_name_zip\">"
            "\n<atem:parameter name=\"first_name\">Daiki</atem:parameter>\n<atem:parameter name=\"last_name\">Johnson"
            "</atem:parameter>\n<atem:parameter name=\"zip\">80273</atem:parameter>\n</atem:invoke>\n"
            "</atem:function_calls><|eot|>")
    assert parse_turn(text, TOOLS).tool_calls[0].arguments["zip"] == "80273"
