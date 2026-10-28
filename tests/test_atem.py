import json
import types

import pytest

from glint.atem import AtemParseError, parse_turn, reparse_response, reparsing

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


def response(text, host_calls, content="host content"):
    """A LiteLLM-shaped response: the host's parsed tool calls, plus the generated text as a logprob token stream (cut
    into short tokens, as the server returns it)."""
    tokens = [{"token": text[i:i + 7]} for i in range(0, len(text), 7)]
    message = types.SimpleNamespace(content=content, reasoning_content=None, tool_calls=[
        types.SimpleNamespace(id=f"call_{i}", function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(host_calls)])
    choice = types.SimpleNamespace(message=message, model_dump=lambda: {"logprobs": {"content": tokens}})
    return types.SimpleNamespace(choices=[choice])


ZIP_TURN = " to=self<|message|>Look up the user.<|eom|>" + call(
    "find_user_id_by_name_zip", {"first_name": "Chen", "last_name": "Silva", "zip": "02134"})


def test_reparsing_rewrites_arguments_from_the_generated_text():
    # the hosted parser json.loads every value: "02134" arrived as the number 2134
    host = response(ZIP_TURN, [("find_user_id_by_name_zip", {"first_name": "Chen", "last_name": "Silva", "zip": 2134})])
    out = reparsing(lambda **kw: host)(model="m", messages=[], tools=TOOLS)
    (tc,) = out.choices[0].message.tool_calls
    assert tc.id == "call_0"  # the host's call id is kept
    assert json.loads(tc.function.arguments) == {"first_name": "Chen", "last_name": "Silva", "zip": "02134"}
    assert out.choices[0].message.reasoning_content == "Look up the user."
    assert out.choices[0].message.content is None  # no message to the user in this turn


def test_reparsing_leaves_a_call_without_tools_alone():
    host = response("", [])  # no token stream: re-parsing it would raise
    assert reparsing(lambda **kw: host)(model="m", messages=[]).choices[0].message.content == "host content"


def test_a_response_that_cannot_be_reparsed_raises():
    with pytest.raises(AtemParseError, match="no logprob token stream"):
        reparse_response(response("", [("find_user_id_by_name_zip", {"zip": 1})]), TOOLS)
    # the host found a call the generated text does not hold
    with pytest.raises(AtemParseError, match="host calls"):
        reparse_response(response(" to=user<|message|>Done.<|eot|>", [("find_user_id_by_name_zip", {})]), TOOLS)
    nameless = '<|start|>assistant to=x<|message|><atem:invoke><atem:parameter name="a">1</atem:parameter></atem:invoke>'
    with pytest.raises(AtemParseError, match="invoke without a name"):
        reparse_response(response(nameless, [("x", {})]), TOOLS)
    assert str(AtemParseError("x")).startswith("AtemParseError: ")  # runlog charges it to Glint by this prefix


def test_doubled_generated_name_matches_a_normalized_host_name():
    text = call("find_user_id_by_name_zip.find_user_id_by_name_zip", {"zip": "02134"})
    host = response(text, [("find_user_id_by_name_zip", {"zip": 2134})])
    reparse_response(host, TOOLS)
    assert json.loads(host.choices[0].message.tool_calls[0].function.arguments) == {"zip": "02134"}


@pytest.mark.xfail(strict=True, reason="BUG: reparse_response compares the host's raw call name with the normalized "
                   "generated name, so a doubled name the host passes through unnormalized raises AtemParseError")
def test_doubled_name_from_both_host_and_text_is_not_a_parse_error():
    text = call("find_user_id_by_name_zip.find_user_id_by_name_zip", {"zip": "02134"})
    host = response(text, [("find_user_id_by_name_zip.find_user_id_by_name_zip", {"zip": 2134})])
    reparse_response(host, TOOLS)
    assert json.loads(host.choices[0].message.tool_calls[0].function.arguments) == {"zip": "02134"}
