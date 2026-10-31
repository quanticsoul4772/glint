"""glint.calllog: a recorded call links back to its turn in a saved run and to its task and trial, with its token
counts; token ids and logprobs are not kept (decision 2026-10-30)."""
import json
import types

import pytest
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.utils.llm_utils import to_litellm_messages

from glint.calllog import CallLogError, key_for_turn, load, recording


def test_recorded_call_matches_its_turn_and_names_its_run(tmp_path):
    history = [UserMessage(role="user", content="cancel order #W1"),
               AssistantMessage(role="assistant", content=None,
                                tool_calls=[ToolCall(id="c1", name="get_order_details", arguments={"order_id": "#W1"})]),
               ToolMessage(id="c1", role="tool", content='{"status": "pending"}')]
    system = SystemMessage(role="system", content="policy")

    def fake_completion(**kwargs):  # a vLLM response: usage, plus token ids and logprobs that are not kept
        choice = {"message": {"content": "ok"}, "provider_specific_fields": {"token_ids": [5, 6]},
                  "logprobs": {"content": [{"token": "o"}, {"token": "k"}]}}
        return types.SimpleNamespace(model_dump=lambda: {"prompt_token_ids": [1, 2, 3], "choices": [choice],
                                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    path = tmp_path / "calls.jsonl"
    call = recording(fake_completion, path, context=lambda: ("task_041", "2"))
    call(model="openai/Muse-Glimmer-30B-modal", messages=to_litellm_messages([system, *history]))
    call(model="openai/muse-spark-1.3-contributor", messages=[{"role": "user", "content": "x"}])  # not Glimmer
    saved = SimulationRun.model_validate_json(json.dumps({
        "id": "s", "task_id": "t", "trial": 0, "start_time": "2026-10-27T00:00:00", "end_time": "2026-10-27T00:01:00",
        "duration": 60.0, "termination_reason": "user_stop", "messages": [m.model_dump() for m in history]}))
    rows = load(path)
    assert len(rows) == 1
    row = rows[key_for_turn(saved.messages)]
    assert (row["task_id"], row["trial"], row["prompt_tokens"], row["completion_tokens"]) == ("task_041", "2", 3, 2)
    assert not {"prompt_ids", "completion_ids", "logprob_tokens"} & set(row)  # no longer kept


def test_counts_fall_back_to_token_ids_and_a_missing_context_is_unknown(tmp_path):
    def completion(**kwargs):  # no usage block: count the token ids vLLM returned
        return {"prompt_token_ids": [1, 2, 3, 4], "choices": [{"token_ids": [9]}]}

    path = tmp_path / "calls.jsonl"
    recording(completion, path)(model="openai/Muse-Glimmer-30B", messages=[])
    row = json.loads(path.read_text(encoding="utf-8"))
    assert (row["prompt_tokens"], row["completion_tokens"], row["task_id"], row["trial"]) == (4, 1, "unknown", "unknown")


def test_any_recording_failure_is_a_named_glint_error(tmp_path):
    path = tmp_path / "calls.jsonl"
    wrapped = recording(lambda **kwargs: object(), path)  # a response that cannot be read as a dict
    with pytest.raises(CallLogError, match="recording a openai/Muse-Glimmer-30B-modal call failed: TypeError") as e:
        wrapped(model="openai/Muse-Glimmer-30B-modal", messages=[])
    assert str(e.value).startswith("CallLogError: ")  # runlog charges it to Glint by this prefix
    assert isinstance(e.value.__cause__, TypeError)
    assert not path.exists() or path.read_text(encoding="utf-8") == ""  # nothing half-written


def test_load_keeps_the_last_call_of_a_retried_request(tmp_path, capsys):
    def completion(**kwargs):
        return {"choices": [{}], "usage": {"prompt_tokens": 1, "completion_tokens": kwargs["attempt"]}}

    path = tmp_path / "calls.jsonl"
    call = recording(completion, path)
    msgs = [{"role": "user", "content": "x"}]
    for attempt in (1, 2, 3):
        call(model="openai/Muse-Glimmer-30B", messages=msgs, attempt=attempt)
    call(model="openai/Muse-Glimmer-30B", messages=[{"role": "user", "content": "y"}], attempt=9)
    rows = load(path)
    assert len(rows) == 2
    assert [r["completion_tokens"] for r in rows.values()] == [3, 9]  # the last retry wins
    assert "2 of 4 recorded calls repeat an earlier request" in capsys.readouterr().err


def test_a_call_record_carries_the_attempts_simulation_id(tmp_path):
    # 2026-10-30 review: a resumed rerun keeps (task, trial); its records are told apart by τ²'s simulation id
    import types

    def answer(**kwargs):
        return types.SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}],
                                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    path = tmp_path / "calls.jsonl"
    call = recording(answer, path, context=lambda: ("task_041", 2, "sim-uuid"))
    call(model="Muse-Glimmer", messages=[{"role": "user", "content": "hi"}])
    row = json.loads(path.read_text(encoding="utf-8"))
    assert (row["task_id"], row["trial"], row["simulation_id"]) == ("task_041", 2, "sim-uuid")


def test_the_saved_copy_of_a_glimmer_response_is_lean_and_the_live_one_is_whole(tmp_path):
    # 2026-10-31: τ² saves each response's to_dict() in its checkpoint; prompt token ids and logprobs made the unseen
    # tools arm's checkpoint 5.4 GB for 100 runs, and every save rewrote it
    import litellm

    from glint.calllog import lean_saved

    def answer(**kwargs):
        r = litellm.ModelResponse(choices=[{
            "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop",
            "logprobs": {"content": [{"token": "o", "logprob": -0.1, "top_logprobs": []}]},
            "provider_specific_fields": {"token_ids": [5, 6], "other": 1}}])
        r.prompt_token_ids = list(range(1000))  # vLLM's return_token_ids extra, as on the live response
        return r

    path = tmp_path / "calls.jsonl"
    call = lean_saved(recording(answer, path, context=lambda: ("task_041", 2)))
    r = call(model="openai/Muse-Glimmer-30B-modal", messages=[{"role": "user", "content": "hi"}])
    saved = r.to_dict()
    assert "prompt_token_ids" not in saved and "logprobs" not in saved["choices"][0]
    assert saved["choices"][0]["provider_specific_fields"] == {"other": 1}
    assert saved["choices"][0]["message"]["content"] == "ok"
    assert r.choices[0].logprobs is not None and len(r.prompt_token_ids) == 1000  # the live object is whole
    assert json.loads(path.read_text(encoding="utf-8"))["prompt_tokens"] == 1000  # the call log read it first
    # τ²'s message keeps the lean copy
    message = AssistantMessage(role="assistant", content="ok", raw_data=r.to_dict())
    assert "prompt_token_ids" not in json.dumps(message.model_dump())
    other = lean_saved(answer)(model="openai/muse-spark", messages=[])
    assert "prompt_token_ids" in other.to_dict()  # only Glimmer's responses are trimmed
