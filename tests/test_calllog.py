"""glint.calllog: a recorded call links back to its turn in a saved run, and token ids survive packing."""
import json
import types

from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.utils.llm_utils import to_litellm_messages

from glint.calllog import key_for_turn, load, pack, recording, unpack


def test_pack_round_trip():
    ids = [0, 1, 200001, 2**31 - 1, 7] * 1000
    assert unpack(pack(ids)) == ids


def test_recorded_call_matches_its_turn_in_the_saved_run(tmp_path):
    history = [UserMessage(role="user", content="cancel order #W1"),
               AssistantMessage(role="assistant", content=None,
                                tool_calls=[ToolCall(id="c1", name="get_order_details", arguments={"order_id": "#W1"})]),
               ToolMessage(id="c1", role="tool", content='{"status": "pending"}')]
    system = SystemMessage(role="system", content="policy")

    def fake_completion(**kwargs):
        choice = {"message": {"content": "ok"}, "provider_specific_fields": {"token_ids": [5, 6]},
                  "logprobs": {"content": [{"token": "o"}, {"token": "k"}]}}
        return types.SimpleNamespace(model_dump=lambda: {"prompt_token_ids": [1, 2, 3], "choices": [choice]})

    path = tmp_path / "calls.jsonl"
    call = recording(fake_completion, path)
    call(model="openai/Muse-Glimmer-30B-modal", messages=to_litellm_messages([system, *history]))
    call(model="openai/muse-spark-1.3-contributor", messages=[{"role": "user", "content": "x"}])  # not Glimmer
    # the saved run, read back from JSON as a later analysis would
    saved = SimulationRun.model_validate_json(json.dumps({
        "id": "s", "task_id": "t", "trial": 0, "start_time": "2026-10-27T00:00:00", "end_time": "2026-10-27T00:01:00",
        "duration": 60.0, "termination_reason": "user_stop", "messages": [m.model_dump() for m in history]}))
    rows = load(path)
    assert len(rows) == 1
    row = rows[key_for_turn(saved.messages)]
    assert unpack(row["prompt_ids"]) == [1, 2, 3] and unpack(row["completion_ids"]) == [5, 6]
    assert row["logprob_tokens"] == ["o", "k"]


def test_a_vllm_response_without_token_ids_is_a_named_glint_error(tmp_path):
    import pytest

    from glint.calllog import CallLogError, recording

    def completion(**kwargs):
        return {"choices": [{"message": {"content": "hi"}}]}  # no prompt_token_ids, no token_ids

    wrapped = recording(completion, tmp_path / "calls.jsonl", require_ids=True)
    with pytest.raises(CallLogError, match="no token ids"):
        wrapped(model="openai/Muse-Glimmer-30B-modal", messages=[{"role": "user", "content": "x"}])
    # a hosted API returns no ids: without require_ids the call is recorded as before
    recording(completion, tmp_path / "calls.jsonl")(model="openai/Muse-Glimmer-30B", messages=[])
    assert (tmp_path / "calls.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_any_other_recording_failure_is_a_named_glint_error(tmp_path):
    import pytest

    from glint.calllog import CallLogError

    path = tmp_path / "calls.jsonl"
    wrapped = recording(lambda **kwargs: object(), path)  # a response that cannot be read as a dict
    with pytest.raises(CallLogError, match="recording a openai/Muse-Glimmer-30B-modal call failed: TypeError") as e:
        wrapped(model="openai/Muse-Glimmer-30B-modal", messages=[])
    assert str(e.value).startswith("CallLogError: ")  # runlog charges it to Glint by this prefix
    assert isinstance(e.value.__cause__, TypeError)
    assert not path.exists() or path.read_text(encoding="utf-8") == ""  # nothing half-written


def test_load_keeps_the_last_call_of_a_retried_request(tmp_path, capsys):
    def completion(**kwargs):
        return {"prompt_token_ids": [1], "choices": [{"token_ids": [kwargs["attempt"]]}]}

    path = tmp_path / "calls.jsonl"
    call = recording(completion, path)
    msgs = [{"role": "user", "content": "x"}]
    for attempt in (1, 2, 3):
        call(model="openai/Muse-Glimmer-30B", messages=msgs, attempt=attempt)
    call(model="openai/Muse-Glimmer-30B", messages=[{"role": "user", "content": "y"}], attempt=9)
    rows = load(path)
    assert len(rows) == 2
    assert [unpack(r["completion_ids"]) for r in rows.values()] == [[3], [9]]  # the last retry wins
    assert "2 of 4 recorded calls repeat an earlier request" in capsys.readouterr().err
