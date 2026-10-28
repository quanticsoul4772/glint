"""glint.desk.events: the timeline the Bank Desk shows, from τ² messages."""
from glint.desk.events import customer_summary, outcome, timeline


def _call(i, name, args):
    return {"id": str(i), "name": name, "arguments": args}


def test_timeline_orders_customer_agent_verdict_lookup_and_action_events():
    messages = [
        {"role": "user", "content": "I want a higher limit"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [
            _call(1, "check_credit_limit_increase", {"account": "a"}),
            _call(2, "unlock_discoverable_agent_tool", {}),
            _call(3, "call_discoverable_agent_tool", {"agent_tool_name": "approve_credit_limit_increase_5847",
                                                      "arguments": '{"amount": 500}'}),
            _call(4, "get_account", {"id": "a"})]},
        {"role": "tool", "id": "1", "content": '{"verdict": "deny"}'},
        {"role": "tool", "id": "3", "content": "Error: not allowed"},
        {"role": "tool", "id": "4", "content": "x" * 600},
    ]
    events = timeline(messages, {1: {"tool": "approve_credit_limit_increase_5847", "reasons": ["policy denies"]}})
    kinds = [e["kind"] for e in events]
    assert kinds == ["customer", "agent", "verdict", "action", "lookup"]  # the unlock call is not shown
    action = events[3]
    assert action["tool"] == "approve_credit_limit_increase_5847" and action["args"] == {"amount": 500}
    assert action["error"] is True and action["overruled"] == ["policy denies"]
    assert events[4]["result"].endswith(" …") and len(events[4]["result"]) < 410  # long results are cut


def test_timeline_keeps_malformed_inner_arguments_visible():
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        _call(1, "call_discoverable_agent_tool", {"agent_tool_name": "get_x", "arguments": "{not json"})]}]
    (event,) = timeline(messages)
    assert event["tool"] == "get_x" and event["args"] == {"arguments": "{not json"} and event["result"] is None


def test_outcome_reads_reward_and_database_check():
    assert outcome({"reward_info": {"reward": 1.0, "db_check": {"db_match": True}}}) == \
        {"passed": True, "database_matches_gold": True}
    assert outcome({"reward_info": None}) == {"passed": False, "database_matches_gold": None}


def test_customer_summary_shows_situation_and_goal_only():
    brief = ("**Persona:** Tired.\n\n**Your situation:** Your card was **stolen**.\n\n"
             "**Your goal:** Dispute the charge.\n\n**Hidden facts:** PIN 1234")
    assert customer_summary(brief) == "Your card was stolen. Dispute the charge."
    assert customer_summary("**Plain** brief") == "Plain brief"
