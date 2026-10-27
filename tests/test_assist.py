"""The decision tools on the real τ² databases, including the cases where Glimmer decided wrong in recorded runs."""
from pathlib import Path

import pytest
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.retail.data_model import RetailDB

from glint.assist import airline, retail

DOMAINS = Path(__file__).resolve().parents[2] / "vendor" / "tau2-bench" / "data" / "tau2" / "domains"


@pytest.fixture(scope="module")
def rdb():
    return RetailDB.load(DOMAINS / "retail" / "db.json")


@pytest.fixture(scope="module")
def adb():
    return FlightDB.load(DOMAINS / "airline" / "db.json")


def test_airline_task7_basic_economy_cancel_has_a_cabin_path(adb):
    # Glimmer refused: "Policy prohibits modifications to Basic Economy flights and cancellation is not allowed".
    # Gold upgrades XEHM4B to business, then cancels it.
    cancel = airline.check_eligibility(adb, "XEHM4B", "cancel", reason="change of plan")
    assert cancel["allowed"] is False and cancel["paths_to_allowed"]
    assert "change_cabin" in cancel["paths_to_allowed"][0]
    assert airline.check_eligibility(adb, "XEHM4B", "change_cabin")["allowed"] is True
    assert airline.check_eligibility(adb, "XEHM4B", "change_flights")["allowed"] is False


def test_airline_fixed_rules(adb):
    assert airline.check_eligibility(adb, "XEHM4B", "remove_baggage")["allowed"] is False
    assert airline.check_eligibility(adb, "XEHM4B", "add_insurance")["allowed"] is False
    assert airline.check_eligibility(adb, "NOPE00", "cancel")["allowed"] is False


def test_retail_task82_refund_only_to_original_or_gift_card(rdb):
    # Glimmer refunded a return to credit_card_1565124; the order was paid with gift_card_7250692.
    res = retail.check_eligibility(rdb, "#W9571698", "return", payment_method_id="credit_card_1565124",
                                   item_ids=["6065192424"])
    assert res["allowed"] is False
    assert res["details"]["allowed_payment_methods"] == ["gift_card_7250692"]
    ok = retail.check_eligibility(rdb, "#W9571698", "return", payment_method_id="gift_card_7250692",
                                  item_ids=["6065192424"])
    assert ok["allowed"] is True


def test_retail_status_rules(rdb):
    assert retail.check_eligibility(rdb, "#W9571698", "cancel")["allowed"] is False      # delivered
    assert retail.check_eligibility(rdb, "#W9911714", "cancel")["allowed"] is True       # pending
    assert retail.check_eligibility(rdb, "#W9911714", "return")["allowed"] is False


def test_retail_task93_find_the_laptop_the_user_described(rdb):
    # The user said the laptop is "15-inch, 32GB"; Glimmer exchanged the 15-inch 16GB one from another order.
    found = retail.find_items(rdb, "lei_wilson_4541", product_name="laptop",
                              options={"screen size": "15-inch", "ram": "32GB"})
    assert [(i["order_id"], i["item_id"]) for i in found["items"]] == [("#W4073673", "2216662955")]


def test_retail_task20_most_expensive_variant(rdb):
    # "Upgrade all items to the most expensive variants"; for the makeup kit gold is 2882812427, Glimmer chose
    # 5012998807.
    top = retail.find_variants(rdb, "1763705424", sort_by_price="desc")["variants"][0]
    assert top["item_id"] == "2882812427"
    shoe = retail.find_variants(rdb, "9791469541", keep_same=["size"], sort_by_price="desc")["variants"]
    assert shoe and all(v["options"]["size"] == "9" for v in shoe)


def test_retail_task93_no_exact_match_names_the_blocking_constraint(rdb):
    # i7 / 8GB / 1TB with the same colour does not exist; Glimmer dropped the 8GB it was asked for and kept the
    # colour. Gold is 9844888101 (black), which breaks only the colour.
    res = retail.find_variants(rdb, "2216662955", require={"processor": "i7", "ram": "8GB", "storage": "1TB SSD"},
                               keep_same=["screen size", "color"])
    assert res["count"] == 0 and res["closest"]["constraints_broken"] == 1
    first = res["closest"]["variants"][0]
    assert first["item_id"] == "9844888101" and first["breaks"] == ["color=black (wanted same as current: space grey)"]
    assert "closest" not in retail.find_variants(rdb, "2216662955", require={"processor": "i7"})


def test_airline_task7_upcoming_trips_total(adb):
    # "What is the total cost of my upcoming flights?" Gold communicates $1,628; most Glimmer runs gave other sums.
    res = airline.upcoming_trips(adb, "daiki_muller_1116")
    assert {r["reservation_id"] for r in res["reservations"]} == {"7WPL39", "XEHM4B", "59XX6W", "3EMQJ6"}
    assert res["total_cost"] == 1628
    assert airline.upcoming_trips(adb, "nobody_000")["reservations"] == []


def test_retail_exchange_checks_items_and_difference(rdb):
    res = retail.check_eligibility(rdb, "#W9911714", "modify_items", item_ids=["1763705424"],
                                   new_item_ids=["2882812427"])
    assert res["allowed"] is True and "price_difference" in res["details"]
    wrong = retail.check_eligibility(rdb, "#W9911714", "modify_items", item_ids=["1763705424"],
                                     new_item_ids=["4107812777"])  # a shoe variant, not a makeup kit
    assert wrong["allowed"] is False


def test_tau2_environment_exposes_and_runs_the_tools():
    """Through τ²: the registered constructor builds an environment whose tool list includes the decision tools,
    with valid OpenAI schemas, and a tool call is executed against the live database."""
    import json

    from tau2.data_model.message import ToolCall
    from tau2.registry import registry

    from glint.assist.tau2_tools import install

    for domain, tool, args in (
        ("retail", "check_order_action", {"order_id": "#W9571698", "action": "return",
                                          "payment_method_id": "credit_card_1565124", "item_ids": ["6065192424"]}),
        ("airline", "check_reservation_action", {"reservation_id": "XEHM4B", "action": "cancel",
                                                 "reason": "change of plan"}),
    ):
        install(domain)
        install(domain)  # idempotent
        env = registry.get_env_constructor(domain)()
        names = {t.name for t in env.get_tools()}
        assert tool in names and (domain != "retail" or {"find_user_items", "find_variants"} <= names)
        schema = next(t for t in env.get_tools() if t.name == tool).openai_schema
        assert schema["function"]["parameters"]["properties"]
        out = env.get_response(ToolCall(id="1", name=tool, arguments=args))
        # τ²'s to_json_str renders every bool as a string for all tools
        assert not out.error and json.loads(out.content)["allowed"] == "False"


def test_mcp_server_lists_and_calls_the_tools():
    """Through the MCP layer: the server lists all four tools and returns the same verdict as the in-run tool."""
    import asyncio
    import json

    from glint.assist.server import server

    async def go():
        names = {t.name for t in await server.list_tools()}
        res = await server.call_tool("check_reservation_action",
                                     {"reservation_id": "XEHM4B", "action": "cancel", "reason": "change of plan"})
        return names, res

    names, res = asyncio.run(go())
    assert names == {"check_order_action", "find_user_items", "find_variants", "check_reservation_action",
                     "get_upcoming_trips_cost", "check_credit_limit_increase", "check_card_dispute"}
    payload = res.structured_content if getattr(res, "structured_content", None) else json.loads(res.content[0].text)
    payload = payload.get("result", payload)
    assert payload["allowed"] is False and "change_cabin" in payload["paths_to_allowed"][0]


BANKING_WRITES = ("order_replacement_credit_card_7291", "file_credit_card_transaction_dispute_4829")


def _banking_episode(task_id: str):
    """The task's starting database and its gold calls to discoverable tools."""
    import json

    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.domains.banking_knowledge.tools import KnowledgeTools

    tasks = json.loads((DOMAINS / "banking_knowledge" / "tasks.json").read_text(encoding="utf-8"))
    task = next(t for t in tasks if t["id"] == task_id)
    tools = KnowledgeTools(TransactionalDB.load(str(DOMAINS / "banking_knowledge" / "db.json")))
    agent = ((task.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data")
    if agent:
        tools.update_db(agent)
    calls = [(a["arguments"]["agent_tool_name"], json.loads(a["arguments"].get("arguments") or "{}"))
             for a in task["evaluation_criteria"].get("actions") or [] if a["name"] == "call_discoverable_agent_tool"]
    return tools, calls


def _card_of(db, user_id: str, transaction_id: str) -> str:
    txn = db.credit_card_transaction_history.data[transaction_id]
    return txn.get("credit_card_account_id") or next(
        k for k, v in db.credit_card_accounts.data.items()
        if v.get("user_id") == user_id and v.get("card_type") == txn.get("credit_card_type"))


BANKING_TASKS = ["task_031", "task_037", "task_038", "task_039", "task_040", "task_041", "task_050", "task_051",
                 "task_052", "task_053", "task_054", "task_080", "task_081"]


@pytest.mark.parametrize("task_id", BANKING_TASKS)
def test_banking_verdicts_match_every_gold_call(task_id):
    """Replay the gold trajectory; before each dispute and each credit limit increase, the tool's values must equal
    gold's (card_action, eligible_for_provisional_credit, purchase date; approve with the new limit or the denial
    reason)."""
    from glint.assist import banking

    tools, calls = _banking_episode(task_id)
    names = [n for n, _ in calls]
    checked = 0
    for i, (name, args) in enumerate(calls):
        later = [n for n in names[i + 1:] if n in BANKING_WRITES]
        if name == "file_credit_card_transaction_dispute_4829":
            v = banking.check_dispute(tools.db, args["user_id"], _card_of(tools.db, args["user_id"],
                                      args["transaction_id"]), args["transaction_id"], args["dispute_reason"],
                                      args["contacted_merchant"],
                                      replacement_requested="order_replacement_credit_card_7291" in later)
            assert (v["card_action"], v["eligible_for_provisional_credit"], v["purchase_date"]) == (
                args["card_action"], args["eligible_for_provisional_credit"], args["purchase_date"])
            checked += 1
        elif name == "submit_credit_limit_increase_request_7392":
            v = banking.check_credit_limit_increase(tools.db, args["user_id"], args["credit_card_account_id"],
                                                    args["requested_increase_amount"], later)
            n, a = next((n, a) for n, a in calls[i + 1:] if n.startswith(("approve_credit", "deny_credit")))
            want = ("approve", a["new_credit_limit"]) if n.startswith("approve") else ("deny", a["denial_reason"])
            assert (v["verdict"], v.get("new_credit_limit") or v.get("denial_reason")) == want
            checked += 1
        getattr(tools, name)(**args)
    assert checked


def test_banking_task054_amount_and_order():
    # The user first asks for +$4,000 on a $5,000 Gold card (max 50%); Glimmer filed the dispute and the replacement
    # first, which made the increase fail, then denied it.
    from glint.assist import banking

    tools, _ = _banking_episode("task_054")
    args = ("584f9c5d00", "cc_584f9c5d00_gold")
    first = banking.check_credit_limit_increase(tools.db, *args, 4000, ["dispute a charge", "replacement card"])
    assert first["verdict"] == "reduce_amount" and first["max_increase"] == 2500 and "BEFORE" in first["sequencing"]
    assert banking.check_credit_limit_increase(tools.db, *args, 2500)["verdict"] == "approve"
    tools.order_replacement_credit_card_7291("cc_584f9c5d00_gold", "584f9c5d00", "1456 Astoria Boulevard", "fraud_suspected")
    after = banking.check_credit_limit_increase(tools.db, *args, 2500)
    assert after["verdict"] == "deny" and after["denial_reason"] == "pending_replacement_card"


def test_banking_task038_every_dispute_on_a_reissued_card():
    # Glimmer filed dispute 2 with keep_active while the card was being reissued.
    from glint.assist import banking

    tools, _ = _banking_episode("task_038")
    tools.order_replacement_credit_card_7291("cc_890389b165_silver", "890389b165", "2500 Woodward Avenue",
                                             "fraud_suspected")
    for txn, reason in (("txn_e45e7acb56a0", "unauthorized_fraudulent_charge"),
                        ("txn_28cb62125078", "goods_services_not_as_described")):
        assert banking.check_dispute(tools.db, "890389b165", "cc_890389b165_silver", txn, reason,
                                     False)["card_action"] == "cancel_and_reissue"
    for txn, reason in (("txn_e45e7acb56a0", "unauthorized_fraudulent_charge"),
                        ("txn_28cb62125078", "goods_services_not_as_described")):
        # The merchant answer is filed for every transaction, fraud included, so it must be asked first
        ask = banking.check_dispute(tools.db, "890389b165", "cc_890389b165_silver", txn, reason)
        assert ask["ask_user"] and ask["ready"] is False


def test_banking_missing_credit_limit_gives_no_verdict():
    # Most base-database cards have no credit_limit; a $0 limit must not turn into "reduce_amount, max $0".
    from tau2.domains.banking_knowledge.data_model import TransactionalDB

    from glint.assist import banking

    db = TransactionalDB.load(str(DOMAINS / "banking_knowledge" / "db.json"))
    acc_id, acc = next((k, v) for k, v in db.credit_card_accounts.data.items()
                       if not v.get("credit_limit") and "gold" in v.get("card_type", "").lower())
    res = banking.check_credit_limit_increase(db, acc["user_id"], acc_id, 1000)
    assert res["verdict"] == "unknown" and "no credit limit" in res["reasons"][0]


def test_tau2_banking_environment_exposes_and_runs_the_tools():
    import json

    from tau2.data_model.message import ToolCall
    from tau2.registry import registry

    from glint.assist.tau2_tools import install

    install("banking_knowledge")
    constructor = registry.get_env_constructor("banking_knowledge")
    env = constructor(retrieval_variant="bm25_grep")
    names = {t.name for t in env.get_tools()}
    assert {"check_credit_limit_increase", "check_card_dispute"} <= names
    assert type(constructor(retrieval_variant="bm25_grep").tools) is type(env.tools)  # one class, reused
    out = env.get_response(ToolCall(id="1", name="check_card_dispute", arguments={
        "user_id": "890389b165", "credit_card_account_id": "cc_890389b165_silver",
        "transaction_id": "txn_e45e7acb56a0", "dispute_reason": "unauthorized_fraudulent_charge",
        "contacted_merchant": False, "replacement_requested": True}))
    assert not out.error and json.loads(out.content)["card_action"] == "cancel_and_reissue"
