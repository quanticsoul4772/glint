"""The decision tools on the real τ² databases, including the cases where Glimmer decided wrong in recorded runs."""

from datetime import date, timedelta

import pytest
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.retail.data_model import RetailDB

from glint import paths
from glint.assist import airline, retail

DOMAINS = paths.tau2_data() / "tau2" / "domains"


@pytest.fixture(scope="module")
def rdb():
    return RetailDB.load(DOMAINS / "retail" / "db.json")


@pytest.fixture(scope="module")
def adb():
    return FlightDB.load(DOMAINS / "airline" / "db.json")


def test_airline_task7_basic_economy_cancel_has_a_cabin_path(adb):
    # Glimmer refused: "Policy prohibits modifications to Basic Economy flights and cancellation is not allowed".
    # Gold upgrades XEHM4B to business, then cancels it.
    cancel = airline.check_eligibility(adb, "XEHM4B", "cancel", reason="change of plan", user_asked_how=True)
    assert cancel["allowed"] is False and cancel["paths_to_allowed"]
    assert "change_cabin" in cancel["paths_to_allowed"][0]
    assert airline.check_eligibility(adb, "XEHM4B", "change_cabin")["allowed"] is True
    assert airline.check_eligibility(adb, "XEHM4B", "change_flights")["allowed"] is False


def test_airline_held_out_0_no_workaround_unless_asked(adb):
    # Offered unprompted, "upgrade to business, then cancel" made Glimmer upgrade and cancel EHGLP3; gold is no change
    verdict = airline.check_eligibility(adb, "EHGLP3", "cancel", reason="change of plan")
    assert verdict["allowed"] is False and "paths_to_allowed" not in verdict
    assert "Do not suggest workarounds" in verdict["note"]
    assert airline.check_eligibility(adb, "EHGLP3", "cancel", reason="change of plan",
                                     user_asked_how=True)["paths_to_allowed"]


def test_retail_held_out_19_return_and_exchange_exclude_each_other(rdb):
    # Both were cleared separately on #W2890441; after the exchange the return failed (status 'exchange requested')
    for action, other in (("return", "exchange"), ("exchange", "return")):
        verdict = retail.check_eligibility(rdb, "#W2890441", action, item_ids=["2366567022"],
                                           new_item_ids=["2366567022"] if action == "exchange" else None)
        assert any("not both" in rule for rule in verdict["rules"])
        assert f"no {other}" in verdict["details"]["after_this"]


def test_airline_fixed_rules(adb):
    # airline/policy.md "Modify flight > Change baggage and insurance": "The user can add but not remove checked bags."
    bags = airline.check_eligibility(adb, "XEHM4B", "remove_baggage")
    assert bags["allowed"] is False and bags["reasons"] == ["bags can be added but not removed"]
    assert bags["rules"] == ["The user can add but not remove checked bags."]
    # same section: "The user cannot add insurance after initial booking."
    ins = airline.check_eligibility(adb, "XEHM4B", "add_insurance")
    assert ins["allowed"] is False and ins["reasons"] == ["insurance cannot be added after booking"]
    assert ins["rules"] == ["The user cannot add insurance after initial booking."]
    missing = airline.check_eligibility(adb, "NOPE00", "cancel")
    assert missing["allowed"] is False and missing["reasons"] == ["reservation NOPE00 does not exist"]


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
    # retail/policy.md "Cancel pending order": "An order can only be cancelled if its status is 'pending'"
    delivered = retail.check_eligibility(rdb, "#W9571698", "cancel")
    assert delivered["allowed"] is False
    assert delivered["reasons"] == ["status is 'delivered', action requires 'pending'"]
    pending = retail.check_eligibility(rdb, "#W9911714", "cancel")
    assert pending["allowed"] is True and pending["reasons"] == []
    # retail/policy.md "Return delivered order": "An order can only be returned if its status is 'delivered'"
    ret = retail.check_eligibility(rdb, "#W9911714", "return")
    assert ret["allowed"] is False and ret["reasons"] == ["status is 'pending', action requires 'delivered'"]


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
    # retail/policy.md "Modify items": the user pays or is refunded "the price difference" between the new item and
    # the one it replaces: makeup kit 2882812427 at $261.11 replaces 1763705424, bought at $235.44 in #W9911714
    res = retail.check_eligibility(rdb, "#W9911714", "modify_items", item_ids=["1763705424"],
                                   new_item_ids=["2882812427"])
    old_price = next(i.price for i in rdb.orders["#W9911714"].items if i.item_id == "1763705424")
    new_price = rdb.products["5149340237"].variants["2882812427"].price
    assert (old_price, new_price) == (235.44, 261.11)
    assert res["allowed"] is True and res["details"]["price_difference"] == round(new_price - old_price, 2) == 25.67
    # same section: "There cannot be any change of product types, e.g. modify shirt to shoe."
    wrong = retail.check_eligibility(rdb, "#W9911714", "modify_items", item_ids=["1763705424"],
                                     new_item_ids=["4107812777"])  # a shoe variant, not a makeup kit
    assert wrong["allowed"] is False
    assert wrong["reasons"] == ["4107812777 is not a variant of the same product as 1763705424"]


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
                                     {"reservation_id": "XEHM4B", "action": "cancel", "reason": "change of plan",
                                      "user_asked_how": True})
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
    assert first["verdict"] == "reduce_amount" and first["max_increase"] == 2500 and "credit limit increase first" in first["sequencing"]
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


def test_banking_task054_dispute_tool_holds_the_dispute_until_the_increase_is_decided():
    # On Modal (2026-10-27) Glimmer called check_card_dispute first and filed the dispute and the replacement before
    # the credit limit increase, which then failed; the dispute tool now carries the order
    from glint.assist import banking

    tools, _ = _banking_episode("task_054")
    args = ("584f9c5d00", "cc_584f9c5d00_gold", "txn_584f9c5d00_001", "unauthorized_fraudulent_charge", False, True)
    first = banking.check_dispute(tools.db, *args, other_requests=["credit limit increase", "replacement card"])
    assert "credit limit increase first" in first["sequencing"] and first["ready"] is False
    # once the increase is done it is no longer among the requests still to do
    after = banking.check_dispute(tools.db, *args, other_requests=["replacement card"])
    assert "sequencing" not in after and after["ready"] is True


def test_banking_increase_of_zero_asks_for_the_amount():
    # Glimmer passed requested_increase_amount=0 and got "approve" with the limit unchanged
    from glint.assist import banking

    tools, _ = _banking_episode("task_054")
    res = banking.check_credit_limit_increase(tools.db, "584f9c5d00", "cc_584f9c5d00_gold", 0)
    assert res["verdict"] == "ask_amount"


def test_banking_steps_say_hidden_tools_go_through_the_discoverable_wrapper():
    from glint.assist import banking

    steps = " ".join(banking.CLI_STEPS)
    assert "unlock_discoverable_agent_tool" in steps and "call_discoverable_agent_tool" in steps
    assert "Never call it by its own name" in steps


def test_review_retail_payment_check_holds_when_no_method_qualifies(rdb):
    # An empty list of allowed methods switched the check off (review 2026-10-27)
    same = retail.check_eligibility(rdb, "#W5918442", "modify_payment", "credit_card_5051208")
    assert same["allowed"] is False and "no payment method covers it" in same["reasons"][-1]
    short = retail.check_eligibility(rdb, "#W4316152", "exchange", "gift_card_7245904",
                                     ["7292993796", "7292993796"], ["3761330360", "9647374798"])
    assert short["allowed"] is False  # gift card $17.00 < price difference $21.10; τ² rejects it too


def test_review_retail_repeated_item_ids_need_that_many_in_the_order(rdb):
    res = retail.check_eligibility(rdb, "#W9911714", "modify_items", item_ids=["1763705424", "1763705424"],
                                   new_item_ids=["2882812427", "2882812427"])
    assert res["allowed"] is False and "more times than the order holds" in res["reasons"][0]


def test_review_airline_cancelled_past_segment_is_not_flown(adb):
    # 9FNHDR: business, HAT277 on 2024-05-10 cancelled by the airline; it was refused as "already flown"
    res = airline.check_eligibility(adb, "9FNHDR", "cancel", reason="change of plan")
    assert res["allowed"] is True


def test_review_string_false_does_not_leak_the_workaround(adb):
    res = airline.check_eligibility(adb, "EHGLP3", "cancel", "change of plan", "False")
    assert "paths_to_allowed" not in res
    assert airline.check_eligibility(adb, "EHGLP3", "cancel", "change of plan", "True")["paths_to_allowed"]


def test_review_banking_string_flags_and_cooldown():
    from glint.assist import banking

    tools, _ = _banking_episode("task_038")
    args = ("890389b165", "cc_890389b165_silver", "txn_28cb62125078", "goods_services_not_as_described")
    assert banking.check_dispute(tools.db, *args, "False", "False")["card_action"] == "keep_active"
    assert banking.check_dispute(tools.db, *args, "False")["provisional_credit_criteria"][
        "merchant_contacted_if_not_fraud"] is False
    # cooldown counts from the submission date, and only if the most recent request was approved
    tools, _ = _banking_episode("task_052")
    acc = "cc_5e4c1a83b0_bronze"
    reqs = tools.db.credit_limit_increase_requests.data
    reqs["later_denied"] = {"credit_card_account_id": acc, "user_id": "5e4c1a83b0", "status": "DENIED",
                            "submitted_date": "2025-10-01", "requested_increase_amount": 500}
    assert banking.check_credit_limit_increase(tools.db, "5e4c1a83b0", acc, 500)["checks"][
        "cooldown_period_active"]["passed"] is True


def test_assisting_an_environment_twice_is_a_no_op():
    # an installed constructor already assists the environment; the desk's tools pane assists it again
    from tau2.domains.banking_knowledge.environment import get_environment

    from glint.assist.tau2_tools import assist_environment

    env = assist_environment(get_environment(retrieval_variant="bm25_grep"), "banking_knowledge")
    cls = type(env.tools)
    assert type(assist_environment(env, "banking_knowledge").tools) is cls



# --- Airline rules the policy states, each checked for the reason that fires -------------------------------------
# Reservations from τ²'s airline db (now = 2024-05-15 15:00 EST, airline/policy.md "Domain Basic"):
# EMZ7JS business, all four segments landed; PLRJB9 basic economy, HAT176 2024-05-12 landed, HAT191 ahead;
# CDXEBS economy with insurance, regular member, booked 2024-05-08, nothing flown or cancelled;
# VAAOXJ economy, no insurance, regular member; SF5VA1 economy, no insurance, silver member.


def test_airline_cancel_after_a_flown_segment_needs_transfer(adb):
    # airline/policy.md "Cancel flight": "If any portion of the flight has already been flown, the agent cannot help
    # and transfer is needed." This comes before the grounds ("Otherwise, flight can be cancelled if..."), so even a
    # business reservation (otherwise cancellable) is refused
    res = airline.check_eligibility(adb, "EMZ7JS", "cancel", reason="change of plan")
    assert res["allowed"] is False and res["reasons"][0].startswith("segments already flown:")
    assert res["rules"] == [airline.RULES["flown_cancel"]] and "cancellation_grounds" not in res["details"]


def test_airline_change_cabin_after_a_flown_segment(adb):
    # airline/policy.md "Change cabin": "Cabin cannot be changed if any flight in the reservation has already been
    # flown." PLRJB9 has flown one of its two segments
    res = airline.check_eligibility(adb, "PLRJB9", "change_cabin")
    assert res["allowed"] is False and res["reasons"] == ["segments already flown: ['HAT176 2024-05-12 (landed)']"]
    # "In other cases, all reservations, including basic economy, can change cabin without changing the flights."
    assert airline.check_eligibility(adb, "VAAOXJ", "change_cabin")["allowed"] is True


def test_airline_already_cancelled_reservation(adb):
    # airline/policy.md "Cancel flight" cancels an active reservation; none of its actions applies to one whose
    # status is already "cancelled". A business reservation, cancellable while active, is refused once cancelled
    db = adb.model_copy(deep=True)
    assert airline.check_eligibility(db, "VAAOXJ", "cancel", reason="change of plan")["allowed"] is False
    db.reservations["VAAOXJ"].cabin = "business"
    assert airline.check_eligibility(db, "VAAOXJ", "cancel", reason="change of plan")["allowed"] is True
    db.reservations["VAAOXJ"].status = "cancelled"
    for action in ("cancel", "change_cabin"):
        res = airline.check_eligibility(db, "VAAOXJ", action, reason="change of plan")
        assert res["allowed"] is False and res["reasons"] == ["reservation is already cancelled"]


@pytest.mark.parametrize("reason, allowed", [("health", True), ("weather", True), ("change of plan", False)])
def test_airline_cancel_with_insurance_and_a_covered_reason(adb, reason, allowed):
    # airline/policy.md "Cancel flight": allowed if "The user has travel insurance and the reason for cancellation is
    # covered by insurance"; "Book flight > Travel insurance": insurance "enables full refund if the user needs to
    # cancel the flight given health or weather reasons". CDXEBS has insurance and no other ground
    res = airline.check_eligibility(adb, "CDXEBS", "cancel", reason=reason)
    assert res["allowed"] is allowed
    grounds = res["details"]["cancellation_grounds"]
    assert grounds["insurance covers the reason"] is allowed and sum(grounds.values()) == int(allowed)
    assert res["reasons"] == ([] if allowed else ["none of the cancellation conditions holds"])
    # without insurance the same reason is not a ground
    assert airline.check_eligibility(adb, "VAAOXJ", "cancel", reason=reason)["allowed"] is False


def test_airline_compensation_eligibility(adb):
    # airline/policy.md "Refunds and Compensation": "Only compensate if the user is a silver/gold member or has travel
    # insurance or flies business." and "Do not compensate if the user is regular member and has no travel insurance
    # and flies (basic) economy."
    regular = airline.check_eligibility(adb, "VAAOXJ", "compensation")
    assert regular["allowed"] is False
    assert regular["reasons"] == ["regular member, no insurance and not business: no compensation"]
    assert airline.check_eligibility(adb, "SF5VA1", "compensation")["allowed"] is True   # silver
    assert airline.check_eligibility(adb, "CDXEBS", "compensation")["allowed"] is True   # regular, insured
    db = adb.model_copy(deep=True)
    db.reservations["VAAOXJ"].cabin = "business"                                            # regular, business
    assert airline.check_eligibility(db, "VAAOXJ", "compensation")["allowed"] is True


# --- Banking credit limit increase: every check, at its threshold, and the order of the checks ---------------------
# Policy: banking_knowledge/documents credit_card_account_logistics_005 (account age, cooldown, utilization by tier),
# _006 (on-time payment months, maximum increase share), _007 (the order of the checks). "Today" is
# 2025-11-14 (τ² banking get_current_time: "The current time is 2025-11-14 03:40:00 EST").
# Base: task_054's starting database, card cc_584f9c5d00_gold ($5,000 limit, $487.50 balance, opened 03/20/2023,
# six ON_TIME payments 05/2025-10/2025, no requests, disputes or orders); the tier is set through card_type, using
# card names listed under each tier in credit_cards_(general)_015.
POLICY_TODAY = date(2025, 11, 14)
CLI_USER, CLI_CARD = "584f9c5d00", "cc_584f9c5d00_gold"
# card: (min account age days, cooldown days, utilization must be below, consecutive on-time months, max share)
CLI_POLICY = {"Bronze Rewards Card": (120, 120, 0.70, 6, 0.25), "Silver Rewards Card": (90, 90, 0.80, 3, 0.50),
              "Gold Rewards Card": (60, 60, 0.90, 3, 0.50)}


def _cli_db(card_type: str = "Gold Rewards Card"):
    tools, _ = _banking_episode("task_054")
    tools.db.credit_card_accounts.data[CLI_CARD]["card_type"] = card_type
    return tools.db


def _days_ago(n: int) -> str:
    return (POLICY_TODAY - timedelta(days=n)).strftime("%m/%d/%Y")


def _cli(db, amount: float = 1000):
    from glint.assist import banking

    return banking.check_credit_limit_increase(db, CLI_USER, CLI_CARD, amount)


def _assert_cli(res, reason):
    if reason is None:
        assert res["verdict"] == "approve", res.get("denial_reason")
    else:
        assert (res["verdict"], res.get("denial_reason")) == ("deny", reason)


def test_cli_base_account_is_approved_in_every_tier():
    for card in CLI_POLICY:
        res = _cli(_cli_db(card))
        assert res["verdict"] == "approve" and res["new_credit_limit"] == 6000


@pytest.mark.parametrize("card", CLI_POLICY)
def test_cli_max_increase_boundary(card):
    # logistics_006 "Maximum Credit Limit Increase Amounts": entry 25%, mid and premium 50% of the current limit;
    # "If the customer requests more than the maximum allowed ... inform them of the maximum" (_007 Step 0: do not
    # submit it). At the maximum is allowed; a cent over is not
    top = 5000 * CLI_POLICY[card][4]
    at = _cli(_cli_db(card), top)
    assert at["max_increase"] == top and at["verdict"] == "approve" and at["new_credit_limit"] == 5000 + top
    assert _cli(_cli_db(card), top + 0.01)["verdict"] == "reduce_amount"
    assert _cli(_cli_db(card), top - 1)["verdict"] == "approve"


@pytest.mark.parametrize("card", CLI_POLICY)
@pytest.mark.parametrize("offset, reason", [(-1, "insufficient_account_age"), (0, None), (1, None)])
def test_cli_account_age_boundary(card, offset, reason):
    # logistics_005 §1: "You qualify on or after the day the minimum is reached." (120 / 90 / 60 days; "An entry-tier
    # account open for 119 days does not qualify; on day 120 and beyond it does")
    db = _cli_db(card)
    db.credit_card_accounts.data[CLI_CARD]["date_of_account_open"] = _days_ago(CLI_POLICY[card][0] + offset)
    _assert_cli(_cli(db), reason)


@pytest.mark.parametrize("card", CLI_POLICY)
@pytest.mark.parametrize("offset, reason", [(-1, "cooldown_period_active"), (0, None), (1, None)])
def test_cli_cooldown_boundary(card, offset, reason):
    # logistics_005 §2: wait 120 / 90 / 60 days between submissions, counted "from the date you last submitted",
    # "until 90 full days have passed"; it "applies only if your most recent request was approved"
    db = _cli_db(card)
    db.credit_limit_increase_requests.data["cli_prev"] = {
        "request_id": "cli_prev", "credit_card_account_id": CLI_CARD, "user_id": CLI_USER,
        "requested_increase_amount": "$500.00", "status": "approved",
        "submitted_date": (POLICY_TODAY - timedelta(days=CLI_POLICY[card][1] + offset)).isoformat()}
    _assert_cli(_cli(db), reason)
    db.credit_limit_increase_requests.data["cli_prev"]["status"] = "denied"  # "Denied requests do not trigger"
    _assert_cli(_cli(db), None)


@pytest.mark.parametrize("card", CLI_POLICY)
@pytest.mark.parametrize("cents, reason", [(-1, None), (0, "high_utilization"), (1, "high_utilization")])
def test_cli_utilization_boundary(card, cents, reason):
    # logistics_005 §3: utilization "must be below the maximum threshold ... Utilization at or above the threshold does
    # not qualify" ("If your premium-tier utilization is exactly 90%, it does not meet the 'below' requirement")
    db = _cli_db(card)
    balance = 5000 * CLI_POLICY[card][2] + cents / 100
    db.credit_card_accounts.data[CLI_CARD]["current_balance"] = f"${balance:,.2f}"
    _assert_cli(_cli(db), reason)


def _payments(db, statuses: list[str]):
    """Replace the card's payment history with one payment a month, newest (10/15/2025) first."""
    table = db.payment_history.data
    for k in [k for k, p in table.items() if p.get("credit_card_account_id") == CLI_CARD]:
        del table[k]
    for i, status in enumerate(statuses):
        table[f"pay_test_{i}"] = {"payment_id": f"pay_test_{i}", "credit_card_account_id": CLI_CARD,
                                  "user_id": CLI_USER, "payment_date": f"{10 - i:02d}/15/2025", "amount": "$100.00",
                                  "status": status}


@pytest.mark.parametrize("card", CLI_POLICY)
def test_cli_payment_history_boundary(card):
    # logistics_006 "Payment History Requirements": 6 (entry) / 3 (mid, premium) consecutive months of on-time payments
    need = CLI_POLICY[card][3]
    db = _cli_db(card)
    _payments(db, ["ON_TIME"] * need + ["LATE"])  # exactly the required run, a late payment before it
    _assert_cli(_cli(db), None)
    _payments(db, ["ON_TIME"] * (need + 1))
    _assert_cli(_cli(db), None)
    _payments(db, ["ON_TIME"] * (need - 1) + ["LATE"] + ["ON_TIME"] * 3)  # the latest run is one month short
    _assert_cli(_cli(db), "insufficient_payment_history")
    _payments(db, ["ON_TIME"] * (need - 1))  # only need - 1 months on record
    _assert_cli(_cli(db), "insufficient_payment_history")


def test_cli_pending_dispute_and_past_due_balance():
    # logistics_007 Step 2.3 "No Pending Disputes: Verify the account has no active disputes."
    db = _cli_db()
    db.transaction_disputes.data["dsp_test"] = {"dispute_id": "dsp_test", "transaction_id": "txn_584f9c5d00_001",
                                                "user_id": CLI_USER, "card_last_4_digits": "7823",
                                                "submitted_at": "11/12/2025", "status": "UNDER_REVIEW"}
    _assert_cli(_cli(db), "pending_disputes")
    db.transaction_disputes.data["dsp_test"]["status"] = "RESOLVED"
    _assert_cli(_cli(db), None)
    # Step 2.5 "Account Good Standing: The account must be current with no past-due balance."
    db.credit_card_accounts.data[CLI_CARD]["past_due_amount"] = "$0.01"
    _assert_cli(_cli(db), "past_due_balance")
    db.credit_card_accounts.data[CLI_CARD]["past_due_amount"] = "$0.00"
    _assert_cli(_cli(db), None)


def _break(db, check: str):
    acc = db.credit_card_accounts.data[CLI_CARD]
    if check == "insufficient_account_age":
        acc["date_of_account_open"] = _days_ago(10)
    elif check == "cooldown_period_active":
        db.credit_limit_increase_requests.data["cli_prev"] = {
            "credit_card_account_id": CLI_CARD, "user_id": CLI_USER, "status": "approved",
            "submitted_date": (POLICY_TODAY - timedelta(days=10)).isoformat()}
    elif check == "pending_disputes":
        db.transaction_disputes.data["dsp_test"] = {"dispute_id": "dsp_test", "transaction_id": "txn_584f9c5d00_001",
                                                    "user_id": CLI_USER, "submitted_at": "11/12/2025",
                                                    "status": "UNDER_REVIEW"}
    elif check == "pending_replacement_card":
        db.credit_card_orders.data["ccord_test"] = {"order_id": "ccord_test", "credit_card_account_id": CLI_CARD,
                                                    "user_id": CLI_USER, "order_date": "11/10/2025",
                                                    "status": "SHIPPED"}
    elif check == "past_due_balance":
        acc["past_due_amount"] = "$50.00"
    elif check == "high_utilization":
        acc["current_balance"] = "$4,900.00"
    elif check == "insufficient_payment_history":
        _payments(db, ["LATE"] * 6)


# logistics_007: "These steps MUST be followed in the exact order listed": Step 2 checks 1-6 (account age, cooldown,
# pending disputes, pending replacement, good standing, utilization), then Step 3 (payment history). The denial
# reason is the first requirement in that order that is not met.
CLI_ORDER = ["insufficient_account_age", "cooldown_period_active", "pending_disputes", "pending_replacement_card",
             "past_due_balance", "high_utilization", "insufficient_payment_history"]


@pytest.mark.parametrize("first, second", [(a, b) for i, a in enumerate(CLI_ORDER) for b in CLI_ORDER[i + 1:]])
def test_cli_denial_reason_follows_the_policy_order(first, second):
    db = _cli_db()
    _break(db, second)
    _assert_cli(_cli(db), second)  # each break alone fires its own reason
    _break(db, first)
    res = _cli(db)
    _assert_cli(res, first)
    assert not res["checks"][first]["passed"] and not res["checks"][second]["passed"]


# --- Banking dispute: provisional credit criteria at their thresholds ---------------------------------------------
# credit_cards_(general)_015 "Provisional Credit Eligibility Guidelines": eligible only if ALL criteria hold.
# Base: task_054, txn_584f9c5d00_001 ($487.50 on 11/11/2025, Gold card = premium tier, max $10,000), no prior disputes.


def _dispute(db, reason="unauthorized_fraudulent_charge", contacted=False):
    from glint.assist import banking

    return banking.check_dispute(db, CLI_USER, CLI_CARD, "txn_584f9c5d00_001", reason, contacted)


@pytest.mark.parametrize("days, eligible", [(59, False), (60, True), (61, True)])
def test_dispute_account_open_60_days(days, eligible):
    # criterion 1 "Account Standing: The customer's credit card account has been open for at least 60 days"
    db = _cli_db()
    db.credit_card_accounts.data[CLI_CARD]["date_of_account_open"] = _days_ago(days)
    res = _dispute(db)
    assert res["provisional_credit_criteria"]["account_open_60_days"] is eligible
    assert res["eligible_for_provisional_credit"] is eligible


@pytest.mark.parametrize("amount, eligible", [("$24.99", False), ("$25.00", True), ("$10,000.00", True),
                                              ("$10,000.01", False)])
def test_dispute_amount_range(amount, eligible):
    # criterion 3: "at least $25.00 and must not exceed the card's maximum provisional credit limit" (Premium Tier,
    # Gold Rewards Card: Maximum $10000.00)
    db = _cli_db()
    db.credit_card_transaction_history.data["txn_584f9c5d00_001"]["transaction_amount"] = amount
    res = _dispute(db)
    assert res["provisional_credit_criteria"]["amount_in_range"] is eligible
    assert res["eligible_for_provisional_credit"] is eligible


@pytest.mark.parametrize("days, eligible", [(30, False), (31, True)])
def test_dispute_goods_not_received_needs_more_than_30_days(days, eligible):
    # criterion 2: "'goods_services_not_received' (only if purchase was made more than 30 days ago)"
    db = _cli_db()
    db.credit_card_transaction_history.data["txn_584f9c5d00_001"]["transaction_date"] = _days_ago(days)
    res = _dispute(db, "goods_services_not_received", contacted=True)
    assert res["provisional_credit_criteria"]["reason_qualifies"] is eligible
    assert res["eligible_for_provisional_credit"] is eligible


@pytest.mark.parametrize("prior, eligible", [(2, True), (3, False)])
def test_dispute_at_most_two_prior_disputes(prior, eligible):
    # criterion 4: "The customer has not filed more than 2 disputes in the past 12 months"
    db = _cli_db()
    for i in range(prior):
        db.transaction_disputes.data[f"dsp_prior_{i}"] = {"dispute_id": f"dsp_prior_{i}", "user_id": CLI_USER,
                                                          "transaction_id": f"txn_other_{i}",
                                                          "submitted_at": "06/01/2025", "status": "RESOLVED"}
    res = _dispute(db)
    assert res["provisional_credit_criteria"]["at_most_2_disputes_past_12_months"] is eligible
    assert res["eligible_for_provisional_credit"] is eligible


def test_dispute_merchant_contact_only_required_for_non_fraud():
    # criterion 5: "For non-fraud disputes (anything other than 'unauthorized_fraudulent_charge'), the customer must
    # have attempted to resolve with the merchant first"
    db = _cli_db()
    assert _dispute(db, "unauthorized_fraudulent_charge", contacted=False)["eligible_for_provisional_credit"] is True
    assert _dispute(db, "duplicate_charge", contacted=False)["eligible_for_provisional_credit"] is False
    assert _dispute(db, "duplicate_charge", contacted=True)["eligible_for_provisional_credit"] is True


# Refusal paths: the inputs a model most often gets wrong (a wrong id, another user's record). The verdict text is
# what the model reads, so each test asserts it exactly (coverage review 2026-10-28).
def test_banking_wrong_account_or_user_is_refused_with_the_reason():
    from glint.assist import banking

    db = _cli_db()
    assert _cli_missing(banking, db, "cc_nope") == {
        "verdict": "error", "reasons": ["credit card account cc_nope does not exist"]}
    assert banking.check_credit_limit_increase(db, "someone_else", CLI_CARD, 1000) == {
        "verdict": "error", "reasons": [f"credit card account {CLI_CARD} does not belong to user someone_else"]}
    assert banking.check_dispute(db, CLI_USER, "cc_nope", "txn_584f9c5d00_001", "unauthorized_fraudulent_charge") == {
        "ready": False, "reasons": ["credit card account cc_nope does not exist"]}


def _cli_missing(banking, db, card_id):
    return banking.check_credit_limit_increase(db, CLI_USER, card_id, 1000)


def test_banking_dispute_of_a_transaction_not_on_record_is_refused():
    from glint.assist import banking

    res = banking.check_dispute(_cli_db(), CLI_USER, CLI_CARD, "txn_nope", "unauthorized_fraudulent_charge")
    assert res == {"ready": False, "reasons": [f"transaction txn_nope does not exist for user {CLI_USER}"]}


def test_banking_card_with_no_tier_asks_for_the_knowledge_base():
    from glint.assist import banking

    res = _cli(_cli_db("Mystery Card"))
    assert res == {"verdict": "unknown", "reasons": [
        "the knowledge base defines no CLI tier for Mystery Card; search it before deciding"]}
    assert banking._tier("Mystery Card") is None


def test_banking_dates_in_both_database_formats():
    from glint.assist import banking

    assert banking._date("") is None and banking._date(None) is None
    assert banking._date("2025-11-14") == banking._date("11/14/2025") == date(2025, 11, 14)


def test_retail_unknown_order_or_action_is_refused(rdb):
    res = retail.check_eligibility(rdb, "#W9911714", "upgrade")
    assert res["allowed"] is False and res["reasons"][0].startswith("unknown action 'upgrade'; one of [")
    assert retail.check_eligibility(rdb, "#W0000000", "cancel") == {
        "allowed": False, "reasons": ["order #W0000000 does not exist"]}


def test_retail_modify_items_refuses_bad_item_lists(rdb):
    def reasons(item_ids, new_item_ids):
        return retail.check_eligibility(rdb, "#W9911714", "modify_items", item_ids=item_ids,
                                        new_item_ids=new_item_ids)["reasons"]

    assert "item_ids and new_item_ids must be non-empty and the same length" in reasons([], [])
    assert "item_ids and new_item_ids must be non-empty and the same length" in reasons(["1763705424"], [])
    assert "item 0000000000 is not in order #W9911714" in reasons(["0000000000"], ["2882812427"])
    assert "1763705424 is the same item as 1763705424" in reasons(["1763705424"], ["1763705424"])
    product = rdb.products["5149340237"]
    gone = next(v for v, var in product.variants.items() if not var.available)
    assert f"variant {gone} is not available" in reasons(["1763705424"], [gone])


def test_retail_return_of_items_not_in_the_order_is_refused(rdb):
    delivered = next(o for o in rdb.orders.values() if o.status == "delivered")
    res = retail.check_eligibility(rdb, delivered.order_id, "return", item_ids=["0000000000"])
    assert f"items ['0000000000'] are not in order {delivered.order_id}" in res["reasons"]


def test_retail_find_items_unknown_user_and_status_filter(rdb):
    assert retail.find_items(rdb, "nobody_0000") == {"items": [], "reasons": ["user nobody_0000 does not exist"]}
    user_id = next(iter(rdb.users))
    assert retail.find_items(rdb, user_id, order_status="no such status")["items"] == []


def test_retail_find_variants_unknown_item_and_nothing_available(rdb):
    assert retail.find_variants(rdb, "0000000000") == {"variants": [], "reasons": [
        "item 0000000000 is not in the catalogue"]}
    db = rdb.model_copy(deep=True)
    for var in db.products["5149340237"].variants.values():
        var.available = False
    res = retail.find_variants(db, "1763705424", require={"brand": "Brand C"})
    assert res["count"] == 0 and res["closest"] == {"variants": [], "note": "no available variant of this product"}


def test_banking_verification_check_needs_two_matching_fields_and_gives_golds_log_arguments():
    # banking-plan-v9-gated 037 t3: the customer gave email and phone (2 of the 4 fields the agent instructions
    # accept), and the agent asked 6 times for a date of birth or an address and never compared them
    import json

    from glint.assist import banking

    tools, _ = _banking_episode("task_037")
    tasks = json.loads((DOMAINS / "banking_knowledge" / "tasks.json").read_text(encoding="utf-8"))
    gold = next(a["arguments"] for a in next(t for t in tasks if t["id"] == "task_037")["evaluation_criteria"]["actions"]
                if a["name"] == "log_verification")
    user = tools.db.users.data[gold["user_id"]]
    digits = "".join(c for c in user["phone_number"] if c.isdigit())
    given = {"email": f"  {user['email'].upper()} ", "phone_number": f"({digits[:3]}) {digits[3:6]}.{digits[6:]}"}
    out = banking.check_verification(tools.db, gold["user_id"], given)
    assert out["verified"] and out["matched"] == ["email", "phone_number"] and "Do not ask" in out["next_step"]
    assert out["log_verification"] == {k: v for k, v in gold.items() if k != "time_verified"}
    month, day, year = user["date_of_birth"].split("/")
    assert banking.check_verification(tools.db, gold["user_id"], {"date_of_birth": f"{year}-{month}-{day}",
                                                                  "address": user["address"].lower()})["verified"]
    one = banking.check_verification(tools.db, gold["user_id"], given | {"phone_number": "000-000-0000"})
    assert not one["verified"] and one["did_not_match"] == ["phone_number"] and "log_verification" not in one
    assert "1 more matching value(s) needed" in one["next_step"]
    assert banking.check_verification(tools.db, gold["user_id"], {"date_of_birth": "not a date"})["matched"] == []
    assert banking.check_verification(tools.db, "nobody", given) == {"verified": False,
                                                                    "reasons": ["user nobody does not exist"]}
