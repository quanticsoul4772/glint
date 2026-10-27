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
    assert names == {"check_order_action", "find_user_items", "find_variants", "check_reservation_action"}
    payload = res.structured_content if getattr(res, "structured_content", None) else json.loads(res.content[0].text)
    payload = payload.get("result", payload)
    assert payload["allowed"] is False and "change_cabin" in payload["paths_to_allowed"][0]
