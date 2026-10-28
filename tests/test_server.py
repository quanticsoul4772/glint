"""glint.assist.server: every MCP tool answers from the τ² databases (loaded once) with the in-run verdicts.

Each tool is checked against the glint.assist function on a database loaded independently of the server's, on a case
the policy allows and one it does not (the cases of tests/test_assist.py)."""
import pytest
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import KNOWLEDGE_DB_PATH
from tau2.domains.retail.data_model import RetailDB

from glint import paths
from glint.assist import airline, banking, retail, server

DOMAINS = paths.tau2_data() / "tau2" / "domains"


@pytest.fixture(scope="module")
def dbs():
    return {"retail": RetailDB.load(DOMAINS / "retail" / "db.json"),
            "airline": FlightDB.load(DOMAINS / "airline" / "db.json"),
            "banking": TransactionalDB.load(str(KNOWLEDGE_DB_PATH))}


def test_each_database_is_loaded_once():
    assert server._retail_db() is server._retail_db()
    assert server._airline_db() is server._airline_db()
    assert server._banking_db() is server._banking_db()


def test_retail_tools_give_the_in_run_verdicts(dbs):
    db = dbs["retail"]
    denied = server.check_order_action("#W9571698", "cancel")  # delivered
    assert denied == retail.check_eligibility(db, "#W9571698", "cancel") and denied["allowed"] is False
    allowed = server.check_order_action("#W9911714", "cancel")  # pending
    assert allowed == retail.check_eligibility(db, "#W9911714", "cancel") and allowed["allowed"] is True
    refund = server.check_order_action("#W9571698", "return", payment_method_id="credit_card_1565124",
                                       item_ids=["6065192424"])
    assert refund == retail.check_eligibility(db, "#W9571698", "return", "credit_card_1565124", ["6065192424"])
    assert refund["allowed"] is False and refund["details"]["allowed_payment_methods"] == ["gift_card_7250692"]
    to_gift = server.check_order_action("#W9571698", "return", "gift_card_7250692", ["6065192424"])
    assert to_gift == retail.check_eligibility(db, "#W9571698", "return", "gift_card_7250692", ["6065192424"])
    assert to_gift["allowed"] is True
    found = server.find_user_items("lei_wilson_4541", "laptop", {"screen size": "15-inch", "ram": "32GB"})
    assert found == retail.find_items(db, "lei_wilson_4541", "laptop", {"screen size": "15-inch", "ram": "32GB"})
    assert [(i["order_id"], i["item_id"]) for i in found["items"]] == [("#W4073673", "2216662955")]
    variants = server.find_variants("1763705424")
    assert variants == retail.find_variants(db, "1763705424", None, None, True, "desc")
    assert variants["variants"][0]["item_id"] == "2882812427"


def test_airline_tools_give_the_in_run_verdicts(dbs):
    db = dbs["airline"]
    denied = server.check_reservation_action("XEHM4B", "change_flights")  # basic economy
    assert denied == airline.check_eligibility(db, "XEHM4B", "change_flights") and denied["allowed"] is False
    allowed = server.check_reservation_action("XEHM4B", "change_cabin")
    assert allowed == airline.check_eligibility(db, "XEHM4B", "change_cabin") and allowed["allowed"] is True
    asked = server.check_reservation_action("XEHM4B", "cancel", reason="change of plan", user_asked_how=True)
    assert asked == airline.check_eligibility(db, "XEHM4B", "cancel", "change of plan", True)
    assert asked["allowed"] is False and "change_cabin" in asked["paths_to_allowed"][0]
    trips = server.get_upcoming_trips_cost("nobody_000")
    assert trips == airline.upcoming_trips(db, "nobody_000") and trips["reservations"] == []


def test_banking_tools_give_the_in_run_verdicts(dbs):
    db = dbs["banking"]
    acc = ("584f9c5d00", "cc_584f9c5d00_gold")  # a $5,000 Gold card: at most +50%
    over = server.check_credit_limit_increase(*acc, 4000)
    assert over == banking.check_credit_limit_increase(db, *acc, 4000, None) and over["verdict"] == "reduce_amount"
    ok = server.check_credit_limit_increase(*acc, 2500)
    assert ok == banking.check_credit_limit_increase(db, *acc, 2500, None) and ok["verdict"] == "approve"
    dispute = ("890389b165", "cc_890389b165_silver", "txn_28cb62125078", "goods_services_not_as_described")
    ask = server.check_card_dispute(*dispute)  # the merchant answer is needed first
    assert ask == banking.check_dispute(db, *dispute, None, False, None) and ask["ready"] is False and ask["ask_user"]
    ready = server.check_card_dispute(*dispute, contacted_merchant=False)
    assert ready == banking.check_dispute(db, *dispute, False, False, None)
    assert ready["ready"] is True and ready["card_action"] == "keep_active"
