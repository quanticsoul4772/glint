"""MCP server exposing Glint's decision tools over the τ² retail, airline and banking_knowledge databases.

Run: python -m glint.assist.server   (stdio transport). Each tool loads its domain database once, from the vendored
τ²-bench data, and answers with the same verdicts the in-run tools give (`glint.assist.retail` / `.airline` /
`.banking`).
"""
from __future__ import annotations

from functools import cache

from mcp.server.mcpserver import MCPServer

from glint.assist import airline, banking, retail

server = MCPServer(
    name="glint-assist",
    instructions="Decision tools for retail orders, airline reservations, and bank credit limit increases and card "
                 "disputes. Call them before deciding what policy allows or which item a customer means; their "
                 "verdicts are deterministic and authoritative.",
)


@cache
def _retail_db():
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    return RetailDB.load(RETAIL_DB_PATH)


@cache
def _airline_db():
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    return FlightDB.load(AIRLINE_DB_PATH)


@cache
def _banking_db():
    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.domains.banking_knowledge.environment import KNOWLEDGE_DB_PATH

    return TransactionalDB.load(str(KNOWLEDGE_DB_PATH))


@server.tool()
def check_order_action(order_id: str, action: str, payment_method_id: str = "", item_ids: list[str] | None = None,
                       new_item_ids: list[str] | None = None) -> dict:
    """Retail policy verdict for an action on an order ('cancel', 'modify_address', 'modify_payment',
    'modify_items', 'return', 'exchange'): allowed, reasons, rules, allowed payment methods, price difference."""
    return retail.check_eligibility(_retail_db(), order_id, action, payment_method_id or None, item_ids, new_item_ids)


@server.tool()
def find_user_items(user_id: str, product_name: str = "", options: dict | None = None, order_status: str = "") -> dict:
    """Items across a user's orders matching a product name and option values."""
    return retail.find_items(_retail_db(), user_id, product_name or None, options, order_status or None)


@server.tool()
def find_variants(item_id: str, require: dict | None = None, keep_same: list[str] | None = None,
                  sort_by_price: str = "desc") -> dict:
    """Available variants of an item's product with required options and unchanged named options, sorted by price."""
    return retail.find_variants(_retail_db(), item_id, require, keep_same, True, sort_by_price)


@server.tool()
def check_reservation_action(reservation_id: str, action: str, reason: str = "", user_asked_how: bool = False) -> dict:
    """Airline policy verdict for an action on a reservation. With user_asked_how (the user asked how it could be made
    possible), also the allowed changes that would make it allowed."""
    return airline.check_eligibility(_airline_db(), reservation_id, action, reason or None, user_asked_how)


@server.tool()
def get_upcoming_trips_cost(user_id: str) -> dict:
    """A user's reservations with every flight still ahead, the amount paid for each, and the total."""
    return airline.upcoming_trips(_airline_db(), user_id)


@server.tool()
def check_credit_limit_increase(user_id: str, credit_card_account_id: str, requested_increase_amount: float,
                                other_requests: list[str] | None = None) -> dict:
    """Credit limit increase verdict (reduce_amount / approve / deny), every required check, and the step order."""
    return banking.check_credit_limit_increase(_banking_db(), user_id, credit_card_account_id,
                                               requested_increase_amount, other_requests)


@server.tool()
def check_card_dispute(user_id: str, credit_card_account_id: str, transaction_id: str, dispute_reason: str,
                       contacted_merchant: bool | None = None, replacement_requested: bool = False) -> dict:
    """The card_action and provisional-credit values a credit card dispute must be filed with."""
    return banking.check_dispute(_banking_db(), user_id, credit_card_account_id, transaction_id, dispute_reason,
                                 contacted_merchant, replacement_requested)


if __name__ == "__main__":
    server.run("stdio")
