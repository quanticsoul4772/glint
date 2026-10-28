"""glint.assist.server: every MCP tool answers from the τ² databases (loaded once) with the in-run verdicts."""
from glint.assist import server


def test_every_tool_answers_with_a_verdict_from_the_real_databases():
    retail, airline, banking = server._retail_db(), server._airline_db(), server._banking_db()
    assert server._retail_db() is retail  # loaded once
    order_id, order = next(iter(retail.orders.items()))
    item_id = order.items[0].item_id
    assert "allowed" in server.check_order_action(order_id, "cancel")
    assert isinstance(server.find_user_items(order.user_id), dict)
    assert isinstance(server.find_variants(item_id, keep_same=["color"]), dict)
    reservation_id, reservation = next(iter(airline.reservations.items()))
    assert isinstance(server.check_reservation_action(reservation_id, "cancel", reason="change of plan"), dict)
    assert isinstance(server.get_upcoming_trips_cost(reservation.user_id), dict)
    assert isinstance(banking, object)
    assert isinstance(server.check_credit_limit_increase("no_user", "no_account", 500.0), dict)
    assert isinstance(server.check_card_dispute("no_user", "no_account", "no_txn", "fraud"), dict)
