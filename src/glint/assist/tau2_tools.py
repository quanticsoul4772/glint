"""Adds the decision tools to τ²'s retail and airline toolkits, so Glimmer can call them in a run.

`install("retail")` / `install("airline")` wraps the registered environment constructor: every environment it builds
gets the domain toolkit subclass below on the same live database, so the tools see the episode's current state. The
tools are READ tools: they never change the database and are skipped when τ² replays an episode for evaluation.
"""
from __future__ import annotations

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.retail.tools import RetailTools
from tau2.environment.toolkit import ToolType, is_tool
from tau2.registry import registry

from glint.assist import airline, retail


class AssistedRetailTools(RetailTools):
    @is_tool(ToolType.READ)
    def check_order_action(self, order_id: str, action: str, payment_method_id: str = "",
                           item_ids: list[str] | None = None, new_item_ids: list[str] | None = None) -> dict:
        """Policy decision tool: call this BEFORE cancelling, modifying, returning or exchanging an order, and before
        telling the user what is or is not possible. It applies the retail policy to the order's current state and
        returns the authoritative verdict: allowed (true/false), the reasons, the policy rules, the payment methods
        that may be used, and the price difference. Follow its verdict.

        Args:
            order_id: The order id, such as '#W0000000'.
            action: One of 'cancel', 'modify_address', 'modify_payment', 'modify_items', 'return', 'exchange'.
            payment_method_id: The payment method the user wants to use, if any (for the refund or price difference).
            item_ids: For 'modify_items', 'exchange' or 'return': the item ids in the order.
            new_item_ids: For 'modify_items' or 'exchange': the new item ids, in the same order as item_ids.

        Returns:
            dict: allowed, reasons, rules, details (status, original payment method, allowed payment methods,
            price difference).
        """
        return retail.check_eligibility(self.db, order_id, action, payment_method_id or None, item_ids, new_item_ids)

    @is_tool(ToolType.READ)
    def find_user_items(self, user_id: str, product_name: str = "", options: dict | None = None,
                        order_status: str = "") -> dict:
        """Item finder: call this to work out WHICH item and order the user means from what they described (product
        name and option values such as size, color, ram), instead of comparing orders yourself. It searches every
        order of the user and returns each matching item with its order id, order status, item id and options.

        Args:
            user_id: The user id, such as 'sara_doe_496'.
            product_name: Part of the product name, such as 'laptop' or 'jigsaw puzzle'. Empty matches all.
            options: Option values the item must have, such as {"screen size": "15-inch", "ram": "32GB"}.
            order_status: Only orders with this status, such as 'delivered' or 'pending'. Empty matches all.

        Returns:
            dict: items (order_id, order_status, item_id, product_id, name, options, price) and count.
        """
        return retail.find_items(self.db, user_id, product_name or None, options, order_status or None)

    @is_tool(ToolType.READ)
    def find_variants(self, item_id: str, require: dict | None = None, keep_same: list[str] | None = None,
                      sort_by_price: str = "desc") -> dict:
        """Variant chooser: call this to pick a replacement variant for an item instead of comparing variants
        yourself, for example "the most expensive one", "the cheapest one", "same size but black". It returns the
        available variants of the item's product that have the required option values and keep the named options
        unchanged, sorted by price. The first result is the most expensive ('desc') or cheapest ('asc').

        Args:
            item_id: The current item id.
            require: Option values the new variant must have, such as {"color": "black"}.
            keep_same: Option names that must stay as on the current item, such as ["size"].
            sort_by_price: 'desc' (most expensive first) or 'asc' (cheapest first).

        Returns:
            dict: product, current_item, variants (item_id, options, price, available), count.
        """
        return retail.find_variants(self.db, item_id, require, keep_same, True, sort_by_price)


class AssistedAirlineTools(AirlineTools):
    @is_tool(ToolType.READ)
    def check_reservation_action(self, reservation_id: str, action: str, reason: str = "") -> dict:
        """Policy decision tool: call this BEFORE cancelling or changing a reservation, before offering compensation,
        and before telling the user what is or is not possible. It applies the airline policy (with the policy's
        current time) to the reservation and returns the authoritative verdict: allowed (true/false), the reasons, the
        policy rules, and, when not allowed, any allowed change that would make it allowed. Follow its verdict.

        Args:
            reservation_id: The reservation id, such as '8JX2WO'.
            action: One of 'cancel', 'change_flights', 'change_cabin', 'add_baggage', 'remove_baggage',
                'add_insurance', 'change_passengers', 'compensation'.
            reason: For 'cancel': the user's reason (e.g. 'change of plan', 'health', 'weather', 'airline cancelled').

        Returns:
            dict: allowed, reasons, rules, paths_to_allowed, details.
        """
        return airline.check_eligibility(self.db, reservation_id, action, reason or None)


_ASSISTED = {"retail": AssistedRetailTools, "airline": AssistedAirlineTools}


def install(domain: str) -> None:
    """Make every environment τ² builds for `domain` carry the decision tools. Call once, before the run starts."""
    base = registry.get_env_constructor(domain)
    if getattr(base, "glint_assisted", False):
        return
    toolkit_cls = _ASSISTED[domain]

    def constructor(*args, **kwargs):
        env = base(*args, **kwargs)
        env.tools = toolkit_cls(env.tools.db)
        return env

    constructor.glint_assisted = True
    registry._domains[domain] = constructor
