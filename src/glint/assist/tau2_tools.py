"""Adds the decision tools to τ²'s retail, airline and banking_knowledge toolkits, so Glimmer can call them in a run.

`install(domain)` wraps the registered environment constructor: every environment it builds gets the domain toolkit
subclass below on the same live database, so the tools see the episode's current state. The
tools are READ tools: they never change the database and are skipped when τ² replays an episode for evaluation. The
one exception is banking's plan_disputes, which τ² replays (see DisputePlanTools and glint.gates.enforce): it still
never changes the database.
"""
from __future__ import annotations

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.retail.tools import RetailTools
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.registry import registry

from glint.assist import airline, banking, kb, retail

SEARCH_WARN = 60  # knowledge-base calls in one conversation before the run log warns (see _count_search)


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
            dict: product, current_item, variants (item_id, options, price, available), count. When no variant
            matches, 'closest' lists the variants that break the fewest constraints and which ones each breaks: show
            those to the user and let the user choose which constraint to give up.
        """
        return retail.find_variants(self.db, item_id, require, keep_same, True, sort_by_price)


class AssistedAirlineTools(AirlineTools):
    @is_tool(ToolType.READ)
    def check_reservation_action(self, reservation_id: str, action: str, reason: str = "",
                                 user_asked_how: bool = False) -> dict:
        """Policy decision tool: call this BEFORE cancelling or changing a reservation, before offering compensation,
        and before telling the user what is or is not possible. It applies the airline policy (with the policy's
        current time) to the reservation and returns the authoritative verdict: allowed (true/false), the reasons and
        the policy rules. Follow its verdict. When an action is not allowed, tell the user so; do not suggest
        workarounds. Only if the user asks how it could be made possible, call it with user_asked_how=true to get
        the allowed changes that would make it allowed.

        Args:
            reservation_id: The reservation id, such as '8JX2WO'.
            action: One of 'cancel', 'change_flights', 'change_cabin', 'add_baggage', 'remove_baggage',
                'add_insurance', 'change_passengers', 'compensation'.
            reason: For 'cancel': the user's reason (e.g. 'change of plan', 'health', 'weather', 'airline cancelled').
            user_asked_how: True only if the user has asked how the action could be made possible.

        Returns:
            dict: allowed, reasons, rules, details; paths_to_allowed only when user_asked_how is true.
        """
        return airline.check_eligibility(self.db, reservation_id, action, reason or None, user_asked_how)

    @is_tool(ToolType.READ)
    def get_upcoming_trips_cost(self, user_id: str) -> dict:
        """Upcoming trips calculator: call this whenever the user asks about their upcoming flights or what they cost
        in total, instead of adding up prices yourself. It returns every reservation of the user whose flights are
        all still ahead (not cancelled, nothing flown), the amount paid for each, and the exact total. Quote its
        total_cost to the user.

        Args:
            user_id: The user id, such as 'sara_doe_496'.

        Returns:
            dict: reservations (reservation_id, cabin, passengers, flights, amount_paid), count, total_cost.
        """
        return airline.upcoming_trips(self.db, user_id)


class DisputePlanTools(ToolKitBase):
    """plan_disputes alone. Every gated banking toolkit carries it (glint.gates.enforce attaches it), because the plan
    gate on credit card disputes needs it; BankingDecisionTools includes it."""

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def plan_disputes(self, user_id: str, transactions: list[dict], other_requests: list[str]) -> dict:
        """Credit card dispute planning tool: call this BEFORE filing any credit card transaction dispute, with EVERY
        transaction the customer wants disputed in this conversation (one transaction too). It returns the order to
        file them in and, for each, the eligible_for_provisional_credit value to file with and the policy criteria
        behind it. Provisional credit needs at most 2 disputes filed in the past 12 months (knowledge base
        credit_cards_(general)_015, criterion 4), so the disputes filed first take the credit. Each dispute filing is
        checked against this plan: file them in its order, with its values. Calling it again replaces the plan (for
        example when the customer adds a transaction or answers the merchant question).

        The order puts first the disputes that meet every credit criterion except the count (a qualifying reason:
        unauthorized_fraudulent_charge, duplicate_charge, goods_services_not_received older than 30 days; and the
        account age, amount and merchant criteria), largest amount first, then the rest in the given order. The
        policy text sets no order; this one is taken from τ²-bench train tasks (037-041, 081). It does not change the
        database.

        Args:
            user_id: The customer's user id.
            transactions: Every dispute the customer wants, in the order they gave them, each an object
                {"transaction_id": "...", "dispute_reason": one of 'unauthorized_fraudulent_charge',
                'duplicate_charge', 'incorrect_amount', 'goods_services_not_received',
                'goods_services_not_as_described', 'canceled_subscription_still_charging', 'refund_never_processed',
                "contacted_merchant": the customer's answer to "did you try to resolve this with the merchant
                first?" (true/false; leave it out if not asked yet), "issue_noticed_date": the customer's answer to
                "when did you first notice this charge?" (MM/DD/YYYY; leave it out if not asked yet, and the plan asks
                for it), "card_action": the customer's answer to "keep the card active, or cancel it and get a
                new one?" ('keep_active' or 'cancel_and_reissue'; one answer per card; leave it out if not asked yet,
                and the plan asks for it), and for an 'incorrect_amount' dispute either
                "expected_amount" (what the customer should have been charged) or "partial_refund_amount" (the
                overcharge); leave both out until the customer has said, and the plan asks for it. Ask what the
                customer should have been charged without offering or suggesting a full refund; only if the customer
                asks for the whole charge back without being offered it, give "resolution_requested": "full_refund"
                instead}.
                An incorrect_amount dispute is filed as resolution_requested 'partial_refund' with the plan's
                partial_refund_amount unless the customer asked for a full refund (the partial default is taken from
                τ²-bench train tasks 039-041; the policy says to ask the customer).

            other_requests: The customer's other requests in this conversation that are not done yet, such as
                ['credit limit increase', 'replacement card']; [] if none. When it names a credit limit increase,
                the increase comes first (submit, the required lookups, approve or deny) and no dispute is filed
                before it is decided.

        Returns:
            dict: order (position, transaction_id, dispute_reason, contacted_merchant, issue_noticed_date,
            card_action, eligible_for_provisional_credit, provisional_credit_criteria, and resolution_requested with
            partial_refund_amount for incorrect_amount), already_filed, rejected, ask_user, ready, and
            increase_first with sequencing when the increase must come first.
        """
        return banking.plan_disputes(self.db, user_id, transactions, other_requests)

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def record_retention_answer(self, credit_card_account_id: str, user_id: str, offer: str, accepted: bool,
                                customer_words: str) -> dict:
        """Record the customer's answer to the retention offer before closing a credit card account. The retention
        protocol (knowledge base credit_card_account_logistics_003) makes ONE retention offer by card tier after the
        reason is logged and the concern addressed (entry: 500 points or a $5 statement credit; mid: 2,000 points or
        $20; premium and above: 5,000 points or $50), then closes only if the customer declines. Ask the customer a
        direct yes/no question about the offer, then call this with their answer. A closure is checked against it:
        no closure without a recorded answer, and none after the customer accepted. Not needed when a closure reason
        was already on record within the past year (retention is skipped). It does not change the database.

        Args:
            credit_card_account_id: The credit card account the customer wants to close.
            user_id: The customer's user id.
            offer: The offer made, for example "$20 statement credit".
            accepted: true if the customer accepted the offer and keeps the account, false if they declined.
            customer_words: The customer's own words answering the offer, quoted.

        Returns:
            dict: recorded, the answer as recorded, and the next step; or an error.
        """
        return banking.record_retention_answer(self.db, credit_card_account_id, user_id, offer, accepted,
                                               customer_words)


    @is_tool(ToolType.GENERIC, mutates_state=True)
    def record_requests(self, user_id: str, requests: list[dict]) -> dict:
        """Record the account actions the customer asked for, before making any of them. Each write below is checked
        against this record: a close, payment, filing or other change it does not cover is refused until the customer
        asks for it and it is recorded (the refusal shows the record_requests call to make). A closure reason log or a
        debit card order it does not cover still runs and is flagged for review. A close_debit_card request also covers
        the replacement order on that card's checking account. Call it again to add requests when the customer asks for
        more (each call adds to the record). It does not change the database.

        Args:
            user_id: The customer's user id.
            requests: One object per action the customer asked for: {"action": one of 'dispute_debit_transaction'
                (target: transaction_id), 'close_debit_card' (card_id), 'order_debit_card' (checking account_id),
                'clear_fraud_alert_or_velocity_block' (card_id), 'temporary_limit_increase' (card_id),
                'reset_or_change_pin' (card_id), 'pay_credit_card' (credit_card_account_id), 'correct_rewards'
                (transaction_id), 'close_credit_card' (credit_card_account_id), 'savings_interest_correction'
                (savings account_id); "target_id": the id of the card, account or transaction the customer means
                (look it up first; leave it empty only if the customer's request covers any); "customer_words":
                the customer's own words asking for it}.

        Returns:
            dict: recorded, the requests as recorded; or an error.
        """
        return banking.record_requests(self.db, user_id, requests)


class BankingDecisionTools(DisputePlanTools):
    """Mixed in ahead of the banking toolkit class τ² composes per retrieval variant (see `install`). Its KB_search
    and grep replace τ²'s (whole documents) with excerpts within a context budget (glint.assist.kb), wherever this
    class is applied: the tools arm and glint-bank with --decision-tools. The base arm and glint-bank without the flag
    keep τ²'s."""

    @is_tool(ToolType.READ)
    def check_verification(self, user_id: str, date_of_birth: str = "", email: str = "", phone_number: str = "",
                           address: str = "") -> dict:
        """Identity verification decision tool. After finding the user (get_user_information_by_name or similar),
        pass the values the CUSTOMER gave, exactly as they gave them; leave the others empty. The customer is verified
        when any 2 of date of birth, email, phone number and address match the record. When verified, it returns the
        log_verification arguments.

        Args:
            user_id: The user's id from the lookup.
            date_of_birth: The date of birth the customer gave, if any.
            email: The email the customer gave, if any.
            phone_number: The phone number the customer gave, if any.
            address: The address the customer gave, if any.

        Returns:
            dict: verified, matched, did_not_match, next_step, and log_verification when verified.
        """
        return banking.check_verification(self.db, user_id, {"date_of_birth": date_of_birth, "email": email,
                                                             "phone_number": phone_number, "address": address})

    @is_tool(ToolType.READ)
    def KB_search(self, query: str) -> str:
        """Search the knowledge base for relevant documents.

        Returns the 5 best documents: the best one in full (its first 6,000 characters), the others with their title,
        ID and the paragraph that best matches the query. Read any document with KB_read(doc_id). Repeating a search
        returns a pointer to the documents it found the first time.

        Args:
            query: The search query to find relevant documents

        Returns:
            The best document's text and the other documents' titles, IDs and best-matching paragraphs
        """
        return self._search("KB_search", query, self._kb_pipeline.retrieve, lambda results: kb.format_results(
            self._kb_pipeline, results, kb.query_score(query), "No relevant documents found."))

    @is_tool(ToolType.READ)
    def grep(self, pattern: str) -> str:
        """Search for a regex pattern in all knowledge base documents (case-insensitive).

        Returns the 5 documents with the most matches: the first in full (its first 6,000 characters), the others
        with their title, ID and the paragraph with the most matches. Read any document with KB_read(doc_id).
        Repeating a search returns a pointer to the documents it found the first time.

        Args:
            pattern: The regex pattern to search for (e.g., 'credit.*card', 'fee|charge')

        Returns:
            The first document's text and the other matching documents' titles, IDs and best paragraphs
        """
        return self._search("grep", pattern, self._grep_pipeline.retrieve, lambda results: kb.format_results(
            self._grep_pipeline, results, kb.pattern_score(pattern), f"No matches found for pattern: {pattern}"))

    def _search(self, tool: str, query: str, retrieve, render) -> str:
        """A search's answer, or for a repeat of an earlier search in this conversation a short pointer to what it
        returned. A repeat is the same tool with the same text: for KB_search after trimming, case and spacing; for
        grep after trimming only, since a regex's escapes are case-sensitive (\\d is not \\D) and its spaces count. In
        banking-unseen-gated one search in five of the runs that ran out of τ²'s 200 steps repeated an earlier one
        exactly (median 16 a run, up to 76; 2026-10-31). The toolkit lives for one conversation, so the record does
        too. The pointer names the earlier call by its number among all of this conversation's KB_search and grep
        calls, repeats included, so it matches the agent's own history."""
        self._count_search()
        seen = self.__dict__.setdefault("_glint_searches", {})
        calls = self.__dict__["_glint_search_calls"] = self.__dict__.get("_glint_search_calls", 0) + 1
        text = str(query)
        key = (tool, " ".join(text.lower().split()) if tool == "KB_search" else text.strip())
        if key in seen:
            call, ids, whole = seen[key]
            if not ids:
                return (f"Same {tool} as your search {call} in this conversation; it found no documents. Search with "
                        "different words.")
            shown = "shown whole" if whole else "shown up to its first page"
            return (f"Same {tool} as your search {call} in this conversation; its documents were: {', '.join(ids)} "
                    f"(the first {shown}). Read one with KB_read(doc_id), or search with different words.")
        results = retrieve(query)
        ids = [doc_id for doc_id, _ in results[:kb.TOP_DOCS]]
        top = (self._kb_pipeline if tool == "KB_search" else self._grep_pipeline).get_document_content(ids[0]) if ids \
            else ""
        seen[key] = (calls, ids, len(top or "") <= kb.PAGE_CHARS)
        return render(results)

    @is_tool(ToolType.READ)
    def KB_read(self, doc_id: str, offset: int = 0) -> str:
        """Read a knowledge base document in full, 6,000 characters at a time.

        Args:
            doc_id: The document ID from a KB_search or grep result
            offset: The character to start from (0 for the beginning; each page names where the next one starts)

        Returns:
            The document's title and the requested characters, with where the next page starts
        """
        self._count_search()
        return kb.read_page(self._kb_pipeline, doc_id, offset)

    def _count_search(self) -> None:
        """One WARNING line, to stderr (the run log), when this conversation reaches SEARCH_WARN knowledge-base calls:
        in banking-unseen-gated every tools-arm run that ran out of τ²'s 200 steps had made at least 72 (2026-10-31),
        and the run log said nothing until each one ended. Named by τ²'s per-run log folder (task_<id>/sim_<id>)."""
        n = self.__dict__["_glint_search_n"] = self.__dict__.get("_glint_search_n", 0) + 1
        if n == SEARCH_WARN:
            import sys

            from tau2.utils import llm_utils

            log_dir = llm_utils.llm_log_dir.get()
            run = (f"{log_dir.parent.parent.name} {log_dir.parent.name}" if log_dir is not None
                   else "a conversation")
            print(f"WARNING: {run} has made {n} knowledge-base calls (KB_search, grep, KB_read); in "
                  "banking-unseen-gated every run that ran out of τ²'s 200 steps had made at least 72",
                  file=sys.stderr, flush=True)

    # GENERIC with mutates_state=True, like plan_disputes: τ²'s strict replay skips READ tools, and the enforcer records
    # this verdict for the open-items footer, so the replay must rebuild it. It never writes the database.
    @is_tool(ToolType.GENERIC, mutates_state=True)
    def check_credit_limit_increase(self, user_id: str, credit_card_account_id: str, requested_increase_amount: float,
                                    other_requests: list[str] | None = None) -> dict:
        """Credit limit increase decision tool: call this as soon as the user asks for a credit limit increase,
        BEFORE submitting it, and again before approving or denying it. It applies the knowledge base's credit limit
        increase procedure to the account's current state and returns the authoritative verdict: 'reduce_amount'
        (with max_increase; do not submit more than that), 'approve' (with new_credit_limit) or 'deny' (with
        denial_reason), every required check with its value, the exact steps to follow in order, and whether the
        increase must be done before the user's other requests. Follow its verdict and its step order.

        Args:
            user_id: The user's id.
            credit_card_account_id: The credit card account id, such as 'cc_584f9c5d00_gold'.
            requested_increase_amount: The dollar amount of the increase the user asks for (not the new limit).
            other_requests: The user's other requests in this conversation, such as ['dispute a charge',
                'replacement card'].

        Returns:
            dict: verdict, new_credit_limit or denial_reason, max_increase, payment_months, checks, steps, sequencing.
        """
        return banking.check_credit_limit_increase(self.db, user_id, credit_card_account_id,
                                                   requested_increase_amount, other_requests)

    @is_tool(ToolType.READ)
    def check_card_dispute(self, user_id: str, credit_card_account_id: str, transaction_id: str, dispute_reason: str,
                           contacted_merchant: bool | None = None, replacement_requested: bool = False,
                           other_requests: list[str] | None = None) -> dict:
        """Credit card dispute decision tool for one transaction (plan_disputes gives the filing order and the
        provisional credit for all of a customer's disputes; call it first). Call this BEFORE filing each dispute. It
        returns the authoritative values to file with: card_action ('keep_active' or 'cancel_and_reissue', the same
        for every dispute on a card being replaced), eligible_for_provisional_credit with each criterion, the
        purchase date and card digits, and any question still to ask the user. Use its values in the dispute call.

        Args:
            user_id: The user's id.
            credit_card_account_id: The credit card account the transaction was charged to.
            transaction_id: The transaction to dispute.
            dispute_reason: One of 'unauthorized_fraudulent_charge', 'duplicate_charge', 'incorrect_amount',
                'goods_services_not_received', 'goods_services_not_as_described',
                'canceled_subscription_still_charging', 'refund_never_processed'.
            contacted_merchant: The user's answer to "did you try to resolve this with the merchant first?" for this
                transaction; leave empty if not asked yet.
            replacement_requested: True if the user wants this card replaced or cancelled in this conversation.
            other_requests: The user's other requests in this conversation that are not done yet, such as
                ['credit limit increase', 'replacement card']. If one is a credit limit increase, do that first.

        Returns:
            dict: card_action, eligible_for_provisional_credit, provisional_credit_criteria, purchase_date,
            card_last_4_digits, amount, ask_user, ready; when ready and planned, filing (the call that files it).
        """
        out = banking.check_dispute(self.db, user_id, credit_card_account_id, transaction_id, dispute_reason,
                                    contacted_merchant, replacement_requested, other_requests)
        plan = getattr(self, "glint_plans", {}).get(user_id)
        decided = user_id in getattr(self, "glint_increase_decided", set())
        if out.get("ready") and plan and (filing := banking.filing_call(self.db, user_id, plan, transaction_id,
                                                                        decided)):
            out["filing"] = filing
        return out


_ASSISTED = {"retail": AssistedRetailTools, "airline": AssistedAirlineTools}
_ASSISTED_BANKING: dict[type, type] = {}  # composed banking toolkit class -> its assisted subclass
_PLANNED_BANKING: dict[type, type] = {}  # composed banking toolkit class -> its subclass with plan_disputes only


def add_plan_tool(env):
    """Put plan_disputes on a banking environment's toolkit (in place, ahead of its class in the MRO), unless it has
    it already. Used by glint.gates.enforce: the plan gate needs the tool on every gated banking environment."""
    cls = type(env.tools)
    if not issubclass(cls, DisputePlanTools):
        if cls not in _PLANNED_BANKING:
            _PLANNED_BANKING[cls] = type(f"Planned{cls.__name__}", (DisputePlanTools, cls), {})
        env.tools.__class__ = _PLANNED_BANKING[cls]
    return env


def assist_environment(env, domain: str, enforcer=None):
    """Give one environment the decision tools, on its own live database. Used by install() for every environment
    of a domain, and by the Bank Desk for just the tools pane. With an `enforcer` (glint.gates.Enforcer), THAT
    environment's assistant writes are also gated: each one is checked before it runs, and a denied write does not
    run."""
    if domain == "banking_knowledge":
        # The banking toolkit class is composed per retrieval variant and holds per-episode state, so keep the
        # instance and put the decision tools ahead of its class in the MRO
        cls = type(env.tools)
        if not issubclass(cls, BankingDecisionTools):  # else already assisted (an installed constructor built it)
            missing = {"KB_search", "grep"} - set(env.tools.tools)
            if missing:  # the excerpt versions would add tools the base arm does not have
                raise RuntimeError(f"the banking decision tools replace KB_search and grep; this retrieval variant "
                                   f"lacks {sorted(missing)} (the runs use bm25_grep)")
            if cls not in _ASSISTED_BANKING:
                _ASSISTED_BANKING[cls] = type(f"Assisted{cls.__name__}", (BankingDecisionTools, cls), {})
            env.tools.__class__ = _ASSISTED_BANKING[cls]
    elif not isinstance(env.tools, _ASSISTED[domain]):
        env.tools = _ASSISTED[domain](env.tools.db)
    if enforcer is not None:
        enforcer.attach(env)
    return env


def install(domain: str, enforcer=None) -> None:
    """Make every environment τ² builds for `domain` carry the decision tools, and, with an `enforcer`, gate its
    writes. Call once per domain, before the run starts; a later call replaces the earlier one's enforcer."""
    base = registry.get_env_constructor(domain)
    if getattr(base, "glint_assisted", False):
        if base.glint_enforcer is enforcer:
            return
        base = base.glint_base

    def constructor(*args, **kwargs):
        return assist_environment(base(*args, **kwargs), domain, enforcer)

    constructor.glint_assisted, constructor.glint_base, constructor.glint_enforcer = True, base, enforcer
    registry._domains[domain] = constructor  # τ²'s registry has no public setter: checked right below
    if getattr(registry.get_env_constructor(domain), "glint_enforcer", None) is not enforcer or             not getattr(registry.get_env_constructor(domain), "glint_assisted", False):
        raise RuntimeError(f"installing the Glint tools for {domain} did not take effect (τ²'s registry changed?); "
                           "the tools arm would run without its tools")
