"""Deterministic per-step labeler for τ²-bench retail trajectories.

Replays a trajectory in a live retail environment. Before each assistant tool call it reads the
pre-call DB and the message history, assigns labels from glint.labels, then executes the call so the
state stays faithful. Design: research/3-project/gap-G4G5-labels-data.md §G4.2–G4.3.

Input messages are plain dicts in τ²'s saved-simulation shape:
  {"role": "assistant"|"user"|"tool", "content": str|None, "tool_calls": [{"id","name","arguments"}],
   "id": <tool call id, tool messages only>, "error": bool}
"""
from __future__ import annotations

import inspect
import re
import typing
from collections import Counter
from dataclasses import dataclass, field

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Action
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import RETAIL_DB_PATH

from glint.labels import Label

AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
PRE_AUTH_ALLOWED = AUTH_TOOLS | {"list_all_product_types", "calculate", "transfer_to_human_agents"}
ORDER_WRITES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_pending_order_address",
    "return_delivered_order_items",
}
WRITES = ORDER_WRITES | {"modify_user_address"}
ONCE_PER_ORDER = {"exchange_delivered_order_items", "modify_pending_order_items"}
CANCEL_REASONS = {"no longer needed", "ordered by mistake"}
TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
# Free-text or computed arguments that are not expected to appear verbatim in the conversation.
UNGROUNDED_EXEMPT_ARGS = {"summary", "expression", "reason"}
AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|confirm(ed)?|correct|go ahead|proceed|sure|please do|do it|that'?s right)\b",
    re.IGNORECASE,
)

_BASE_DB: RetailDB | None = None


def _fresh_db() -> RetailDB:
    global _BASE_DB
    if _BASE_DB is None:
        _BASE_DB = RetailDB.load(RETAIL_DB_PATH)
    return _BASE_DB.model_copy(deep=True)


def _norm(value: str) -> str:
    return value.replace("#", "").strip().lower()


def _signature(tool: str) -> dict[str, tuple[type, bool]] | None:
    """Map arg name -> (python container type, required) for a retail tool, or None if unknown."""
    fn = getattr(RetailTools, tool, None)
    if fn is None or not callable(fn):
        return None
    hints = typing.get_type_hints(fn)
    out = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self":
            continue
        hint = hints.get(name, str)
        kind = list if typing.get_origin(hint) is list else (hint if isinstance(hint, type) else str)
        out[name] = (kind, param.default is inspect.Parameter.empty)
    return out


@dataclass
class StepRow:
    step_id: str
    task_id: str
    trial: int
    turn_idx: int
    call_idx: int
    tool: str
    args: dict
    labels: set[Label] = field(default_factory=set)


@dataclass
class _Ctx:
    """Conversation facts accumulated during replay."""

    auth_user: str | None = None
    fetched_orders: set[str] = field(default_factory=set)
    once_done: set[str] = field(default_factory=set)
    grounding: list[str] = field(default_factory=list)  # normalized user + tool text


class Labeler:
    def __init__(self, task: dict):
        self.task = task
        ec = task.get("evaluation_criteria") or {}
        self.gold = [Action.model_validate(a) for a in (ec.get("actions") or [])]
        self.gold_write_names = {a.name for a in self.gold if a.name in WRITES}
        self.gold_has_transfer = any(a.name == "transfer_to_human_agents" for a in self.gold)

    # ---- per-call checks -------------------------------------------------------------------
    def _fab(self, tool: str, args: dict, db: RetailDB, ctx: _Ctx) -> set[Label]:
        out = set()
        sig = _signature(tool)
        if sig is None:
            return {Label.FAB_SCHEMA}  # hallucinated tool name
        required = {k for k, (_, req) in sig.items() if req}
        if set(args) - set(sig) or required - set(args):
            out.add(Label.FAB_SCHEMA)
        for k, v in args.items():
            if k in sig and sig[k][0] is list and not isinstance(v, list):
                out.add(Label.FAB_SCHEMA)
            if k in sig and sig[k][0] is str and not isinstance(v, str):
                out.add(Label.FAB_SCHEMA)
        if self._nonexistent_ids(args, db):
            out.add(Label.FAB_ID_NONEXISTENT)
        if tool != "calculate" and self._ungrounded(args, ctx):
            out.add(Label.FAB_UNGROUNDED)
        return out

    @staticmethod
    def _nonexistent_ids(args: dict, db: RetailDB) -> bool:
        items = {iid for p in db.products.values() for iid in p.variants}
        if "order_id" in args and args["order_id"] not in db.orders:
            return True
        if "user_id" in args and args["user_id"] not in db.users:
            return True
        if "product_id" in args and args["product_id"] not in db.products:
            return True
        for key in ("item_id",):
            if key in args and args[key] not in items:
                return True
        for key in ("item_ids", "new_item_ids"):
            if key in args and isinstance(args[key], list) and any(i not in items for i in args[key]):
                return True
        if "payment_method_id" in args:
            owner = args.get("user_id")
            if owner is None and args.get("order_id") in db.orders:
                owner = db.orders[args["order_id"]].user_id
            methods = db.users[owner].payment_methods if owner in db.users else {}
            if args["payment_method_id"] not in methods:
                return True
        return False

    @staticmethod
    def _ungrounded(args: dict, ctx: _Ctx) -> bool:
        corpus = "\n".join(ctx.grounding)
        for k, v in args.items():
            if k in UNGROUNDED_EXEMPT_ARGS:
                continue
            values = v if isinstance(v, list) else [v]
            for val in values:
                if isinstance(val, str) and val.strip() and _norm(val) not in corpus:
                    return True
        return False

    def _viol(self, tool: str, args: dict, db: RetailDB, ctx: _Ctx, history: list[dict], msg: dict) -> set[Label]:
        out = set()
        calls = msg.get("tool_calls") or []
        if len(calls) > 1 or (msg.get("content") or "").strip():
            out.add(Label.VIOL_R13_TURN_SHAPE)
        if ctx.auth_user is None and tool not in PRE_AUTH_ALLOWED:
            out.add(Label.VIOL_R1_AUTH)
        order = db.orders.get(args.get("order_id")) if "order_id" in args else None
        if ctx.auth_user is not None:
            if "user_id" in args and args["user_id"] != ctx.auth_user:
                out.add(Label.VIOL_R2_OTHER_USER)
            if order is not None and order.user_id != ctx.auth_user:
                out.add(Label.VIOL_R2_OTHER_USER)
        if tool in ORDER_WRITES and args.get("order_id") not in ctx.fetched_orders:
            out.add(Label.VIOL_R3_NO_STATUS_CHECK)
        if order is not None:
            if tool == "cancel_pending_order" and order.status != "pending":
                out.add(Label.VIOL_R4_CANCEL_NOT_PENDING)
            if tool.startswith("modify_pending_order") and order.status != "pending":
                out.add(Label.VIOL_R5_MODIFY_NOT_PENDING)
            if tool in {"return_delivered_order_items", "exchange_delivered_order_items"} and order.status != "delivered":
                out.add(Label.VIOL_R6_NOT_DELIVERED)
        if tool in ONCE_PER_ORDER and args.get("order_id") in ctx.once_done:
            out.add(Label.VIOL_R7_REPEAT_MODIFY)
        if tool == "cancel_pending_order" and args.get("reason") not in CANCEL_REASONS:
            out.add(Label.VIOL_R8_CANCEL_REASON)
        if order is not None and tool == "modify_pending_order_payment":
            out |= self._r9(args, order, db)
        if order is not None and tool == "return_delivered_order_items":
            out |= self._r10(args, order, db)
        if order is not None and tool in ONCE_PER_ORDER:
            out |= self._r11_r12(args, order, db)
        if tool in WRITES and not self._confirmed(history):
            out.add(Label.VIOL_R15_NO_CONFIRMATION)
        return out

    @staticmethod
    def _r9(args, order, db) -> set[Label]:
        user = db.users.get(order.user_id)
        pm_id = args.get("payment_method_id")
        original = order.payment_history[0].payment_method_id if order.payment_history else None
        if user is None or pm_id not in user.payment_methods or pm_id == original:
            return {Label.VIOL_R9_PAYMENT}
        pm = user.payment_methods[pm_id]
        total = sum(i.price for i in order.items)
        if getattr(pm, "source", None) == "gift_card" and pm.balance < total:
            return {Label.VIOL_R9_PAYMENT}
        return set()

    @staticmethod
    def _r10(args, order, db) -> set[Label]:
        user = db.users.get(order.user_id)
        pm_id = args.get("payment_method_id")
        original = order.payment_history[0].payment_method_id if order.payment_history else None
        if pm_id == original:
            return set()
        if user is not None and pm_id in user.payment_methods and user.payment_methods[pm_id].source == "gift_card":
            return set()
        return {Label.VIOL_R10_REFUND_METHOD}

    @staticmethod
    def _r11_r12(args, order, db) -> set[Label]:
        out = set()
        old, new = args.get("item_ids") or [], args.get("new_item_ids") or []
        in_order = Counter(i.item_id for i in order.items)
        if len(old) != len(new) or any(n > in_order[i] for i, n in Counter(old).items()):
            out.add(Label.VIOL_R12_ITEMS)
        product_of = {i.item_id: i.product_id for i in order.items}
        for o, n in zip(old, new):
            prod = db.products.get(product_of.get(o))
            variant = prod.variants.get(n) if prod else None
            if variant is None or not variant.available or o == n:
                out.add(Label.VIOL_R11_VARIANT)
        return out

    @staticmethod
    def _confirmed(history: list[dict]) -> bool:
        """Last user turn is affirmative and the agent turn before it was text (listing the action)."""
        users = [i for i, m in enumerate(history) if m["role"] == "user"]
        if not users or not AFFIRMATIVE.search(history[users[-1]].get("content") or ""):
            return False
        prior = [m for m in history[: users[-1]] if m["role"] == "assistant"]
        return bool(prior) and not prior[-1].get("tool_calls") and bool((prior[-1].get("content") or "").strip())

    @staticmethod
    def _canonical(name: str, args: dict) -> dict:
        """Order-insensitive form of write args. τ²'s DB hash ignores item order, so gold matching must too."""
        args = dict(args)
        if "item_ids" in args and "new_item_ids" in args:
            # identity pairs (item swapped for itself) leave the DB unchanged, so they don't change the match
            pairs = sorted((o, n) for o, n in zip(args["item_ids"], args["new_item_ids"]) if o != n)
            args["item_ids"], args["new_item_ids"] = [p[0] for p in pairs], [p[1] for p in pairs]
        elif "item_ids" in args:
            args["item_ids"] = sorted(args["item_ids"])
        return args

    def _matches_gold(self, call: ToolCall) -> bool:
        canon = ToolCall(id=call.id, name=call.name, arguments=self._canonical(call.name, call.arguments),
                         requestor=call.requestor)
        for a in self.gold:
            gold = a.model_copy(update={"arguments": self._canonical(a.name, a.arguments)})
            if gold.compare_with_tool_call(canon):
                return True
        return False

    def _resid(self, tool: str, call: ToolCall, labels: set[Label]) -> set[Label]:
        if tool not in WRITES:
            return set()
        out = set()
        matches_gold = self._matches_gold(call)
        if not matches_gold and (not self.gold_write_names or self.gold_has_transfer):
            out.add(Label.RESID_UNWARRANTED_WRITE)
        if not matches_gold and self.gold_has_transfer:
            out.add(Label.RESID_MISSED_TRANSFER)
        if tool in self.gold_write_names and not matches_gold and not labels & {
            Label.FAB_SCHEMA, Label.FAB_ID_NONEXISTENT, Label.FAB_UNGROUNDED
        }:
            out.add(Label.RESID_WRONG_BUT_GROUNDED)
        return out

    # ---- replay ------------------------------------------------------------------------------
    def label(self, messages: list[dict], trial: int = 0) -> list[StepRow]:
        env = get_environment(db=_fresh_db())
        init = self.task.get("initial_state")
        if init:
            from tau2.data_model.tasks import InitialState

            state = InitialState.model_validate(init)
            env.set_state(state.initialization_data, state.initialization_actions, [])
        ctx, rows, history = _Ctx(), [], []
        tool_results = {m["id"]: m for m in messages if m.get("role") == "tool"}
        transferred = False
        for turn_idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "user" and msg.get("content"):
                ctx.grounding.append(_norm(msg["content"]))
            if role == "assistant" and msg.get("tool_calls"):
                for call_idx, tc in enumerate(msg["tool_calls"]):
                    tool, args = tc["name"], tc.get("arguments") or {}
                    row = StepRow(
                        step_id=f"{self.task['id']}:{trial}:{turn_idx}:{call_idx}",
                        task_id=str(self.task["id"]), trial=trial, turn_idx=turn_idx,
                        call_idx=call_idx, tool=tool, args=args,
                    )
                    db = env.tools.db
                    row.labels |= self._fab(tool, args, db, ctx)
                    row.labels |= self._viol(tool, args, db, ctx, history, msg)
                    call = ToolCall(id=tc.get("id", ""), name=tool, arguments=args, requestor="assistant")
                    row.labels |= self._resid(tool, call, row.labels)
                    resp = env.get_response(call)
                    recorded = tool_results.get(tc.get("id"))
                    if resp.error or (recorded and recorded.get("error")):
                        row.labels.add(Label.TOOL_ERROR)
                    else:
                        row.labels |= self._advance(ctx, tool, args, resp.content)
                    if tool == "transfer_to_human_agents":
                        transferred = True
                        nxt = next((m for m in messages[turn_idx + 1:] if m["role"] == "assistant"), None)
                        if nxt is None or (nxt.get("content") or "").strip() != TRANSFER_MESSAGE:
                            row.labels.add(Label.VIOL_R14_TRANSFER_MESSAGE)
                    rows.append(row)
            if role == "tool" and msg.get("content"):
                ctx.grounding.append(_norm(msg["content"]))
            history.append(msg)
        if self.gold_has_transfer and not transferred and rows:
            rows[-1].labels.add(Label.RESID_MISSED_TRANSFER)
        return rows

    @staticmethod
    def _advance(ctx: _Ctx, tool: str, args: dict, content: str) -> set[Label]:
        """Update conversation facts after a successful call; return labels only knowable post-call."""
        out = set()
        if tool in AUTH_TOOLS:
            found = content.strip().strip('"')
            if ctx.auth_user is not None and found != ctx.auth_user:
                out.add(Label.VIOL_R2_OTHER_USER)  # a second user in one conversation
            ctx.auth_user = ctx.auth_user or found
        if tool == "get_order_details":
            ctx.fetched_orders.add(args.get("order_id"))
        if tool in ONCE_PER_ORDER:
            ctx.once_done.add(args.get("order_id"))
        return out
