"""Airline and retail write rules added in phase 2, from each domain's policy (τ² `policy.md`).

- airline send_certificate ("Refunds and Compensation"): a certificate only for a silver/gold member, a reservation
  with travel insurance or a business reservation, with a cancelled flight ($100 x passengers) or a delayed flight
  ($50 x passengers, and only after the reservation is changed or cancelled). The call names no reservation, so the
  gate decides only what the database settles. Denied: a user no reservation of whom qualifies; an amount that is not
  $100 or $50 x the passengers of a qualifying reservation; a $50 x passengers that only an untouched delayed
  reservation (not cancelled, still holding the delayed flight) explains. Allowed as policy-checked: $100 x the
  passengers of a qualifying reservation with a cancelled flight. Anything else (a delay after a change the database
  no longer shows) cannot be settled and is allowed under no_rule_yet, never as policy-checked.
- airline book_reservation ("Book flight"): at most five passengers, at most one certificate, one credit card and three
  gift cards, and nonfree_baggages from the membership's free allowance.
- retail modify_user_address: every required field present. (τ²'s task 110 sets the address already on file, so a
  same-address write is not denied.)
"""
from __future__ import annotations

from glint.gates.core import NO_RULE, GateResult, allow, deny

AIRLINE = "airline policy.md"
FREE_BAGS = {"regular": {"basic_economy": 0, "economy": 1, "business": 2},
             "silver": {"basic_economy": 1, "economy": 2, "business": 3},
             "gold": {"basic_economy": 2, "economy": 3, "business": 4}}
PAYMENT_LIMITS = {"certificate": 1, "credit_card": 1, "gift_card": 3}


def _status(db, number: str, day: str) -> str | None:
    flight = db.flights.get(number)
    return flight.dates[day].status if flight is not None and day in flight.dates else None


def send_certificate(db, a: dict) -> GateResult:
    source = f"{AIRLINE}: Refunds and Compensation"
    user = db.users.get(a.get("user_id", ""))
    if user is None:
        return allow("airline.certificate.unknown_user", source, reason="no such user: τ²'s tool rejects the call")
    member = str(user.membership)
    reservations = [r for r in db.reservations.values() if r.user_id == user.user_id]
    covered = [r for r in reservations if member in ("silver", "gold") or r.insurance == "yes" or r.cabin == "business"]
    details = {"membership": member, "qualifying_reservations": [r.reservation_id for r in covered]}
    if not covered:
        return deny("airline.certificate.not_eligible", source, f"user {user.user_id} is a {member} member with no "
                    "reservation that has travel insurance or flies business; the policy gives such a user no "
                    "compensation", "Do not send a certificate; tell the user the policy offers no compensation "
                    "here.", details)
    amount = float(a.get("amount") or 0)
    allowed = sorted({k * len(r.passengers) for r in covered for k in (50.0, 100.0)})
    cancelled = {r.reservation_id: 100.0 * len(r.passengers) for r in covered
                 if any(_status(db, f.flight_number, f.date) == "cancelled" for f in r.flights)}
    delayed = {r.reservation_id: 50.0 * len(r.passengers) for r in covered
               if any(_status(db, f.flight_number, f.date) == "delayed" for f in r.flights)
               and str(getattr(r, "status", "") or "") != "cancelled"}
    details |= {"cancelled_flight_amounts": cancelled, "untouched_delayed_amounts": delayed}
    if amount not in allowed:
        return deny("airline.certificate.amount", source, f"${amount:g} is neither $100 nor $50 times the passengers "
                    f"of a qualifying reservation (possible: {allowed})", "Use $100 x passengers for a cancelled "
                    "flight, $50 x passengers for a delayed one (after the change or cancellation).", details)
    if amount in cancelled.values():
        return allow("airline.certificate.cancelled_flight", source, details)
    # amounts another reading could explain: any amount of a reservation that is not an untouched delay, and $100 x
    # passengers of the delayed ones
    others = {k * len(r.passengers) for r in covered for k in (50.0, 100.0) if r.reservation_id not in delayed}
    others |= {100.0 * len(r.passengers) for r in covered if r.reservation_id in delayed}
    if amount in delayed.values() and amount not in others:
        return deny("airline.certificate.delay_unchanged", source, "a delay is compensated only after the "
                    "reservation is changed or cancelled; the reservation with the delayed flight is not cancelled "
                    "and still holds that flight", "Change or cancel the reservation as the user asks first; if the "
                    "user only wants compensation, the policy offers none for a delay.", details)
    return allow(NO_RULE, source, details, reason="whether this certificate is due (a delay with a change or "
                 "cancellation, or a flight no longer in the reservation) is not settled by the database")


def book_reservation(db, a: dict) -> GateResult:
    source = f"{AIRLINE}: Book flight"
    user = db.users.get(a.get("user_id", ""))
    if user is None:
        return allow("airline.book.unknown_user", source, reason="no such user: τ²'s tool rejects the call")
    passengers = a.get("passengers") or []
    reasons, details = [], {"membership": str(user.membership), "passengers": len(passengers)}
    if len(passengers) > 5:
        reasons.append(f"{len(passengers)} passengers; a reservation has at most five")
    kinds: dict[str, int] = {}
    for p in a.get("payment_methods") or []:
        pid = p.get("payment_id") if isinstance(p, dict) else getattr(p, "payment_id", None)
        method = user.payment_methods.get(pid)
        if method is not None:
            kinds[method.source] = kinds.get(method.source, 0) + 1
    details["payment_kinds"] = kinds
    over = [f"{n} {kind}s (at most {PAYMENT_LIMITS[kind]})" for kind, n in kinds.items()
            if n > PAYMENT_LIMITS.get(kind, n)]
    if over:
        reasons.append("too many payment methods of one kind: " + ", ".join(over))
    allowance = FREE_BAGS.get(str(user.membership), {}).get(str(a.get("cabin")))
    if allowance is not None:
        total = int(float(a.get("total_baggages") or 0))
        nonfree = int(float(a.get("nonfree_baggages") or 0))
        wanted = max(0, total - allowance * len(passengers))
        details |= {"free_per_passenger": allowance, "nonfree_wanted": wanted}
        if nonfree != wanted:
            reasons.append(f"nonfree_baggages {nonfree} != {wanted} ({total} bags, {allowance} free per "
                           f"{a.get('cabin')} passenger for a {user.membership} member)")
    if reasons:
        return deny("airline.book", source, "; ".join(reasons), "Book it within the policy: at most five "
                    "passengers; at most one certificate, one credit card and three gift cards; nonfree_baggages as "
                    "the verdict says.", details)
    return allow("airline.book", source, details)


def modify_user_address(db, a: dict) -> GateResult:
    source = "retail policy.md: modify their default user address"
    user = db.users.get(a.get("user_id", ""))
    if user is None:
        return allow("retail.user_address.unknown_user", source, reason="no such user: τ²'s tool rejects the call")
    fields = ("address1", "city", "state", "country", "zip")
    missing = [f for f in fields if not str(a.get(f) or "").strip()]
    if missing:
        return deny("retail.user_address.missing", source, f"the new address has no {missing}",
                    "Ask the user for the full address (address1, address2 or '', city, state, country, zip).",
                    {"missing": missing})
    return allow("retail.user_address", source, {"user_id": a.get("user_id")})
