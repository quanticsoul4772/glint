"""Decision tools for τ² airline: deterministic answers to what the policy allows for a reservation.

Reads the live airline database (`tau2.domains.airline.data_model.FlightDB`). "Now" is fixed by the policy
("The current time is 2024-05-15 15:00:00 EST"). No language model is used.
"""
from __future__ import annotations

from datetime import datetime, timedelta

NOW = datetime(2024, 5, 15, 15, 0, 0)
ACTIONS = ("cancel", "change_flights", "change_cabin", "add_baggage", "remove_baggage", "add_insurance",
           "change_passengers", "compensation")
INSURANCE_REASONS = ("health", "weather")

RULES = {
    "flown_cancel": "If any portion of the flight has already been flown, the agent cannot help and transfer is needed.",
    "cancel": "Flight can be cancelled if any of the following is true: the booking was made within the last 24 hrs; "
              "the flight is cancelled by airline; it is a business flight; the user has travel insurance and the "
              "reason for cancellation is covered by insurance.",
    "basic_economy": "Basic economy flights cannot be modified.",
    "cabin": "Cabin cannot be changed if any flight in the reservation has already been flown. In other cases, all "
             "reservations, including basic economy, can change cabin without changing the flights.",
    "baggage": "The user can add but not remove checked bags.",
    "insurance": "The user cannot add insurance after initial booking.",
    "passengers": "The user can modify passengers but cannot modify the number of passengers.",
    "compensation": "Only compensate if the user is a silver/gold member or has travel insurance or flies business.",
}


def _flight_status(db, flight_number: str, date: str) -> str | None:
    flight = db.flights.get(flight_number)
    if flight is None or date not in flight.dates:
        return None
    return flight.dates[date].status


def _flown(db, reservation) -> list[str]:
    out = []
    for f in reservation.flights:
        status = _flight_status(db, f.flight_number, f.date)
        if status in ("landed", "flying") or datetime.fromisoformat(f.date) < NOW.replace(hour=0, minute=0):
            out.append(f"{f.flight_number} {f.date} ({status})")
    return out


def check_eligibility(db, reservation_id: str, action: str, reason: str | None = None) -> dict:
    """Whether `action` is allowed on `reservation_id` under the airline policy, the rule it rests on, and (when it is
    not allowed) any allowed change that would make it allowed."""
    if action not in ACTIONS:
        return {"allowed": False, "reasons": [f"unknown action {action!r}; one of {list(ACTIONS)}"]}
    res = db.reservations.get(reservation_id)
    if res is None:
        return {"allowed": False, "reasons": [f"reservation {reservation_id} does not exist"]}
    user = db.users[res.user_id]
    flown = _flown(db, res)
    airline_cancelled = [f"{f.flight_number} {f.date}" for f in res.flights
                         if _flight_status(db, f.flight_number, f.date) == "cancelled"]
    created = datetime.fromisoformat(res.created_at)
    details = {"reservation_id": reservation_id, "cabin": res.cabin, "insurance": res.insurance,
               "membership": user.membership, "created_at": res.created_at, "flown_segments": flown,
               "airline_cancelled_segments": airline_cancelled, "passengers": len(res.passengers),
               "status": res.status or "active"}
    reasons, rules, paths = [], [], []
    if res.status == "cancelled":
        reasons.append("reservation is already cancelled")
    elif action == "cancel":
        if flown:
            reasons.append(f"segments already flown: {flown}")
            rules.append(RULES["flown_cancel"])
        else:
            grounds = {
                "booked within the last 24 hours": NOW - created <= timedelta(hours=24),
                "flight cancelled by airline": bool(airline_cancelled),
                "business cabin": res.cabin == "business",
                "insurance covers the reason": res.insurance == "yes"
                and any(r in (reason or "").lower() for r in INSURANCE_REASONS),
            }
            details["cancellation_grounds"] = grounds
            rules.append(RULES["cancel"])
            if not any(grounds.values()):
                reasons.append("none of the cancellation conditions holds")
                paths.append("change_cabin to business is allowed (no segment flown) and would make cancellation "
                             "allowed; the user pays the fare difference")
    elif action == "change_flights":
        if res.cabin == "basic_economy":
            reasons.append("basic economy flights cannot be modified")
            rules.append(RULES["basic_economy"])
            paths.append("change_cabin is allowed for basic economy (flights unchanged); after moving to economy or "
                         "business the flights can be modified")
        if flown:
            details["note"] = "flown segments must be kept as they are"
    elif action == "change_cabin":
        rules.append(RULES["cabin"])
        if flown:
            reasons.append(f"segments already flown: {flown}")
    elif action == "remove_baggage":
        reasons.append("bags can be added but not removed")
        rules.append(RULES["baggage"])
    elif action == "add_insurance":
        reasons.append("insurance cannot be added after booking")
        rules.append(RULES["insurance"])
    elif action == "change_passengers":
        rules.append(RULES["passengers"])
        details["note"] = "names and details may change; the number of passengers may not"
    elif action == "compensation":
        rules.append(RULES["compensation"])
        if not (user.membership in ("silver", "gold") or res.insurance == "yes" or res.cabin == "business"):
            reasons.append("regular member, no insurance and not business: no compensation")
    return {"allowed": not reasons, "reasons": reasons, "rules": rules, "paths_to_allowed": paths, "details": details}
