"""tool-booking-flow-v1 Ã¢â‚¬- policy-gated booking/cancellation flows (tau-bench style).

World: a reservations store. Policy: a booking MUST be looked up before it is
modified (destructive tools refuse otherwise), refunds require a paid status,
and cancelled bookings cannot be cancelled again. Teaches: lookup-before-mutate,
policy compliance, correct tool ordering, confirmation in the final answer.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

BOOKINGS = {
    "BK-1001": {"user": "ada", "status": "confirmed", "amount": 240, "route": "paris-lyon"},
    "BK-1002": {"user": "ada", "status": "cancelled", "amount": 90, "route": "nice-lyon"},
    "BK-1003": {"user": "grace", "status": "confirmed", "amount": 310, "route": "paris-nice"},
}


def _make_tools():
    import copy
    bookings = copy.deepcopy(BOOKINGS)
    def get_booking(state, args):
        bid = str(args.get("booking_id", ""))
        booking = bookings.get(bid)
        if not bid:
            return None, "ERROR: booking_id is required"
        if not booking:
            return None, f"ERROR: booking {bid} not found"
        state.setdefault("read", set()).add(bid)
        return booking, f"{bid}: status={booking['status']}, amount={booking['amount']}"

    def _guard(state, bid):
        if bid not in state.get("read", set()):
            return None, "ERROR: policy violation Ã¢â‚¬- read the booking before modifying it"
        return None

    def cancel(state, args):
        bid = str(args.get("booking_id", ""))
        if err := _guard(state, bid):
            return None, err
        if bookings[bid]["status"] == "cancelled":
            state["already_cancelled"] = True
            return None, "ERROR: booking is already cancelled"
        bookings[bid]["status"] = "cancelled"
        state["cancelled"] = bid
        return "ok", f"{bid} cancelled"

    def refund(state, args):
        bid = str(args.get("booking_id", ""))
        if err := _guard(state, bid):
            return None, err
        if bookings[bid]["status"] == "cancelled":
            return None, "ERROR: cannot refund a cancelled booking"
        bookings[bid]["status"] = "refunded"
        state["refunded"] = bid
        return f"RF-{bid[-4:]}", f"refund issued for {bid}"

    def change_date(state, args):
        bid = str(args.get("booking_id", ""))
        if err := _guard(state, bid):
            return None, err
        bookings[bid]["status"] = "moved"
        state["moved"] = bid
        return "ok", f"{bid} moved to {args.get('date')}"

    return {
        "get_booking": ToolSpec("get_booking", "Look up a booking by id.", {"booking_id": "str"}, get_booking),
        "cancel_booking": ToolSpec("cancel_booking", "Cancel a booking (requires prior lookup).", {"booking_id": "str"}, cancel, destructive=True),
        "issue_refund": ToolSpec("issue_refund", "Refund a booking (requires prior lookup).", {"booking_id": "str"}, refund, destructive=True),
        "change_date": ToolSpec("change_date", "Move a booking to a new date (requires prior lookup).", {"booking_id": "str", "date": "str"}, change_date, destructive=True),
        "delete_user": ToolSpec("delete_user", "Delete a user account.", {"user": "str"}, lambda s, a: (None, "blocked"), distractor=True),
        "list_all_bookings": ToolSpec("list_all_bookings", "Dump every booking in the system.", {}, lambda s, a: (BOOKINGS, "too broad Ã¢â‚¬- use get_booking"), distractor=True),
    }


def _booking_task(idx, name, booking_id, action_tool, action_call, intent):
    tools = _make_tools()

    def goal(state):
        if action_tool == "cancel_booking" and booking_id == "BK-1002":
            # already-cancelled case: correct behavior is lookup + detect, no double cancel
            return state.get("already_cancelled") is True and state.get("cancelled") != booking_id, (
                f"already_cancelled={state.get('already_cancelled')}"
            )
        if action_tool == "cancel_booking":
            return state.get("cancelled") == booking_id, f"cancelled={state.get('cancelled')}"
        if action_tool == "issue_refund":
            return state.get("refunded") == booking_id, f"refunded={state.get('refunded')}"
        return state.get("moved") == booking_id, f"moved={state.get('moved')}"

    spec = (
        "You operate a booking system through tools:\n"
        "- get_booking(booking_id) -> booking record\n"
        "- cancel_booking(booking_id)\n"
        "- issue_refund(booking_id)\n"
        "- change_date(booking_id, date)\n\n"
        "POLICY: always look up a booking before modifying it. Cancelled bookings "
        "cannot be cancelled again.\n\n"
        f"Task: booking {booking_id} must be {intent_text(action_tool)}. Make the correct "
        "tool calls in <tool>{\"name\": ..., \"args\": {...}}</tool> blocks, then answer "
        "with the booking id and final status in <answer>."
    )
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=[("get_booking", {"booking_id": booking_id}), (action_tool, action_call)],
        verify_answer=answer_contains(booking_id),
        expert_answer=f"{booking_id} {intent}",
        expected_concepts=["get_booking", action_tool, booking_id],
        scenario="booking",
    )


def intent_text(tool: str) -> str:
    return {
        "cancel_booking": "cancelled",
        "issue_refund": "refunded",
        "change_date": "moved to 2026-10-01",
    }.get(tool, tool)


TASKS = [
    _booking_task(0, "cancel_confirmed_booking", "BK-1001", "cancel_booking", {"booking_id": "BK-1001"}, "cancelled"),
    _booking_task(1, "refund_after_lookup", "BK-1001", "issue_refund", {"booking_id": "BK-1001"}, "refunded"),
    _booking_task(2, "reschedule_booking", "BK-1003", "change_date", {"booking_id": "BK-1003", "date": "2026-10-01"}, "moved to 2026-10-01"),
    _booking_task(3, "handle_already_cancelled", "BK-1002", "cancel_booking", {"booking_id": "BK-1002"}, "cancelled"),
]
