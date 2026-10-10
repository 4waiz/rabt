"""Turn processing, authorization, approvals and tool execution.

Rules enforced here:
- The model only proposes. Read-only tools run after an authorization check; side-effect tools
  run only after the proposal's own account clicks Approve in the authenticated UI.
- An approval is bound to the account, the proposal id and a hash of the exact arguments.
- calendar.create_event additionally needs the tutor and the student to have agreed on the same,
  latest version of the lesson slot, and availability is re-checked right before booking.
- Free/busy shares busy times only, and only if each other participant opted in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from rabt_integrations import store as st
from rabt_integrations.accounts import ACCOUNTS, get_account
from rabt_integrations.google_api import GoogleApiError, GoogleNotConnected
from rabt_integrations.slots import find_slots, overlaps
from rabt_shared.agent import AgentResponse, TurnStatus
from rabt_shared.tools import (
    CreateDraftArgs,
    CreateEventArgs,
    FindMutualSlotArgs,
    ListRecentEmailArgs,
    get_tool_spec,
)

log = logging.getLogger(__name__)


class PolicyError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def args_hash(args: dict) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


def _tz(name: str) -> ZoneInfo:
    return ZoneInfo(name)


class Service:
    def __init__(self, store: st.Store, google, ai, timezone: str = "Asia/Dubai"):
        self.store = store
        self.google = google
        self.ai = ai
        self.timezone = timezone

    # ================================================================== chat turns
    def chat(self, account_id: str, message: str) -> dict:
        acct = get_account(account_id)
        conv = self.store.conversation_id(account_id)
        req = {
            "request_id": str(uuid.uuid4()),
            "conversation_id": conv,
            "account_id": account_id,
            "role": acct.role.value,
            "channel": "web",
            "message": message,
            "timezone": self.timezone,
            "sent_at": datetime.now(_tz(self.timezone)).isoformat(),
        }
        self.store.audit(account_id, "chat.message", request_id=req["request_id"], chars=len(message))
        resp = self.ai.respond(req)
        return self._handle(account_id, resp, replies=[])

    def _handle(self, account_id: str, resp: AgentResponse, replies: list[str]) -> dict:
        """Store proposals, auto-run read-only ones, and follow the turn until it needs a human."""
        self.store.cancel_proposals([str(p) for p in resp.cancelled_proposal_ids])
        if resp.reply_text and resp.status != TurnStatus.AWAITING_RESULTS:
            replies.append(resp.reply_text)
        follow_up: AgentResponse | None = None
        for p in resp.action_proposals:
            data = p.model_dump(mode="json")
            data.update(
                request_id=str(resp.request_id),
                conversation_id=str(resp.conversation_id),
                account_id=account_id,
                args_hash=args_hash(data["args"]),
            )
            # Approval requirement is decided by our own catalog, never by the model's flag alone.
            data["requires_approval"] = (
                get_tool_spec(data["tool"]).requires_approval or data["requires_approval"]
            )
            self.store.add_proposal(data)
            self.store.audit(
                account_id, "proposal.received", proposal_id=data["proposal_id"], tool=data["tool"]
            )
            if not data["requires_approval"]:
                follow_up = self._execute_and_report(account_id, data["proposal_id"], approved=False)
        if follow_up is not None:
            return self._handle(account_id, follow_up, replies)
        return {
            "replies": replies,
            "model": resp.model,
            "mode": resp.mode,
            "ai_mode": self.ai.mode,
            "pending": [p for p in self.store.list_proposals(account_id) if p["status"] == "pending"],
            "lesson": self.lesson_view(account_id),
        }

    # ================================================================== approvals
    def approve(self, account_id: str, proposal_id: str) -> dict:
        p = self._own_pending(account_id, proposal_id)
        if p["tool"] == "calendar.create_event":
            self._check_lesson_gate(p)  # raises PolicyError -> 409 for the UI; proposal stays pending
        if not self.store.claim_proposal(
            proposal_id, "pending", "approved", approved_by=account_id, approved_hash=p["args_hash"]
        ):
            raise PolicyError("proposal_not_pending", "This proposal was already handled.")
        self.store.audit(account_id, "proposal.approved", proposal_id=proposal_id, tool=p["tool"])
        resp = self._execute_and_report(account_id, proposal_id, approved=True)
        return self._handle(account_id, resp, replies=[])

    def reject(self, account_id: str, proposal_id: str) -> dict:
        p = self._own_pending(account_id, proposal_id)
        if not self.store.claim_proposal(proposal_id, "pending", "rejected"):
            raise PolicyError("proposal_not_pending", "This proposal was already handled.")
        self.store.audit(account_id, "proposal.rejected", proposal_id=proposal_id, tool=p["tool"])
        if p["tool"] == "calendar.create_event":
            lesson = self._lesson_for(p)
            if lesson:
                self.store.update_lesson(lesson["lesson_id"], state=st.REJECTED, last_action_by=account_id)
        resp = self.ai.tool_result(self._result_req(p, "rejected"))
        return self._handle(account_id, resp, replies=[])

    def _own_pending(self, account_id: str, proposal_id: str) -> dict:
        p = self.store.get_proposal(proposal_id)
        if p is None or p["account_id"] != account_id:  # never reveal other users' proposals
            raise PolicyError("not_found", "No such proposal for this account.")
        if p["status"] != "pending":
            raise PolicyError("proposal_not_pending", f"Proposal is already {p['status']}.")
        return p

    # ================================================================== execution
    def _result_req(
        self, p: dict, status: str, result: dict | None = None, error: dict | None = None
    ) -> dict:
        body = {
            "request_id": p["request_id"],
            "conversation_id": p["conversation_id"],
            "account_id": p["account_id"],
            "proposal_id": p["proposal_id"],
            "tool": p["tool"],
            "status": status,
        }
        if result is not None:
            body["result"] = result
        if error is not None:
            body["error"] = error
        return body

    def _execute_and_report(self, account_id: str, proposal_id: str, approved: bool) -> AgentResponse:
        p = self.store.get_proposal(proposal_id)
        assert p is not None
        from_status = "approved" if approved else "pending"
        if approved and p["approved_hash"] != p["args_hash"]:
            raise PolicyError("approval_mismatch", "Approval does not match the proposal arguments.")
        if not self.store.claim_proposal(proposal_id, from_status, "executing"):
            raise PolicyError("proposal_not_pending", "This proposal is already being handled.")
        try:
            result = self._run_tool(account_id, p)
            status, error = "succeeded", None
            self.store.claim_proposal(proposal_id, "executing", "succeeded", result=result)
        except PolicyError as exc:
            status, result, error = "denied", None, {"code": exc.code, "message": str(exc)[:500]}
            self.store.claim_proposal(proposal_id, "executing", "denied", error=error)
        except GoogleNotConnected as exc:
            status, result = "failed", None
            error = {"code": "google_not_connected", "message": str(exc)[:500], "retryable": True}
            self.store.claim_proposal(proposal_id, "executing", "failed", error=error)
        except GoogleApiError as exc:
            status, result = "failed", None
            error = {"code": exc.code, "message": str(exc)[:500], "retryable": exc.retryable}
            self.store.claim_proposal(proposal_id, "executing", "failed", error=error)
        except Exception as exc:  # never report an unknown failure as success
            log.exception("tool %s failed", p["tool"])
            status, result = "failed", None
            error = {"code": "execution_error", "message": type(exc).__name__, "retryable": False}
            self.store.claim_proposal(proposal_id, "executing", "failed", error=error)
        self.store.audit(
            account_id,
            f"tool.{status}",
            proposal_id=proposal_id,
            tool=p["tool"],
            error=(error or {}).get("code"),
        )
        if p["tool"] == "calendar.create_event" and status != "succeeded":
            lesson = self._lesson_for(p)
            if lesson:
                self.store.update_lesson(
                    lesson["lesson_id"], state=st.FAILED, note=(error or {}).get("message")
                )
        return self.ai.tool_result(self._result_req(p, status, result, error))

    def _run_tool(self, account_id: str, p: dict) -> dict:
        tool = p["tool"]
        if tool == "gmail.list_recent":
            a = ListRecentEmailArgs.model_validate(p["args"])
            msgs = self.google.list_recent(
                account_id, a.max_results, a.query, a.unread_only
            )  # own mailbox only
            return {"messages": msgs}
        if tool == "calendar.find_mutual_slot":
            return self._find_mutual_slot(account_id, FindMutualSlotArgs.model_validate(p["args"]))
        if tool == "gmail.create_draft":
            a = CreateDraftArgs.model_validate(p["args"])
            to = list(a.to_emails) + [self._email_of(x) for x in a.to_account_ids]
            return {"draft_id": self.google.create_draft(account_id, to, a.subject, a.body)}
        if tool == "calendar.create_event":
            return self._create_event(account_id, p)
        raise PolicyError("tool_not_supported", f"{tool} is not enabled in this backend yet.")

    def _email_of(self, account_id: str) -> str:
        email = get_account(account_id).google_email
        if not email:
            raise PolicyError("recipient_unknown", f"No email configured for {account_id}.")
        return email

    def shares_freebusy(self, account_id: str) -> bool:
        return self.store.get_setting(account_id, "share_freebusy", "0") == "1"

    def _busy_union(self, ids: list[str], start: datetime, end: datetime, tz: str):
        busy = []
        for pid in ids:
            busy.extend(self.google.busy(pid, start, end, tz))  # each person's own token
        return busy

    def _find_mutual_slot(self, account_id: str, a: FindMutualSlotArgs) -> dict:
        ids = list(a.participant_account_ids)
        if account_id not in ids:
            raise PolicyError("requester_not_participant", "The requester must be a participant.")
        for pid in ids:
            if pid not in ACCOUNTS:
                raise PolicyError("unknown_participant", f"{pid} is not a registered account.")
            if pid != account_id and not self.shares_freebusy(pid):
                raise PolicyError("freebusy_not_shared", f"{pid} has not agreed to share free/busy.")
        tz = _tz(a.timezone)
        busy = self._busy_union(ids, a.window_start, a.window_end, a.timezone)
        found = find_slots(
            a.window_start, a.window_end, busy, timedelta(minutes=a.duration_minutes), a.max_slots
        )
        slots = [
            {"start": s.astimezone(tz).isoformat(), "end": e.astimezone(tz).isoformat()} for s, e in found
        ]
        tutor = next((i for i in ids if get_account(i).role.value == "tutor"), None)
        student = next((i for i in ids if get_account(i).role.value == "student"), None)
        if tutor and student and slots:
            self.store.create_lesson(student, tutor, slots, a.timezone, by=account_id)
        return {"slots": slots}

    # ================================================================== lesson workflow
    def _lesson_for(self, p: dict) -> dict | None:
        return self.store.latest_lesson(p["account_id"])

    def _check_lesson_gate(self, p: dict) -> dict:
        a = CreateEventArgs.model_validate(p["args"])
        lesson = self.store.latest_lesson(p["account_id"])
        if not lesson or lesson["state"] != st.AWAITING_APPROVAL:
            raise PolicyError("lesson_not_agreed", "Tutor and student have not both agreed on a slot yet.")
        v = lesson["version"]
        if lesson["tutor_approved_version"] != v or lesson["student_approved_version"] != v:
            raise PolicyError("lesson_not_agreed", "The latest slot version is missing an approval.")
        if (
            datetime.fromisoformat(lesson["slot_start"]) != a.start
            or datetime.fromisoformat(lesson["slot_end"]) != a.end
        ):
            raise PolicyError("slot_mismatch", "This booking time differs from the agreed slot.")
        participants = {lesson["student_id"], lesson["tutor_id"]}
        if {p["account_id"], *a.attendee_account_ids} != participants:
            raise PolicyError("attendee_mismatch", "The invitation must be between the tutor and student.")
        return lesson

    def _create_event(self, account_id: str, p: dict) -> dict:
        lesson = self._check_lesson_gate(p)
        a = CreateEventArgs.model_validate(p["args"])
        self.store.update_lesson(lesson["lesson_id"], state=st.APPROVED, last_action_by=account_id)
        everyone = [account_id, *a.attendee_account_ids]
        if overlaps((a.start, a.end), self._busy_union(everyone, a.start, a.end, a.timezone)):
            raise PolicyError("slot_conflict", "Someone became busy at that time. Pick another slot.")
        event_id = uuid.UUID(p["proposal_id"]).hex  # deterministic: retries cannot double-book
        body = {
            "summary": a.title,
            "description": a.description or "",
            "start": {"dateTime": a.start.isoformat(), "timeZone": a.timezone},
            "end": {"dateTime": a.end.isoformat(), "timeZone": a.timezone},
            "attendees": [{"email": self._email_of(x)} for x in a.attendee_account_ids],
        }
        if a.location:
            body["location"] = a.location
        self.google.create_event(account_id, event_id, body)
        verified = self.google.get_event(account_id, event_id)  # report what Google actually has
        result = {
            "event_id": verified["id"],
            "start": verified["start"]["dateTime"],
            "end": verified["end"]["dateTime"],
            "invited_account_ids": list(a.attendee_account_ids),
            "html_link": verified.get("htmlLink"),
        }
        self.store.update_lesson(
            lesson["lesson_id"],
            state=st.EXECUTED,
            event_id=verified["id"],
            event_link=verified.get("htmlLink"),
        )
        return result

    def lesson_view(self, account_id: str) -> dict | None:
        lesson = self.store.latest_lesson(account_id)
        if not lesson:
            return None
        me = get_account(account_id)
        v = lesson["version"]
        return {
            **{
                k: lesson[k]
                for k in (
                    "lesson_id",
                    "state",
                    "version",
                    "options",
                    "slot_start",
                    "slot_end",
                    "timezone",
                    "event_link",
                    "note",
                    "last_action_by",
                )
            },
            "my_role": me.role.value,
            "tutor_agreed": lesson["tutor_approved_version"] == v,
            "student_agreed": lesson["student_approved_version"] == v,
        }

    def lesson_action(self, account_id: str, lesson_id: str, action: str, start: str | None = None) -> dict:
        lesson = self.store.get_lesson(lesson_id)
        me = get_account(account_id)
        if not lesson or account_id not in (lesson["student_id"], lesson["tutor_id"]):
            raise PolicyError("not_found", "No such lesson for this account.")
        if lesson["state"] not in st.OPEN_STATES:
            raise PolicyError("lesson_closed", f"Lesson is {lesson['state']}.")
        is_tutor = me.role.value == "tutor"
        v = lesson["version"]
        if action == "cancel":
            lesson = self.store.update_lesson(lesson_id, state=st.CANCELLED, last_action_by=account_id)
        elif action in ("propose", "counter"):
            slot = next((o for o in lesson["options"] if o["start"] == start), None)
            if slot is None:
                raise PolicyError("slot_unknown", "Choose one of the mutually free options.")
            # A new slot is a new version: every earlier approval is invalidated.
            fields = {
                "slot_start": slot["start"],
                "slot_end": slot["end"],
                "version": v + 1,
                "tutor_approved_version": v + 1 if is_tutor else None,
                "student_approved_version": None if is_tutor else v + 1,
                "state": st.PROPOSED if is_tutor else st.COUNTERED,
                "last_action_by": account_id,
            }
            lesson = self.store.update_lesson(lesson_id, **fields)
        elif action == "accept":
            if not lesson["slot_start"]:
                raise PolicyError("nothing_to_accept", "No slot has been proposed yet.")
            key = "tutor_approved_version" if is_tutor else "student_approved_version"
            lesson = self.store.update_lesson(lesson_id, **{key: v}, last_action_by=account_id)
            if lesson["tutor_approved_version"] == v and lesson["student_approved_version"] == v:
                lesson = self.store.update_lesson(lesson_id, state=st.AWAITING_APPROVAL)
        else:
            raise PolicyError("bad_action", "Unknown lesson action.")
        self.store.audit(account_id, f"lesson.{action}", lesson_id=lesson_id, version=lesson["version"])
        return self.lesson_view(account_id)  # type: ignore[return-value]
