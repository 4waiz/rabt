# Rabt integrations backend — setup (owner: Huda)

FastAPI backend + one-page web UI. Google Gmail/Calendar per person, tutor/student lesson workflow,
approval gate, client for Awaiz's local AI engine (or a clearly labelled mock).

## 1. Add to `.env` (repo root, never committed)

```ini
RABT_AI_MODE=real                      # "mock" = rule-based stand-in, NO LLM (labelled in the UI)
RABT_AI_BASE_URL=http://127.0.0.1:8765
RABT_INTERNAL_API_KEY=<same value as the AI engine>
RABT_SESSION_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(48))">
RABT_PIN_HUDA=<pick a PIN>
RABT_PIN_AWAIZ=<pick a PIN>
RABT_GOOGLE_EMAIL_HUDA=kanbanrabt@gmail.com
RABT_GOOGLE_EMAIL_AWAIZ=<Awaiz's Gmail>
# optional: RABT_BACKEND_DATA_DIR (default %USERPROFILE%\.rabt-backend), RABT_GOOGLE_CLIENT_SECRET_FILE
```

## 2. Google (one Cloud project, Desktop OAuth client)

Gmail API + Calendar API enabled, consent screen External/Testing with both people as test users,
Desktop client JSON saved in `%USERPROFILE%\.rabt-backend\secrets\`. The downloaded
`client_secret_*.json` is picked up as-is (or name it `google_client_secret.json`).

Scopes (least privilege): `gmail.readonly` (metadata + snippet of own mail), `gmail.compose` (drafts,
never sent), `calendar.freebusy` (busy times only), `calendar.events` (create the approved event).

Each person connects their **own** account (browser opens; they sign in themselves):

```powershell
uv run python -m rabt_integrations.google_auth huda    # sign in as kanbanrabt@gmail.com
uv run python -m rabt_integrations.google_auth awaiz   # Awaiz signs in as himself
```

The token is refused and deleted if the signed-in Google address differs from
`RABT_GOOGLE_EMAIL_<ID>`. Tokens live in
`%USERPROFILE%\.rabt-backend\secrets\google_token_<id>.json`, separate from the AI engine's
`RABT_DATA_DIR`, so they are never stored next to the engine's files.

## 3. Run

Run on the same machine as the AI engine (it listens on loopback only).

```powershell
uv sync
uv run rabt-integrations        # http://127.0.0.1:8800
```

## 4. Demo flow

1. Both sign in (two browser profiles / one normal + one private window), tick
   "Share my free/busy times", click "Test my Google connection" (real API smoke test).
2. Huda: "Can we do tuition tomorrow evening? Check when we're both free." → real free/busy of both
   accounts → mutual options appear in the Lesson planner.
3. Awaiz: picks an option (**propose**). Huda: **Accept** or **counter** with another option
   (a new version; earlier approvals are invalidated, Awaiz must accept again).
4. Huda: "Ask Rabt to book it" → the AI proposes `calendar.create_event` + `gmail.create_draft` →
   previews shown → **Approve** each. The backend re-checks free/busy, creates the event with a
   deterministic id (no duplicates), reads it back from Google and links it.

## Safety

- Model output never executes anything: read-only tools pass an authorization check; side-effect
  tools need an Approve click by the proposal's own account, bound to a hash of the exact args.
- Booking also requires both people to have agreed to the latest slot version, and attendees must
  be exactly tutor + student.
- Free/busy needs each other participant's opt-in; only busy intervals are fetched, never titles.
- Gmail returns sender/subject/≤500-char snippet of the requester's own mailbox only.
- Dashboard login per identity (PIN, signed session cookie); every endpoint checks it; audit log.

## Tests

`uv run pytest tests/integrations` — **all mocked** (fake Google gateway + mock AI). They prove the
logic, not the real integrations; real Google is verified with the "Test my Google connection"
button and the demo flow.
