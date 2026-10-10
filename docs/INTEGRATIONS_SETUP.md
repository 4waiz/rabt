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

## WhatsApp (official Cloud API)

Rabt uses the **WhatsApp Business Platform Cloud API** only. It does no WhatsApp Web automation
and reads no personal chats.

**What the backend does:**
- **Webhook check.** It answers Meta's verification request (`hub.challenge`).
- **Signatures.** It rejects any delivery whose `X-Hub-Signature-256` is not a valid HMAC-SHA256 of
  the raw body with the App Secret.
- **No double processing.** Meta retries deliveries; each message id is processed once.
- **Who is writing.** The sender is identified by the number they write from, never by message
  text. Numbers that are not registered are ignored and never answered.
- **AI turn.** Text goes to the AI engine with `channel="whatsapp"`. The reply is sent back.
  Anything that needs approval is listed, with a pointer to the Rabt app. Approvals happen only
  in the logged-in web app.
- **Opt-out.** `STOP` opts a person out and `START` opts them back in.
- **24-hour rule.** Free-form messages go only to a registered, opted-in person who has messaged
  Rabt in the last 24 hours. Outside that window WhatsApp requires an approved template; Rabt has
  none, so it refuses.
- **`messaging.send_message`.** After approval in the web app it sends to registered accounts only.
- **Errors.** Rate limits and 5xx errors are retried with backoff. Provider error codes are logged;
  message text never is.
- **Delivery status.** Updates (sent, delivered, read, failed) are recorded, and failures appear
  in Activity.
- **Separate webhook port.** The webhook also runs alone on `RABT_WA_WEBHOOK_PORT` (default 8801).
  **Tunnel only that port.** It serves nothing but `/webhooks/whatsapp`; the dashboard (8800),
  the AI engine (8765) and Ollama (11434) stay private.

### `.env`

```ini
RABT_WA_PHONE_NUMBER_ID=<Phone number ID from WhatsApp > API Setup>
RABT_WA_ACCESS_TOKEN=<access token; the temporary one expires after ~24 h>
RABT_WA_APP_SECRET=<App settings > Basic > App secret>
RABT_WA_VERIFY_TOKEN=<any long random string you also paste into the webhook form>
RABT_WA_GRAPH_VERSION=v21.0          # set to the version shown on the API Setup page
RABT_WA_NUMBER_HUDA=<Huda's WhatsApp number, international format>
RABT_WA_NUMBER_AWAIZ=<Awaiz's WhatsApp number>
```

### Meta setup (test number first)

1. Go to https://developers.facebook.com, then **My Apps → Create app**. Choose the
   **Business** type and add the **WhatsApp** product.
2. Open **WhatsApp → API Setup**. Meta provides a free **test number**. Copy the **Phone number
   ID** and generate an access token.
3. Under **To**, add each person's own WhatsApp number as a recipient and confirm it with the code
   WhatsApp sends. The test number can only message verified recipients.
4. Start the backend. Then expose **port 8801 only** over HTTPS, for example
   `cloudflared tunnel --url http://127.0.0.1:8801`.
5. Open **WhatsApp → Configuration → Webhook**:
   - **Callback URL:** `https://<tunnel>/webhooks/whatsapp`
   - **Verify token:** your `RABT_WA_VERIFY_TOKEN`
   - Click **Verify and save**, then subscribe to the **messages** field.
6. From your phone, message the test number: "check my email". Rabt replies in WhatsApp.

### About using your existing business number

A number already registered on the WhatsApp Business app does not move to the Cloud API
automatically. The options are:

- **Coexistence.** Keep the app and add the API. This is only possible through Meta's Embedded
  Signup, run by a Tech Provider or Solution Partner. The number must be recently active, the
  Business app must be up to date, and API conversations are billed.
- **Full migration.** Move the number off the app to the Cloud API. The app no longer works for
  it.

Start with the test number. Decide on the real number separately, because it changes how that
number works.
