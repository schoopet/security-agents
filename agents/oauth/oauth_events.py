"""
Minimal ADK OAuth event inspector.

Triggers the adk_request_credential interactive OAuth flow and prints
every ADK Event exchanged between runner and client as formatted JSON.

Usage (Gemini API key):
    export OAUTH_CLIENT_ID=<your-google-oauth-client-id>
    export OAUTH_CLIENT_SECRET=<your-google-oauth-client-secret>
    export GOOGLE_API_KEY=<your-gemini-api-key>
    python oauth_events.py

Usage (Vertex AI / Application Default Credentials):
    export OAUTH_CLIENT_ID=<your-google-oauth-client-id>
    export OAUTH_CLIENT_SECRET=<your-google-oauth-client-secret>
    export GOOGLE_CLOUD_PROJECT=<your-gcp-project>
    export GOOGLE_CLOUD_LOCATION=us-central1   # optional, defaults to us-central1
    python oauth_events.py
"""

import asyncio
import json
import os

# ---------------------------------------------------------------------------
# Auto-configure Vertex AI when no Gemini API key is present.
# Must be set before any google.adk / google.genai imports.
# ---------------------------------------------------------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mmontan-ml-dev")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowAuthorizationCode
from google.adk.agents import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth import AuthConfig, AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# ---------------------------------------------------------------------------
# OAuth configuration (populate via environment variables)
# ---------------------------------------------------------------------------

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8080/callback")

# OAuth2 scheme – Google calendar scope (any protected Google API works)
AUTH_SCHEME = OAuth2(
    flows=OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl="https://accounts.google.com/o/oauth2/auth",
            tokenUrl="https://oauth2.googleapis.com/token",
            scopes={
                "https://www.googleapis.com/auth/calendar.readonly": "Read Google Calendar"
            },
        )
    )
)

AUTH_CREDENTIAL = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id=OAUTH_CLIENT_ID or "demo-client-id",
        client_secret=OAUTH_CLIENT_SECRET or "demo-client-secret",
        # When real credentials are absent, supply a placeholder auth_uri so ADK
        # skips URL generation (which requires authlib + valid client_id) and still
        # yields the adk_request_credential event. Replace this with a proper OAuth
        # client_id/secret in production; remove the auth_uri line entirely.
        auth_uri=(
            None
            if (OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET)
            else (
                "https://accounts.google.com/o/oauth2/auth"
                "?client_id=demo-client-id"
                "&response_type=code"
                "&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar.readonly"
                "&access_type=offline"
                "&prompt=consent"
                "&state=demo-state-1234"
            )
        ),
    ),
)

# ---------------------------------------------------------------------------
# Tool that requires OAuth
# ---------------------------------------------------------------------------

def list_calendar_events(tool_context: ToolContext) -> dict:
    """List the user's upcoming Google Calendar events (requires OAuth)."""
    auth_config = AuthConfig(
        auth_scheme=AUTH_SCHEME,
        raw_auth_credential=AUTH_CREDENTIAL,
    )

    # Check whether the client already provided credentials via FunctionResponse.
    exchanged = tool_context.get_auth_response(auth_config)

    if not exchanged:
        # No credentials yet – request them from the client.
        # This causes ADK to yield an adk_request_credential FunctionCall event.
        tool_context.request_credential(auth_config)
        return {"status": "pending", "message": "OAuth credentials required."}

    # Credentials are available – in a real tool you would call the Google API here.
    access_token = getattr(getattr(exchanged, "oauth2", None), "access_token", None)
    return {
        "status": "authenticated",
        "message": "Credentials received. Would call Google Calendar API here.",
        "access_token_preview": f"{access_token[:12]}…" if access_token else "(unavailable)",
    }


calendar_tool = FunctionTool(func=list_calendar_events)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = LlmAgent(
    model="gemini-2.0-flash",
    name="calendar_agent",
    instruction="You help users view their Google Calendar events. When asked, call list_calendar_events.",
    tools=[calendar_tool],
)

# ---------------------------------------------------------------------------
# Event helpers (verbatim from ADK docs)
# ---------------------------------------------------------------------------

def get_auth_request_function_call(event: Event) -> types.FunctionCall | None:
    if not event.content or not event.content.parts:
        return None
    for part in event.content.parts:
        if (
            part
            and part.function_call
            and part.function_call.name == "adk_request_credential"
            and event.long_running_tool_ids
            and part.function_call.id in event.long_running_tool_ids
        ):
            return part.function_call
    return None


def get_auth_config(function_call: types.FunctionCall) -> AuthConfig:
    auth_config = (function_call.args or {}).get("authConfig")
    if not auth_config:
        raise ValueError(f"No authConfig in function call: {function_call}")
    if isinstance(auth_config, dict):
        return AuthConfig.model_validate(auth_config)
    if isinstance(auth_config, AuthConfig):
        return auth_config
    raise ValueError(f"Unexpected authConfig type: {type(auth_config)}")

# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------

def strip_nulls(obj):
    """Recursively remove null/None values from dicts and lists."""
    if isinstance(obj, dict):
        return {k: strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_nulls(i) for i in obj]
    return obj


def redact(obj, keys=("clientId", "clientSecret", "client_id", "client_secret")):
    """Replace credential values with a placeholder string."""
    if isinstance(obj, dict):
        return {
            k: "<redacted>" if k in keys else redact(v, keys)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(i, keys) for i in obj]
    return obj


def to_wire(raw: dict) -> dict:
    """Strip nulls and redact credentials from a serialised payload."""
    return redact(strip_nulls(raw))


def event_to_wire(event: Event) -> dict:
    raw = json.loads(event.model_dump_json(by_alias=True))
    return to_wire(raw)


def content_to_wire(content: types.Content) -> dict:
    raw = json.loads(content.model_dump_json(by_alias=True))
    return to_wire(raw)


# ---------------------------------------------------------------------------
# JSONC writer
# ---------------------------------------------------------------------------

def _write(log: list[tuple[list[str], dict]], path: str) -> None:
    """Write exchange entries as a JSONC file (JSON + // comments)."""
    lines = []
    for comments, payload in log:
        for c in comments:
            lines.append(f"// {c}")
        lines.append(json.dumps(payload, indent=2))
        lines.append("")          # blank line between entries
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {len(log)} wire entries → {path}")


# ---------------------------------------------------------------------------
# Interactive async main
# ---------------------------------------------------------------------------

async def get_user_input(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


async def main(output_path: str = "oauth_events.jsonc") -> None:
    # exchange_log: list of (comment_lines, wire_payload) pairs
    exchange_log: list[tuple[list[str], dict]] = []

    # ------------------------------------------------------------------
    # Session / runner setup
    # ------------------------------------------------------------------
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    session = await session_service.create_session(
        app_name="oauth_inspector",
        user_id="demo_user",
        state={},
    )

    runner = Runner(
        app_name="oauth_inspector",
        agent=agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )

    # ------------------------------------------------------------------
    # Phase 1: initial query – client sends user message
    # ------------------------------------------------------------------
    user_message = types.Content(
        role="user",
        parts=[types.Part(text="Show me my upcoming calendar events.")],
    )

    exchange_log.append((
        [
            "════════════════════════════════════════════════════════════",
            "PHASE 1 — Initial query",
            "════════════════════════════════════════════════════════════",
            "",
            "[1/8] client → agent",
            "The user's opening message. Passed as `new_message` to runner.run_async().",
            "Type: google.genai.types.Content",
        ],
        content_to_wire(user_message),
    ))
    print("Logged [1/8]: client → agent  (initial query)")

    events_iter = runner.run_async(
        session_id=session.id,
        user_id="demo_user",
        new_message=user_message,
    )

    auth_call_id: str | None = None
    auth_cfg: AuthConfig | None = None

    async for event in events_iter:
        wire = event_to_wire(event)

        if event.content and event.content.parts:
            part = event.content.parts[0]
            if part.function_call and part.function_call.name == "adk_request_credential":
                comments = [
                    "[3/8] agent → client",
                    "ADK's OAuth challenge event. The framework synthesises this FunctionCall",
                    "named `adk_request_credential` and yields it to the client.",
                    "The client must detect it via longRunningToolIds (non-empty here),",
                    "extract authConfig.exchangedAuthCredential.oauth2.authUri, redirect the",
                    "user there, then send the callback URL back as a FunctionResponse (see [6/8]).",
                    "Type: google.adk.events.Event",
                ]
                label = "[3/8]"
            elif part.function_call:
                comments = [
                    "[2/8] agent → client",
                    "The LLM decides to call the calendar tool. This is a standard",
                    "function-call event; longRunningToolIds is empty — not yet long-running.",
                    "Type: google.adk.events.Event",
                ]
                label = "[2/8]"
            elif part.function_response:
                comments = [
                    "[4/8] agent → client",
                    "The tool's own FunctionResponse returning {status: pending}.",
                    "ADK also records the full requestedAuthConfigs in event.actions,",
                    "which is how it tracks the pending credential request internally.",
                    "Type: google.adk.events.Event",
                ]
                label = "[4/8]"
            elif part.text:
                comments = [
                    "[5/8] agent → client",
                    "The LLM's natural-language reply telling the user that authentication",
                    "is required. This is the last event of Phase 1; the runner pauses here.",
                    "Type: google.adk.events.Event",
                ]
                label = "[5/8]"
            else:
                comments = ["[?] agent → client  (unknown event type)"]
                label = "[?]"
        else:
            comments = ["[?] agent → client  (no content)"]
            label = "[?]"

        exchange_log.append((comments, wire))
        print(f"Logged {label}: agent → client")

        if fc := get_auth_request_function_call(event):
            auth_call_id = fc.id
            auth_cfg = get_auth_config(fc)

    if not auth_call_id or not auth_cfg:
        print("\n[No adk_request_credential event – writing log and exiting.]")
        _write(exchange_log, output_path)
        return

    # ------------------------------------------------------------------
    # Phase 2: client sends FunctionResponse with OAuth callback URL
    # ------------------------------------------------------------------
    synthetic_callback = (
        f"{REDIRECT_URI}"
        "?code=4%2F0demo-auth-code-replace-with-real"
        "&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar.readonly"
        f"&state={getattr(auth_cfg.exchanged_auth_credential.oauth2, 'state', 'demo-state-1234')}"
    )

    auth_cfg.exchanged_auth_credential.oauth2.auth_response_uri = synthetic_callback
    auth_cfg.exchanged_auth_credential.oauth2.redirect_uri = REDIRECT_URI

    auth_response_content = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=auth_call_id,
                    name="adk_request_credential",
                    response=auth_cfg.model_dump(),
                )
            )
        ],
    )

    exchange_log.append((
        [
            "════════════════════════════════════════════════════════════",
            "PHASE 2 — Client returns the OAuth callback URL",
            "════════════════════════════════════════════════════════════",
            "",
            "[6/8] client → agent",
            "After the user completes the browser OAuth flow, the client sends the",
            "full callback URL (containing `code` and `state`) back to ADK as a",
            "FunctionResponse whose name matches the challenge: `adk_request_credential`.",
            "The `id` field must equal the longRunningToolId from event [3/8].",
            "ADK will exchange the code for an access token, store it in session",
            "state under credentialKey, then automatically retry the original tool call.",
            "Type: google.genai.types.Content",
        ],
        content_to_wire(auth_response_content),
    ))
    print("Logged [6/8]: client → agent  (auth FunctionResponse)")

    # ------------------------------------------------------------------
    # Phase 3: agent resumes – token exchange + tool retry
    # ------------------------------------------------------------------
    events_after = runner.run_async(
        session_id=session.id,
        user_id="demo_user",
        new_message=auth_response_content,
    )

    try:
        n = 7
        async for event in events_after:
            wire = event_to_wire(event)
            if event.content and event.content.parts:
                part = event.content.parts[0]
                if part.function_call:
                    comments = [
                        f"[{n}/8] agent → client",
                        "ADK retried the original tool call after a successful token exchange.",
                        "tool_context.get_auth_response() now returns the exchanged credential.",
                        "Type: google.adk.events.Event",
                    ]
                elif part.function_response:
                    comments = [
                        f"[{n}/8] agent → client",
                        "The tool's FunctionResponse after successful authentication.",
                        "In a real deployment this would contain the calendar data.",
                        "Type: google.adk.events.Event",
                    ]
                elif part.text:
                    comments = [
                        f"[{n}/8] agent → client",
                        "The LLM's final reply to the user, summarising the tool result.",
                        "Type: google.adk.events.Event",
                    ]
                else:
                    comments = [f"[{n}/8] agent → client  (unknown)"]
            else:
                comments = [f"[{n}/8] agent → client  (no content)"]

            exchange_log.append((comments, wire))
            print(f"Logged [{n}/8]: agent → client")
            n += 1
    except Exception as exc:
        exchange_log.append((
            [
                "[7/8] agent → client  — token exchange error",
                "ADK attempted to exchange the synthetic auth code and was rejected by",
                "the OAuth server (invalid_grant). With a real callback URL this phase",
                "completes normally and the tool receives a valid access token.",
                f"Error: {exc}",
            ],
            {"error": str(exc)},
        ))
        print(f"Logged [7/8]: agent → client  (token exchange error: {exc})")

    _write(exchange_log, output_path)


if __name__ == "__main__":
    asyncio.run(main())
