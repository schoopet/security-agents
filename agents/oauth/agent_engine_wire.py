"""
Transforms the local ADK runner events into the two Agent Engine wire formats:

  1. async_stream_query  — each ADK Event is one SSE chunk (raw event dict)
  2. streaming_agent_run_with_events — each chunk is wrapped in
     {"events": [...], "artifacts": [], "session_id": "..."}

Key differences from the local runner (oauth_events.jsonc):

  CLIENT → AGENT
    Local:    raw google.genai.types.Content dict passed to runner.run_async()
    Wire:     HTTP POST body to /{name}:streamQuery?alt=sse
              {"classMethod": "async_stream_query",
               "input": {"message": <content>, "user_id": "...", "session_id": "..."}}

  AGENT → CLIENT
    Local:    event.model_dump_json(by_alias=True)   — null fields included
    Wire:     event.model_dump_json(exclude_none=True) — null fields excluded
              (same as our strip_nulls pass)
              Then SSE-framed:  data: <json>\\n\\n

    streaming_agent_run_with_events adds an extra envelope per chunk:
              {"events": [<event_dict>], "artifacts": [], "session_id": "..."}

  SESSION CONTINUITY
    After Phase 1 the client must use the session_id from the stream to
    resume in Phase 2/3 — Agent Engine does not implicitly maintain turn state.

Run:
    python agent_engine_wire.py
Reads:  oauth_events.jsonc  (produced by oauth_events.py)
Writes: agent_engine_wire.jsonc
"""

import json
import re

INPUT  = "oauth_events.jsonc"
OUTPUT = "agent_engine_wire.jsonc"

# Agent Engine resource path (placeholder — replace with your deployed engine)
ENGINE_NAME = (
    "projects/mmontan-ml-dev/locations/us-central1"
    "/reasoningEngines/<your-engine-id>"
)
STREAM_QUERY_URL = f"https://us-central1-aiplatform.googleapis.com/v1/{ENGINE_NAME}:streamQuery?alt=sse"


# ---------------------------------------------------------------------------
# Parse the JSONC file (strip // comments, split on blank lines between objects)
# ---------------------------------------------------------------------------

def load_jsonc(path: str) -> list[dict]:
    """Return the list of JSON payloads from the JSONC file.

    Strips // comments only outside of JSON string literals, so that URLs
    like https://... inside string values are preserved intact.
    """
    with open(path) as f:
        raw = f.read()

    # Walk the file character by character, removing // comments only when
    # we are not inside a JSON string.
    out = []
    i = 0
    in_string = False
    while i < len(raw):
        ch = raw[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < len(raw):
                # Escaped character — consume both so we don't misread \" as end
                i += 1
                out.append(raw[i])
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
                out.append(ch)
            elif ch == "/" and i + 1 < len(raw) and raw[i + 1] == "/":
                # Line comment — skip to end of line
                while i < len(raw) and raw[i] != "\n":
                    i += 1
                continue
            else:
                out.append(ch)
        i += 1

    no_comments = "".join(out)

    # Extract top-level JSON objects using the stdlib decoder
    decoder = json.JSONDecoder()
    payloads = []
    pos = 0
    while pos < len(no_comments):
        idx = no_comments.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(no_comments, idx)
            payloads.append(obj)
            pos = end   # end is absolute index, not relative
        except json.JSONDecodeError:
            pos = idx + 1
    return payloads


# ---------------------------------------------------------------------------
# Comment / payload builders for each phase
# ---------------------------------------------------------------------------

def http_request_block(comments: list[str], body: dict) -> tuple[list[str], dict]:
    return (comments, body)


def sse_event_block(
    comments: list[str],
    event_dict: dict,
    session_id: str,
    for_streaming_run: bool = False,
) -> tuple[list[str], dict | str]:
    if for_streaming_run:
        payload: dict = {
            "events": [event_dict],
            "artifacts": [],
            "session_id": session_id,
        }
    else:
        payload = event_dict
    # Annotate the SSE framing in the comments
    comments = comments + [
        "",
        "SSE wire frame:",
        f"  data: {json.dumps(payload, separators=(',', ':'))[:120]}...",
        "  (followed by \\n\\n)",
    ]
    return (comments, payload)


# ---------------------------------------------------------------------------
# Write JSONC
# ---------------------------------------------------------------------------

def write_jsonc(log: list[tuple[list[str], dict]], path: str) -> None:
    lines = []
    for comments, payload in log:
        for c in comments:
            lines.append(f"// {c}" if c else "//")
        lines.append(json.dumps(payload, indent=2))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(log)} entries → {path}")


# ---------------------------------------------------------------------------
# Main transformation
# ---------------------------------------------------------------------------

def main() -> None:
    payloads = load_jsonc(INPUT)
    # payloads[0..7] correspond to entries [1/8]..[8/8] in order:
    # 0: initial user Content          (client→agent)
    # 1: llm tool call Event           (agent→client)
    # 2: adk_request_credential Event  (agent→client)
    # 3: tool FunctionResponse Event   (agent→client)
    # 4: llm text reply Event          (agent→client)
    # 5: auth FunctionResponse Content (client→agent)
    # 6: tool retry response Event     (agent→client)
    # 7: llm final reply Event         (agent→client)

    if len(payloads) < 8:
        print(f"Expected 8 payloads in {INPUT}, got {len(payloads)}. Run oauth_events.py first.")
        return

    p = payloads
    # Extract the session_id from any event that has one (events 1-4 share invocationId)
    session_id = (
        p[1].get("id") or "session-id-from-stream"
    )
    # Use invocation_id from one of the phase-1 events as a stable session key
    invocation_id = p[1].get("invocationId", "inv-xxx")

    log: list[tuple[list[str], dict]] = []

    # ------------------------------------------------------------------
    # PHASE 1  — client sends initial query via streamQuery
    # ------------------------------------------------------------------
    log.append((
        [
            "════════════════════════════════════════════════════════════",
            "PHASE 1 — Initial query",
            "════════════════════════════════════════════════════════════",
            "",
            "[1/8] client → Agent Engine   HTTP POST",
            f"POST {STREAM_QUERY_URL}",
            "Content-Type: application/json",
            "Authorization: Bearer <gcloud-token>",
            "",
            "The message field is a google.genai.types.Content dict.",
            "session_id is omitted on the first turn — Agent Engine creates one.",
            "The response is a server-sent event (SSE) stream; each line is:",
            "  data: <json>\\n\\n",
        ],
        {
            "classMethod": "async_stream_query",
            "input": {
                "message": p[0],        # raw Content dict
                "user_id": "demo_user",
                "session_id": None,     # omit or null → new session
            },
        },
    ))

    # ------------------------------------------------------------------
    # Phase 1 SSE stream  (4 events)
    # ------------------------------------------------------------------
    log.append(sse_event_block(
        [
            "[2/8] agent → client   SSE chunk  (async_stream_query format)",
            "The LLM's decision to call list_calendar_events.",
            "longRunningToolIds is [] — not yet long-running.",
            "",
            "streaming_agent_run_with_events format wraps this in:",
            '  {"events": [<this>], "artifacts": [], "session_id": "..."}',
            "See the streaming_agent_run_with_events block at the end of the file.",
        ],
        p[1], session_id,
    ))

    log.append(sse_event_block(
        [
            "[3/8] agent → client   SSE chunk  (async_stream_query format)",
            "ADK's OAuth challenge. The client detects this via longRunningToolIds",
            "being non-empty. It must:",
            "  1. Extract authConfig.exchangedAuthCredential.oauth2.authUri",
            "  2. Append &redirect_uri=<your-redirect-uri> and send user to that URL",
            "  3. After the user authorises, capture the full callback URL",
            "  4. Send it back in Phase 2 (see [6/8])",
            "The session_id received alongside this chunk must be preserved",
            "so Phase 2 can resume the same session.",
        ],
        p[2], session_id,
    ))

    log.append(sse_event_block(
        [
            "[4/8] agent → client   SSE chunk  (async_stream_query format)",
            "The tool's own FunctionResponse — status: pending.",
            "actions.requestedAuthConfigs holds the full pending auth config",
            "for Agent Engine's internal bookkeeping.",
        ],
        p[3], session_id,
    ))

    log.append(sse_event_block(
        [
            "[5/8] agent → client   SSE chunk  (async_stream_query format)",
            "LLM natural-language reply: 'I need to authenticate...'",
            "This is the last SSE chunk of Phase 1. The stream closes here.",
            "The client now has the auth URL and waits for the user to complete",
            "the browser OAuth flow before opening a new streamQuery call.",
        ],
        p[4], session_id,
    ))

    # ------------------------------------------------------------------
    # PHASE 2  — client sends auth FunctionResponse
    # ------------------------------------------------------------------
    log.append((
        [
            "════════════════════════════════════════════════════════════",
            "PHASE 2 — Client returns the OAuth callback URL",
            "════════════════════════════════════════════════════════════",
            "",
            "[6/8] client → Agent Engine   HTTP POST  (new streamQuery call)",
            f"POST {STREAM_QUERY_URL}",
            "Content-Type: application/json",
            "Authorization: Bearer <gcloud-token>",
            "",
            "session_id MUST match the one from Phase 1 so Agent Engine can",
            "resume the same session. The message is a Content dict whose",
            "single part is a FunctionResponse named `adk_request_credential`.",
            "Its `id` must equal the longRunningToolId from event [3/8].",
            "Agent Engine will exchange the auth code for a token, store it",
            "in session state, then automatically retry list_calendar_events.",
        ],
        {
            "classMethod": "async_stream_query",
            "input": {
                "message": p[5],        # Content with FunctionResponse part
                "user_id": "demo_user",
                "session_id": session_id,   # resume Phase 1 session
            },
        },
    ))

    # ------------------------------------------------------------------
    # PHASE 3  — agent resumes after token exchange
    # ------------------------------------------------------------------
    log.append(sse_event_block(
        [
            "[7/8] agent → client   SSE chunk  (async_stream_query format)",
            "Tool FunctionResponse after the token exchange. In a real deployment",
            "this contains the calendar data; here it shows {status: authenticated}.",
        ],
        p[6], session_id,
    ))

    log.append(sse_event_block(
        [
            "[8/8] agent → client   SSE chunk  (async_stream_query format)",
            "LLM final reply summarising the tool result.",
            "This is the last SSE chunk; the stream closes.",
        ],
        p[7], session_id,
    ))

    # ------------------------------------------------------------------
    # Appendix: streaming_agent_run_with_events wrapper (one example)
    # ------------------------------------------------------------------
    log.append((
        [
            "════════════════════════════════════════════════════════════",
            "APPENDIX — streaming_agent_run_with_events wire format",
            "════════════════════════════════════════════════════════════",
            "",
            "streaming_agent_run_with_events is the lower-level method used",
            "by AgentSpace. Its SSE chunks wrap each event in an envelope:",
            "  {\"events\": [<event>], \"artifacts\": [], \"session_id\": \"...\"}",
            "",
            "The request body is different too — it takes a JSON-encoded string:",
            "  {\"classMethod\": \"streaming_agent_run_with_events\",",
            "   \"input\": {\"request_json\": \"{...}\"}}",
            "",
            "Example: event [3/8] (adk_request_credential) in this format:",
        ],
        {
            "events": [p[2]],
            "artifacts": [],
            "session_id": session_id,
        },
    ))

    write_jsonc(log, OUTPUT)


if __name__ == "__main__":
    main()
