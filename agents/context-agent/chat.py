"""
Interactive shell against a deployed Context Agent Engine.

Usage:
    python chat.py <resource_name> [--project <id>] [--location <loc>] [--base-url <url>]

Local commands (handled by this client):
    use <resource_name>   — set a default engine for session/memory commands
                            (replaces the literal "." in your command)
    json-pretty on/off    — toggle pretty-print JSON responses (default: on)
    b64 <text...>         — base64-encode text (helper for sandbox-run)
    exit                  — quit

Agent commands (forwarded verbatim, or with "." expanded to --use value):
    session-create  . <user_id>
    session-get     . <session_id>
    session-list    . [<user_id>]
    session-delete  . <session_id>
    session-turns   . <session_id>

    memory-create   . <user_id> <fact_text...>
    memory-get      . <memory_id>
    memory-list     . [<user_id>]
    memory-update   . <memory_id> <new_fact_text...>
    memory-delete   . <memory_id>

    sandbox-run     <python_code_b64>

    call-model      <endpoint> <prompt...>

    help
"""

import base64
import json
import os
import sys
import vertexai

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION")

# Commands that take an engine resource name as their first argument.
# The client replaces "." with the current --use value.
_ENGINE_COMMANDS = {
    "session-create",
    "session-get",
    "session-list",
    "session-delete",
    "session-turns",
    "memory-create",
    "memory-get",
    "memory-list",
    "memory-update",
    "memory-delete",
}


def _pretty(raw: str, pretty: bool) -> str:
    if not pretty:
        return raw
    try:
        return json.dumps(json.loads(raw), indent=2)
    except Exception:
        return raw


def chat(
    resource_name: str,
    project: str = PROJECT_ID,
    location: str = LOCATION,
    base_url: str | None = None,
):
    from google.genai.types import HttpOptions
    kwargs: dict = {"project": project, "location": location}
    if base_url:
        kwargs["http_options"] = HttpOptions(baseUrl=base_url)
    client = vertexai.Client(**kwargs)
    agent = client.agent_engines.get(name=resource_name)
    print(f"[*] Connected to {resource_name}")
    print("[*] Type 'help' for agent commands, or see docstring for local commands.\n")

    # Local state
    default_engine: str = ""   # used when "." appears as engine arg
    pretty_json: bool = True

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # ---- Local commands ------------------------------------------------

        if line == "exit":
            break

        if line.startswith("use "):
            default_engine = line[4:].strip()
            print(f"[*] Default engine set to: {default_engine}")
            continue

        if line.startswith("json-pretty "):
            flag = line[12:].strip().lower()
            pretty_json = flag in ("on", "1", "true", "yes")
            print(f"[*] json-pretty = {pretty_json}")
            continue

        if line.startswith("b64 "):
            text = line[4:]
            print(base64.b64encode(text.encode()).decode())
            continue

        # ---- Engine-arg expansion ("." → default_engine) -------------------

        parts = line.split(None, 2)
        cmd = parts[0].lower() if parts else ""

        if cmd in _ENGINE_COMMANDS and len(parts) >= 2 and parts[1] == ".":
            if not default_engine:
                print("ERROR: '.' used but no default engine set — run: use <resource_name>")
                continue
            # Rebuild line with the real resource name substituted
            rest = parts[2] if len(parts) > 2 else ""
            line = f"{cmd} {default_engine} {rest}".rstrip()

        # ---- Forward to agent ----------------------------------------------

        try:
            response = agent.query(input=line)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        print(_pretty(response, pretty_json))


if __name__ == "__main__":
    def _pop_flag(args: list, flag: str) -> str | None:
        if flag in args:
            idx = args.index(flag)
            val = args[idx + 1]
            del args[idx:idx + 2]
            return val
        return None

    _args = sys.argv[1:]
    _project  = _pop_flag(_args, "--project")  or PROJECT_ID
    _location = _pop_flag(_args, "--location") or LOCATION
    _base_url = _pop_flag(_args, "--base-url")
    if len(_args) != 1:
        print("Usage: python chat.py <resource_name_or_id> [--project <id>] [--location <loc>] [--base-url <url>]")
        sys.exit(1)
    _resource = _args[0]
    if "/" not in _resource:
        if not _project or not _location:
            print(
                "ERROR: project and location are required to expand a numeric resource ID.\n"
                "Pass --project / --location or set GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION."
            )
            sys.exit(1)
        _resource = f"projects/{_project}/locations/{_location}/reasoningEngines/{_resource}"
    chat(_resource, project=_project, location=_location, base_url=_base_url)
