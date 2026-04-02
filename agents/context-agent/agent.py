"""
Context Agent for Vertex AI Agent Engine
Exposes session, memory, and sandbox management via a command interface.
Useful for debugging whether an agent can read/write its own sessions and memories.

Uses the high-level vertexai SDK (vertexai.Client / client.agent_engines.*).
Authentication is handled automatically by the SDK via ADC / workload identity.

Command reference
-----------------
Sessions (all take <engine> as first arg — use "self" to target own engine):
  session-create  <engine> <user_id>
  session-get     <engine> <session_id>
  session-list    <engine> [<user_id>]
  session-delete  <engine> <session_id>
  session-events  <engine> <session_id>
  session-append  <engine> <session_id> <author> <text>   — append a text event

Memories:
  memory-create   <engine> <user_id> <fact_text...>
  memory-get      <engine> <memory_id>
  memory-list     <engine> [<user_id>]
  memory-delete   <engine> <memory_id>

Sandboxes (real Vertex AI Sandbox Environments API):
  sandbox-create  <engine>
  sandbox-get     <engine> <sandbox_id>
  sandbox-list    <engine>
  sandbox-delete  <engine> <sandbox_id>
  sandbox-exec    <engine> <sandbox_id> <python_code_b64>

Model calling (endpoint = model ID, full Vertex path, or https:// URL):
  call-model      <endpoint> <prompt...>

Misc:
  self-inspect    — show env vars & resolved own resource name
  token-info      — show the SDK's access token and tokeninfo (bound/unbound, scopes, SA)
  log             <message...>   — write a line to Cloud Logging (stdout)
  logapi          <message...>   — write a log entry via Cloud Logging API
  help
"""

import vertexai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ID = "mmontan-ml-dev"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://mmontan-ml-dev-staging-bucket"

# Env var candidates searched when engine_ref == "self"
_OWN_RESOURCE_NAME_CANDIDATES = [
    "AGENT_ENGINE_RESOURCE_NAME",
    "REASONING_ENGINE_RESOURCE_NAME",
    "GOOGLE_CLOUD_AGENT_ENGINE_ID",
    "GOOGLE_CLOUD_REASONING_ENGINE_ID",
    "K_SERVICE",
]


# ---------------------------------------------------------------------------
# Resource name helpers
# ---------------------------------------------------------------------------

def _normalize_engine(engine_ref: str) -> str | None:
    """Return a full reasoningEngines resource path."""
    import re
    if not engine_ref:
        return None
    if "reasoningEngines/" in engine_ref:
        return engine_ref
    if re.match(r"^\d+$", engine_ref):
        return f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{engine_ref}"
    return engine_ref


def _discover_own_resource() -> tuple[str | None, dict]:
    """Resolve this agent's own resource name from env vars + metadata server."""
    import os, urllib.request

    inspection: dict = {}
    found = None

    for var in _OWN_RESOURCE_NAME_CANDIDATES:
        val = os.environ.get(var)
        inspection[var] = val
        if val and found is None:
            found = val

    for extra in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "K_REVISION", "K_CONFIGURATION",
                  "GOOGLE_APPLICATION_CREDENTIALS", "CLOUD_RUN_JOB", "CLOUD_RUN_EXECUTION"):
        inspection[extra] = os.environ.get(extra)

    meta: dict = {}
    for key, url in {
        "project_id":      "http://metadata.google.internal/computeMetadata/v1/project/project-id",
        "instance_name":   "http://metadata.google.internal/computeMetadata/v1/instance/name",
        "instance_id":     "http://metadata.google.internal/computeMetadata/v1/instance/id",
        "service_account": "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
        "zone":            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
    }.items():
        try:
            req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                meta[key] = resp.read().decode().strip()
        except Exception as exc:
            meta[key] = f"<unavailable: {exc}>"
    inspection["metadata_server"] = meta

    if found:
        found = _normalize_engine(found)

    if not found:
        inspection["_note"] = (
            f"Could not resolve own resource name. "
            f"Set one of {_OWN_RESOURCE_NAME_CANDIDATES} in env_vars at deploy time."
        )

    return found, inspection


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _to_dict(obj) -> dict:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def _ok(**fields) -> str:
    import json
    return json.dumps({"ok": True, **fields}, indent=2, default=str)


def _err(msg: str, **fields) -> str:
    import json
    return json.dumps({"error": msg, **fields}, indent=2, default=str)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ContextAgent:

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url
        self._last_request: str | None = None  # set by httpx hook after each SDK call

    def set_up(self):
        from google.genai.types import HttpOptions

        kwargs: dict = {"project": PROJECT_ID, "location": LOCATION}
        if self._base_url:
            kwargs["http_options"] = HttpOptions(baseUrl=self._base_url)
        self._client = vertexai.Client(**kwargs)
        self._ae = self._client.agent_engines

        # Attach an httpx event hook to the actual client instance the SDK uses.
        _agent = self
        try:
            httpx_client = self._client._api_client._httpx_client
            def _log_request(request):
                _agent._last_request = f">> CAPTURED: {request.method} {request.url}"
            httpx_client.event_hooks["request"].append(_log_request)
            print("[ContextAgent] httpx event hook attached.")
        except Exception as exc:
            print(f"[ContextAgent] Could not attach httpx event hook: {exc}")

        # Also patch requests.adapters.HTTPAdapter.send — used by google-auth
        # for credential refresh. If auth fails (e.g. 403), the httpx hook never
        # fires, but this one will, so we still capture the URL.
        import requests.adapters
        _orig_adapter_send = requests.adapters.HTTPAdapter.send

        def _adapter_send(adapter_self, request, **kwargs):
            _agent._last_request = f">> CAPTURED (requests): {request.method} {request.url}"
            return _orig_adapter_send(adapter_self, request, **kwargs)

        requests.adapters.HTTPAdapter.send = _adapter_send

        # Log the effective API endpoint
        try:
            api_client = self._client._api_client
            effective_url = getattr(api_client, "custom_base_url", None) or getattr(api_client, "_base_url", None)
            print(f"[ContextAgent] base_url param: {self._base_url!r}")
            print(f"[ContextAgent] api_client type: {type(api_client).__name__}")
            print(f"[ContextAgent] api_client.custom_base_url: {getattr(api_client, 'custom_base_url', '<not found>')!r}")
            print(f"[ContextAgent] api_client.project: {getattr(api_client, 'project', '<not found>')!r}")
            print(f"[ContextAgent] api_client.location: {getattr(api_client, 'location', '<not found>')!r}")
        except Exception as exc:
            print(f"[ContextAgent] Could not inspect api_client: {exc}")
        print("[ContextAgent] Agent is ready.")

    # ------------------------------------------------------------------
    # Engine resolution
    # ------------------------------------------------------------------

    def _resolve_engine(self, engine_ref: str) -> tuple[str | None, str | None]:
        if engine_ref in ("self", ""):
            own, _ = _discover_own_resource()
            if not own:
                return None, (
                    f"could not resolve own resource name — "
                    f"set one of {_OWN_RESOURCE_NAME_CANDIDATES} in env_vars, "
                    f"or run 'self-inspect'"
                )
            return own, None
        normalized = _normalize_engine(engine_ref)
        if not normalized:
            return None, f"could not parse engine reference: {engine_ref!r}"
        return normalized, None

    def _session_name(self, engine: str, session_id: str) -> str:
        return f"{engine}/sessions/{session_id}"

    def _memory_name(self, engine: str, memory_id: str) -> str:
        return f"{engine}/memories/{memory_id}"

    def _sandbox_name(self, engine: str, sandbox_id: str) -> str:
        return f"{engine}/sandboxEnvironments/{sandbox_id}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def query(self, input: str, **kwargs) -> str:
        print(f"[ContextAgent.query] input={input!r}")
        parts = input.strip().split(None, 1)
        if not parts:
            return _err("empty command — try 'help'")
        cmd, args = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        try:
            result = self._dispatch(cmd, args)
        except Exception as exc:
            result = _err(str(exc), exception_type=type(exc).__name__, endpoint=self._last_request)

        print(f"[ContextAgent.query] response={result!r}")
        return result

    def _dispatch(self, cmd: str, args: str) -> str:
        routes = {
            "session-create":  self._session_create,
            "session-get":     self._session_get,
            "session-list":    self._session_list,
            "session-delete":  self._session_delete,
            "session-events":  self._session_events,
            "session-append":  self._session_append,
            "memory-create":   self._memory_create,
            "memory-get":      self._memory_get,
            "memory-list":     self._memory_list,
            "memory-delete":   self._memory_delete,
            "sandbox-create":  self._sandbox_create,
            "sandbox-get":     self._sandbox_get,
            "sandbox-list":    self._sandbox_list,
            "sandbox-delete":  self._sandbox_delete,
            "sandbox-exec":    self._sandbox_exec,
            "call-model":      self._call_model,
            "self-inspect":    self._self_inspect,
            "token-info":      self._token_info,
            "log":             self._log,
            "logapi":          self._logapi,
            "help":            lambda _: __doc__,
            "?":               lambda _: __doc__,
        }
        handler = routes.get(cmd)
        if handler is None:
            return _err(f"unknown command: {cmd!r}", hint="use 'help'")
        return handler(args)

    # ------------------------------------------------------------------
    # Self-inspection
    # ------------------------------------------------------------------

    def _token_info(self, _args: str) -> str:
        """token-info — fetch and inspect the access token the SDK is using."""
        import google.auth
        import google.auth.transport.requests
        import requests as _requests

        # Get the same credentials the SDK uses (ADC).
        try:
            creds, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(google.auth.transport.requests.Request())
            token = creds.token
        except Exception as exc:
            return _err(f"could not get credentials: {exc}", endpoint=self._last_request)

        # Decode JWT claims without verification (works even if tokeninfo rejects it).
        import base64, json as _json
        jwt_claims = None
        try:
            payload_b64 = token.split(".")[1]
            padding = 4 - len(payload_b64) % 4
            jwt_claims = _json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * padding))
        except Exception:
            pass  # not a JWT — that's fine

        # Build an mTLS session if a client cert source is available.
        import tempfile, os
        from google.auth.transport import mtls as _mtls

        mtls_used = False
        session = _requests.Session()
        try:
            if _mtls.has_default_client_cert_source():
                cert_pem, key_pem = _mtls.default_client_cert_source()()
                # requests needs cert+key as files or a combined PEM file.
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
                tmp.write(cert_pem + key_pem)
                tmp.flush()
                tmp.close()
                session.cert = tmp.name
                mtls_used = True
        except Exception as exc:
            mtls_used = f"mtls setup failed: {exc}"

        # Use the mTLS endpoint when mTLS is active.
        base = "https://oauth2.mtls.googleapis.com" if mtls_used is True else "https://oauth2.googleapis.com"

        # Call tokeninfo for both access_token and id_token forms.
        tokeninfo_access, tokeninfo_id = None, None
        try:
            r = session.get(f"{base}/tokeninfo", params={"access_token": token}, timeout=10)
            tokeninfo_access = {"status": r.status_code, "body": r.json()}
        except Exception as exc:
            tokeninfo_access = {"error": str(exc)}

        try:
            r = session.get(f"{base}/tokeninfo", params={"id_token": token}, timeout=10)
            tokeninfo_id = {"status": r.status_code, "body": r.json()}
        except Exception as exc:
            tokeninfo_id = {"error": str(exc)}

        # Clean up temp cert file.
        if mtls_used is True:
            try:
                os.unlink(session.cert)
            except Exception:
                pass

        # Determine bound/unbound from JWT claims or tokeninfo.
        aud = (jwt_claims or {}).get("aud") or (tokeninfo_access or {}).get("body", {}).get("aud", "")
        bound = bool(aud and not str(aud).startswith("https://"))

        return _ok(
            endpoint=self._last_request,
            mtls=mtls_used,
            token=token,
            token_type=creds.__class__.__name__,
            jwt_claims=jwt_claims,
            bound=bound,
            audience=aud or None,
            tokeninfo_as_access_token=tokeninfo_access,
            tokeninfo_as_id_token=tokeninfo_id,
        )

    def _log(self, args: str) -> str:
        """log <message...> — write a line to Cloud Logging (stdout) and return it."""
        msg = args.strip() or "(empty log message)"
        print(f"[LOG] {msg}")
        return _ok(endpoint=self._last_request, logged=msg)

    def _logapi(self, args: str) -> str:
        """logapi <message...> — write a log entry via the Cloud Logging API."""
        import google.cloud.logging as cloud_logging
        msg = args.strip() or "(empty log message)"
        try:
            lc = cloud_logging.Client(project=PROJECT_ID)
            logger = lc.logger("context-agent")
            logger.log_text(msg, severity="INFO")
            return _ok(endpoint=self._last_request, logged=msg, via="cloud-logging-api")
        except Exception as exc:
            return _err(str(exc), endpoint=self._last_request)

    def _self_inspect(self, _args: str) -> str:
        own, inspection = _discover_own_resource()
        return _ok(resolved_resource_name=own, inspection=inspection)

    # ------------------------------------------------------------------
    # Session commands
    # ------------------------------------------------------------------

    def _session_create(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: session-create <engine> <user_id>")
        engine_ref, user_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        op = self._ae.create_session(
            name=engine,
            user_id=user_id,
            config={"wait_for_completion": True},
        )
        return _ok(endpoint=self._last_request, operation=_to_dict(op))

    def _session_get(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: session-get <engine> <session_id>")
        engine_ref, session_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._session_name(engine, session_id)
        session = self._ae.get_session(name=resource)
        return _ok(endpoint=self._last_request, session=_to_dict(session))

    def _session_list(self, args: str) -> str:
        parts = args.split(None, 1)
        if not parts:
            return _err("usage: session-list <engine> [<user_id>]")
        engine_ref = parts[0]
        user_id = parts[1].strip() if len(parts) > 1 else None
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        config = {}
        if user_id:
            config["filter"] = f'user_id="{user_id}"'
        sessions = list(self._ae.list_sessions(name=engine, config=config or None))
        return _ok(endpoint=self._last_request, count=len(sessions), sessions=[_to_dict(s) for s in sessions])

    def _session_delete(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: session-delete <engine> <session_id>")
        engine_ref, session_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._session_name(engine, session_id)
        op = self._ae.delete_session(name=resource)
        return _ok(endpoint=self._last_request, deleted=session_id, operation=_to_dict(op))

    def _session_append(self, args: str) -> str:
        """session-append <engine> <session_id> <author> <text...>"""
        import uuid
        from datetime import datetime, timezone
        from google.genai.types import Content, Part

        parts = args.split(None, 3)
        if len(parts) < 4:
            return _err("usage: session-append <engine> <session_id> <author> <text>")
        engine_ref, session_id, author, text = parts[0], parts[1], parts[2], parts[3]
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)

        resource = self._session_name(engine, session_id)
        resp = self._ae.append_session_event(
            name=resource,
            author=author,
            invocation_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            config={"content": Content(role=author, parts=[Part(text=text)])},
        )
        return _ok(endpoint=self._last_request, response=_to_dict(resp))

    def _session_events(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: session-events <engine> <session_id>")
        engine_ref, session_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._session_name(engine, session_id)
        events = list(self._ae.list_session_events(name=resource))
        return _ok(endpoint=self._last_request, session_id=session_id, count=len(events), events=[_to_dict(e) for e in events])

    # ------------------------------------------------------------------
    # Memory commands
    # ------------------------------------------------------------------

    def _memory_create(self, args: str) -> str:
        parts = args.split(None, 2)
        if len(parts) < 3:
            return _err("usage: memory-create <engine> <user_id> <fact_text>")
        engine_ref, user_id, fact = parts[0], parts[1], parts[2]
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        op = self._ae.create_memory(
            name=engine,
            fact=fact,
            scope={"user_id": user_id},
        )
        return _ok(endpoint=self._last_request, operation=_to_dict(op))

    def _memory_get(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: memory-get <engine> <memory_id>")
        engine_ref, memory_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._memory_name(engine, memory_id)
        memory = self._ae.get_memory(name=resource)
        return _ok(endpoint=self._last_request, memory=_to_dict(memory))

    def _memory_list(self, args: str) -> str:
        parts = args.split(None, 1)
        if not parts:
            return _err("usage: memory-list <engine> [<user_id>]")
        engine_ref = parts[0]
        user_id = parts[1].strip() if len(parts) > 1 else None
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        config = {}
        if user_id:
            config["filter"] = f'scope.user_id="{user_id}"'
        memories = list(self._ae.list_memories(name=engine, config=config or None))
        return _ok(endpoint=self._last_request, count=len(memories), memories=[_to_dict(m) for m in memories])

    def _memory_delete(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: memory-delete <engine> <memory_id>")
        engine_ref, memory_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._memory_name(engine, memory_id)
        op = self._ae.delete_memory(name=resource)
        return _ok(endpoint=self._last_request, deleted=memory_id, operation=_to_dict(op))

    # ------------------------------------------------------------------
    # Sandbox commands (Vertex AI Sandbox Environments API)
    # ------------------------------------------------------------------

    def _sandbox_create(self, args: str) -> str:
        engine_ref = args.strip()
        if not engine_ref:
            return _err("usage: sandbox-create <engine>")
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        op = self._ae.sandboxes.create(name=engine)
        return _ok(endpoint=self._last_request, operation=_to_dict(op))

    def _sandbox_get(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: sandbox-get <engine> <sandbox_id>")
        engine_ref, sandbox_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._sandbox_name(engine, sandbox_id)
        sb = self._ae.sandboxes.get(name=resource)
        return _ok(endpoint=self._last_request, sandbox=_to_dict(sb))

    def _sandbox_list(self, args: str) -> str:
        engine_ref = args.strip()
        if not engine_ref:
            return _err("usage: sandbox-list <engine>")
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        sandboxes = list(self._ae.sandboxes.list(name=engine))
        return _ok(endpoint=self._last_request, count=len(sandboxes), sandboxes=[_to_dict(s) for s in sandboxes])

    def _sandbox_delete(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: sandbox-delete <engine> <sandbox_id>")
        engine_ref, sandbox_id = parts[0], parts[1].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        resource = self._sandbox_name(engine, sandbox_id)
        op = self._ae.sandboxes.delete(name=resource)
        return _ok(endpoint=self._last_request, deleted=sandbox_id, operation=_to_dict(op))

    def _sandbox_exec(self, args: str) -> str:
        """sandbox-exec <engine> <sandbox_id> <python_code_b64>"""
        import base64
        parts = args.split(None, 2)
        if len(parts) < 3:
            return _err("usage: sandbox-exec <engine> <sandbox_id> <python_code_b64>")
        engine_ref, sandbox_id, b64 = parts[0], parts[1], parts[2].strip()
        engine, err = self._resolve_engine(engine_ref)
        if err:
            return _err(err)
        try:
            code = base64.b64decode(b64).decode("utf-8")
        except Exception as exc:
            return _err(f"base64 decode failed: {exc}")
        resource = self._sandbox_name(engine, sandbox_id)
        result = self._ae.sandboxes.execute_code(
            name=resource,
            input_data={"code": code},
        )
        return _ok(endpoint=self._last_request, result=_to_dict(result))

    # ------------------------------------------------------------------
    # Model calling
    # ------------------------------------------------------------------

    def _call_model(self, args: str) -> str:
        parts = args.split(None, 1)
        if len(parts) < 2:
            return _err("usage: call-model <endpoint> <prompt>")
        endpoint, prompt = parts[0], parts[1]
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return self._call_http(endpoint, prompt)
        return self._call_vertex(endpoint, prompt)

    def _call_vertex(self, model_id: str, prompt: str) -> str:
        try:
            from vertexai.generative_models import GenerativeModel
            model = GenerativeModel(model_name=model_id)
            response = model.generate_content(prompt)
            return _ok(endpoint=self._last_request, response=response.text)
        except Exception as exc:
            return _err(str(exc), endpoint=self._last_request)

    def _call_http(self, url: str, prompt: str) -> str:
        import json as _json
        import requests as _requests
        import google.auth, google.auth.transport.requests

        try:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(google.auth.transport.requests.Request())
            token = creds.token
        except Exception:
            token = None

        payload = _json.dumps({"instances": [{"content": prompt}]})
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = _requests.post(url, data=payload, headers=headers, timeout=60)
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return _ok(endpoint=self._last_request, response=body)
        except Exception as exc:
            return _err(str(exc), endpoint=self._last_request)


# ---------------------------------------------------------------------------
# Deploy / query helpers
# ---------------------------------------------------------------------------

def _make_client(
    project: str = PROJECT_ID,
    location: str = LOCATION,
    base_url: str | None = None,
) -> vertexai.Client:
    """Create a vertexai.Client, optionally pointing at a custom base URL."""
    from google.genai.types import HttpOptions
    kwargs: dict = {"project": project, "location": location}
    if base_url:
        kwargs["http_options"] = HttpOptions(baseUrl=base_url)
    client = vertexai.Client(**kwargs)
    try:
        api_client = client._api_client
        print(f"[_make_client] project={project!r} location={location!r} base_url={base_url!r}")
        print(f"[_make_client] api_client.custom_base_url: {getattr(api_client, 'custom_base_url', '<not found>')!r}")
    except Exception as exc:
        print(f"[_make_client] Could not inspect api_client: {exc}")
    return client


def deploy(
    project: str = PROJECT_ID,
    location: str = LOCATION,
    base_url: str | None = None,
):
    client = _make_client(project=project, location=location, base_url=base_url)
    remote_agent = client.agent_engines.create(
        agent=ContextAgent(base_url=base_url),
        config={
            "display_name": "context-agent",
            "identity_type": "AGENT_IDENTITY",
            "requirements": [
                "google-cloud-aiplatform[agent_engines]",
                "cloudpickle",
                "pydantic",
                "google-auth[cryptography]",
                "pyOpenSSL",
                "google-cloud-storage",
                "google-cloud-logging",
            ],
            "staging_bucket": STAGING_BUCKET,
            "env_vars": {
                "GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES": "false",
                "GOOGLE_API_USE_CLIENT_CERTIFICATE": "false",
            },
        },
    )
    dump = remote_agent.model_dump()
    print(f"Deployed agent: {dump.get('name', dump)}")
    print(f"Full details:   {dump}")
    return remote_agent


def query_remote(
    resource_name: str,
    user_input: str,
    project: str = PROJECT_ID,
    location: str = LOCATION,
    base_url: str | None = None,
):
    client = _make_client(project=project, location=location, base_url=base_url)
    remote_agent = client.agent_engines.get(name=resource_name)
    response = remote_agent.query(input=user_input)
    print(f"Remote response: {response}")
    return response


if __name__ == "__main__":
    import sys as _sys

    def _pop_flag(args: list, flag: str) -> str | None:
        if flag in args:
            idx = args.index(flag)
            val = args[idx + 1]
            del args[idx:idx + 2]
            return val
        return None

    # Usage: agent.py [--project <id>] [--location <loc>] [--base-url <url>]
    _args = _sys.argv[1:]
    _project  = _pop_flag(_args, "--project")  or PROJECT_ID
    _location = _pop_flag(_args, "--location") or LOCATION
    _base_url = _pop_flag(_args, "--base-url")
    deploy(project=_project, location=_location, base_url=_base_url)
