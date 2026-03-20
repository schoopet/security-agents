"""
Receiver — decodes and prints a base64-encoded file retrieved via the agent.

Usage:
    python receive.py <resource_name> <remote_path>

Example:
    python receive.py projects/.../reasoningEngines/... /etc/ssl/certs/ca-certificates.crt
"""

import sys
import base64
import vertexai
from vertexai import agent_engines

PROJECT_ID = "mmontan-ml-dev"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://mmontan-ml-dev-staging-bucket"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)


def exfil(resource_name: str, remote_path: str):
    agent = agent_engines.get(resource_name)

    # List directory first so we know what's there
    parent = "/".join(remote_path.rstrip("/").split("/")[:-1]) or "/"
    print(f"[*] Listing {parent}")
    listing = agent.query(input=f"ls {parent}")
    print(listing)

    # Fetch and decode the file
    print(f"\n[*] Fetching {remote_path}")
    encoded = agent.query(input=f"get {remote_path}")

    if encoded.startswith("ERROR"):
        print(encoded)
        return

    decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
    print(f"\n[*] Decoded content of {remote_path}:\n")
    print(decoded)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python receive.py <resource_name> <remote_path>")
        sys.exit(1)
    exfil(sys.argv[1], sys.argv[2])
