"""
Interactive shell against a deployed Agent Engine.

Usage:
    python receive.py <resource_name>

Commands:
    ls <path>             — list directory contents (full paths)
    get <file>            — fetch and print file contents
    decode <file>         — fetch a PEM/JSON certificate file and print human-readable details
    download <file>       — save remote file to current directory
    exec <cmd>            — execute a shell command remotely
    gsls gs://<bucket>    — list objects in a GCS bucket (uses agent ADC)
    gsfetch gs://<b>/<f>  — fetch and print a GCS file (uses agent ADC)
    exit                  — quit
"""

import sys
import base64
import vertexai

PROJECT_ID = "mmontan-ml-dev"
LOCATION = "us-central1"


def print_cert(cert):
    from cryptography import x509
    print(f"  Subject:    {cert.subject.rfc4514_string()}")
    print(f"  Issuer:     {cert.issuer.rfc4514_string()}")
    print(f"  Valid from: {cert.not_valid_before_utc}")
    print(f"  Valid to:   {cert.not_valid_after_utc}")
    print(f"  Serial:     {cert.serial_number}")
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        print(f"  SANs:       {san.value.get_values_for_type(x509.DNSName) + san.value.get_values_for_type(x509.UniformResourceIdentifier)}")
    except x509.ExtensionNotFound:
        pass
    print()


def print_trust_bundle(data: bytes):
    import json
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    bundle = json.loads(data)
    trust_domains = bundle.get("trust_domains", {})
    for domain, domain_data in trust_domains.items():
        print(f"=== Trust Domain: {domain} ===\n")
        for i, key in enumerate(domain_data.get("keys", [])):
            for j, der_b64 in enumerate(key.get("x5c", [])):
                der = base64.b64decode(der_b64)
                cert = x509.load_der_x509_certificate(der, default_backend())
                print(f"--- Key {i+1}, Certificate {j+1} ---")
                print_cert(cert)


def print_certs(pem_data: bytes):
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    pem_str = pem_data.decode("utf-8", errors="replace")
    certs = [
        ("-----BEGIN CERTIFICATE-----" + chunk).encode("utf-8")
        for chunk in pem_str.split("-----BEGIN CERTIFICATE-----")
        if "-----END CERTIFICATE-----" in chunk
    ]

    for i, pem_cert in enumerate(certs):
        cert = x509.load_pem_x509_certificate(pem_cert, default_backend())
        print(f"--- Certificate {i + 1} ---")
        print_cert(cert)


def chat(resource_name: str):
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    agent = client.agent_engines.get(name=resource_name)
    print(f"[*] Connected to {resource_name}")
    print("[*] Commands: ls <path>, get <file>, decode <file>, download <file>, exec <cmd>, gsls gs://<bucket>, gsfetch gs://<bucket>/<file>, exit\n")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        if cmd == "exit":
            break

        if cmd.startswith("download "):
            path = cmd[9:].strip()
            response = agent.query(input=f"get {path}")
            if response.startswith("ERROR"):
                print(response)
            else:
                filename = path.split("/")[-1]
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(response))
                print(f"Saved to {filename}")
            continue

        if cmd.startswith("decode "):
            path = cmd[7:].strip()
            response = agent.query(input=f"get {path}")
            if response.startswith("ERROR"):
                print(response)
            elif path.endswith(".json"):
                print_trust_bundle(base64.b64decode(response))
            else:
                print_certs(base64.b64decode(response))
            continue

        response = agent.query(input=cmd)

        if cmd.startswith("get ") and not response.startswith("ERROR"):
            decoded = base64.b64decode(response).decode("utf-8", errors="replace")
            print(decoded)
        else:
            print(response)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python receive.py <resource_name>")
        sys.exit(1)
    chat(sys.argv[1])
