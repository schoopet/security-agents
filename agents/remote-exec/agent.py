"""
Custom Agent for Vertex AI Agent Engine
Reference: https://docs.cloud.google.com/agent-builder/agent-engine/develop/custom
"""

import argparse
import os
import vertexai


# --- Custom Agent Class ---
class MyAgent:

    def set_up(self):
        print("[MyAgent] Agent is ready.")

    def query(self, input: str, **kwargs):
        import os
        import base64

        print(f"[MyAgent.query] input={input}")

        if input.startswith("ls "):
            path = input[3:].strip()
            try:
                result = "\n".join(os.path.join(path, e) for e in os.listdir(path))
            except Exception as e:
                result = f"ERROR: {e}"

        elif input.startswith("get "):
            path = input[4:].strip()
            try:
                with open(path, "rb") as f:
                    result = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                result = f"ERROR: {e}"

        elif input.startswith("exec "):
            import subprocess
            cmd = input[5:].strip()
            try:
                out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                result = out.stdout + out.stderr
            except Exception as e:
                result = f"ERROR: {e}"

        elif input.startswith("gsls "):
            from google.cloud import storage
            uri = input[5:].strip()
            if not uri.startswith("gs://"):
                result = "ERROR: URI must start with gs://"
            else:
                bucket_name = uri[5:].rstrip("/")
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blobs = client.list_blobs(bucket_name)
                    lines = [f"gs://{bucket_name}/{b.name}" for b in blobs]
                    result = "\n".join(lines) if lines else "(empty bucket)"
                except Exception as e:
                    result = f"ERROR: {e}"

        elif input.startswith("gsfetch "):
            from google.cloud import storage
            uri = input[8:].strip()
            if not uri.startswith("gs://"):
                result = "ERROR: URI must start with gs://"
            else:
                path = uri[5:]
                bucket_name, _, blob_name = path.partition("/")
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    result = blob.download_as_text()
                except Exception as e:
                    result = f"ERROR: {e}"

        else:
            result = "ERROR: unknown command. Use 'ls <path>', 'get <file>', 'exec <cmd>', 'gsls gs://<bucket>', or 'gsfetch gs://<bucket>/<file>'"

        print(f"[MyAgent.query] response={result}")
        return result


# --- Deploy to Agent Engine ---
def deploy(project: str, region: str, staging_bucket: str, base_url: str | None = None):
    from google.genai.types import HttpOptions
    kwargs: dict = {"project": project, "location": region}
    if base_url:
        kwargs["http_options"] = HttpOptions(baseUrl=base_url)
    client = vertexai.Client(**kwargs)

    effective_base = (base_url or f"https://{region}-aiplatform.googleapis.com").rstrip("/")
    deploy_endpoint = f"{effective_base}/v1/projects/{project}/locations/{region}/reasoningEngines"
    print(f"[*] Deploying to: {deploy_endpoint}")
    remote_agent = client.agent_engines.create(
        agent=MyAgent(),
        config={
            "display_name": "working-custom",
            "identity_type": "AGENT_IDENTITY",
            "requirements": ["google-cloud-aiplatform[agent_engines]", "cloudpickle", "pydantic", "google-auth[cryptography]", "google-cloud-storage"],
            "staging_bucket": staging_bucket,
        },
    )
    print(f"Deployed agent: {remote_agent.model_dump()}")
    return remote_agent


# --- Query a deployed agent ---
def query_remote(resource_name: str, user_input: str, project: str, region: str):
    client = vertexai.Client(project=project, location=region)
    remote_agent = client.agent_engines.get(resource_name)
    response = remote_agent.query(input=user_input)
    print(f"Remote response: {response}")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a custom agent to Vertex AI Agent Engine")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), required=not os.environ.get("GOOGLE_CLOUD_PROJECT"), help="GCP project ID")
    parser.add_argument("--region", default=os.environ.get("GOOGLE_CLOUD_REGION"), required=not os.environ.get("GOOGLE_CLOUD_REGION"), help="GCP region")
    parser.add_argument("--staging-bucket", default=os.environ.get("STAGING_BUCKET"), required=not os.environ.get("STAGING_BUCKET"), help="GCS staging bucket (gs://...)")
    parser.add_argument("--base-url", default=None, help="Override Vertex AI API base URL")
    args = parser.parse_args()
    deploy(args.project, args.region, args.staging_bucket, base_url=args.base_url)
