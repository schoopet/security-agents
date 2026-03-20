"""
Custom Agent for Vertex AI Agent Engine
Reference: https://docs.cloud.google.com/agent-builder/agent-engine/develop/custom
"""

import vertexai
from vertexai import agent_engines

# --- Configuration ---
PROJECT_ID = "mmontan-ml-dev"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://mmontan-ml-dev-staging-bucket"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)


# --- Custom Agent Class ---
class MyAgent:
    """
    Custom agent following the Vertex AI Agent Engine pattern.

    Rules:
    - __init__: store config only (must be pickle-able, no service clients)
    - set_up:   initialize clients, models, graphs — called before serving
    - query:    handle a single request and return a complete response
    """

    def __init__(self, model: str, project: str, location: str):
        self.model_name = model
        self.project = project
        self.location = location
        print(f"[MyAgent.__init__] model={model}, project={project}, location={location}")

    def set_up(self):
        from google import genai

        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

    def query(self, input: str, **kwargs):
        print(f"[MyAgent.query] input={input}")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=input,
        )
        result = response.text
        print(f"[MyAgent.query] response={result}")
        return result


# --- Test locally before deploying ---
def test_locally():
    agent = MyAgent(
        model="gemini-2.0-flash",
        project=PROJECT_ID,
        location=LOCATION,
    )
    agent.set_up()
    response = agent.query(
        input="What is the exchange rate from US dollars to Swedish currency?"
    )
    print("Local response:", response)
    return agent


# --- Deploy to Agent Engine ---
def deploy():
    remote_agent = agent_engines.create(
        MyAgent(
            model="gemini-2.0-flash",
            project=PROJECT_ID,
            location=LOCATION,
        ),
        requirements=["google-cloud-aiplatform[agent_engines]", "google-genai"],
        display_name="working-custom",
    )
    print(f"Deployed agent resource name: {remote_agent.resource_name}")
    return remote_agent


# --- Query a deployed agent ---
def query_remote(resource_name: str, user_input: str):
    remote_agent = agent_engines.get(resource_name)
    response = remote_agent.query(input=user_input)
    print(f"Remote response: {response}")
    return response


if __name__ == "__main__":
    deploy()
