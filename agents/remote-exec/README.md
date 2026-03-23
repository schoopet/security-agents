# Remote Exec Agent for Vertex AI Agent Engine

This agent demonstrates remote interactive code execution on a deployed Vertex AI Agent Engine. By sending arbitrary Python via the agent's `query()` handler, a caller can execute code inside the agent's runtime environment — reading files, querying the GCP metadata server, and inspecting workload identity credentials.

`get_token.py` uses this capability to fetch and analyze the agent's SPIFFE certificate and GCP access token from the metadata server.

---

# Custom Agent for Vertex AI Agent Engine

## How does the code get shipped??

When you call `agent_engines.create(MyAgent(...), ...)`, the SDK needs to serialize your agent object and upload it to GCS so the remote environment can reconstruct it. It does this using **pickle** — but the behavior differs depending on where your class is defined.

### The key: `MyAgent.__module__`

Python tracks where every class was defined via `__module__`. When the remote environment unpickles your agent, it uses this to find the class.

- **Regular pickle** stores only a *pointer*: the module name + class name. On unpickling it does `import <module>; getattr(<module>, <classname>)`. If the module isn't available, it crashes.
- **Cloudpickle** (automatically added to requirements by the SDK) stores the actual *bytecode* of the class — methods, code objects, closures, everything. No imports needed to reconstruct it.

Cloudpickle kicks in when `__module__ == "__main__"`. That happens when the class is defined in the file Python is currently executing as the entry point.

---

### Case 1: Class defined in `__main__` — no `extra_packages` needed

```python
# agent.py — run as: python agent.py

import vertexai
from vertexai import agent_engines

class MyAgent:
    def __init__(self, model):
        self.model = model

    def set_up(self):
        pass

    def query(self, input, **kwargs):
        return f"response to: {input}"

def deploy():
    agent_engines.create(
        MyAgent(model="gemini-2.0-flash"),
        requirements=["google-cloud-aiplatform[agent_engines]", "google-genai"],
        display_name="my-agent",
    )

if __name__ == "__main__":
    deploy()
```

When you run `python agent.py`, Python sets `__name__ = "__main__"` for that file, so `MyAgent.__module__` becomes `"__main__"`. Cloudpickle serializes the full class bytecode into the pickle (~2KB). The remote environment reconstructs the class directly from the pickle — no source file needed.

Same applies to **Colab**: any class defined directly in a cell has `__module__ == "__main__"`.

---

### Case 2: Class imported from another file — `extra_packages` required

```python
# my_agent.py
class MyAgent:
    def __init__(self, model):
        self.model = model

    def set_up(self):
        pass

    def query(self, input, **kwargs):
        return f"response to: {input}"
```

```python
# deploy.py — run as: python deploy.py

from my_agent import MyAgent  # MyAgent.__module__ is now "my_agent"
from vertexai import agent_engines

agent_engines.create(
    MyAgent(model="gemini-2.0-flash"),
    requirements=["google-cloud-aiplatform[agent_engines]", "google-genai"],
    extra_packages=["my_agent.py"],  # required — pickle only stored a pointer to "my_agent.MyAgent"
    display_name="my-agent",
)
```

Here `MyAgent.__module__` is `"my_agent"` — where it was *defined*, not where `deploy()` is called from. Regular pickle stores a pointer. The remote environment tries `import my_agent` and fails unless `my_agent.py` is shipped via `extra_packages`.

---

### Summary

| Where class is defined | `__module__` | Pickle behavior | `extra_packages` needed? |
|---|---|---|---|
| Same file, run as entry point (`python agent.py`) | `__main__` | Cloudpickle embeds bytecode | No |
| Colab cell | `__main__` | Cloudpickle embeds bytecode | No |
| Imported from another file | `"my_agent"` | Regular pickle stores pointer | Yes |
