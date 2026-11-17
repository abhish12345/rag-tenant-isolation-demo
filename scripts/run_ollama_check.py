#!/usr/bin/env python3
"""Verify that Ollama is reachable and command-r:35b is available."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from app.config import settings

def check_ollama():
    base = settings.ollama_base_url
    print(f"Checking Ollama at {base} ...")
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = resp.json()
        models = [m["name"] for m in tags.get("models", [])]
        print(f"Available models: {models}")
        target = settings.ollama_model
        # Check if the target model (or a variant) is listed
        found = any(target in m for m in models)
        if found:
            print(f"OK: {target} is available.")
        else:
            print(f"WARNING: {target} not found in model list. Pull it with: ollama pull {target}")
            sys.exit(1)
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to Ollama at {base}. Is it running?")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Test a minimal generate call
    print(f"\nTesting generate with {target} (short prompt) ...")
    try:
        resp = httpx.post(
            f"{base}/api/generate",
            json={"model": target, "prompt": "Say OK", "stream": False, "options": {"num_predict": 5}},
            timeout=120,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")
        print(f"Response: {response_text!r}")
        print("Ollama check passed.")
    except Exception as e:
        print(f"ERROR during generate: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_ollama()
