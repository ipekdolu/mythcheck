"""Phase 0 checkpoint: verify Neo4j AuraDB, Claude API, and Langfuse are all reachable.

Run with: python scripts/check_setup.py
"""
import sys

sys.path.insert(0, ".")

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


def check_neo4j() -> bool:
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 'hello from AuraDB' AS message")
            message = result.single()["message"]
        driver.close()
        print(f"[OK] Neo4j AuraDB: {message}")
        return True
    except Exception as e:
        print(f"[FAIL] Neo4j AuraDB: {e}")
        return False


def check_anthropic() -> bool:
    from anthropic import Anthropic

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        )
        text = response.content[0].text.strip()
        print(f"[OK] Claude API ({CLAUDE_MODEL}): {text!r}")
        return True
    except Exception as e:
        print(f"[FAIL] Claude API: {e}")
        return False


def check_langfuse() -> bool:
    from langfuse import Langfuse

    try:
        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        trace = client.trace(name="phase0-connectivity-check", input="setup check")
        trace.update(output="ok")
        client.flush()
        print(
            "[OK] Langfuse: trace sent (name='phase0-connectivity-check') — "
            f"check the dashboard at {LANGFUSE_HOST}"
        )
        return True
    except Exception as e:
        print(f"[FAIL] Langfuse: {e}")
        return False


if __name__ == "__main__":
    results = {
        "Neo4j AuraDB": check_neo4j(),
        "Claude API": check_anthropic(),
        "Langfuse": check_langfuse(),
    }
    print()
    if all(results.values()):
        print("All checks passed.")
        sys.exit(0)
    else:
        failed = [name for name, ok in results.items() if not ok]
        print(f"Failed checks: {', '.join(failed)}")
        sys.exit(1)
