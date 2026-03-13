"""Generate all 2 Jupyter notebooks for the Caching module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "11-caching"))
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def write_notebook(filename: str, cells: list) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = KERNEL
    nb.cells = cells
    path = os.path.join(OUTPUT_DIR, filename)
    nbformat.write(nb, path)
    print(f"  Created {path}")


def md(source: str):
    return nbformat.v4.new_markdown_cell(source)


def code(source: str):
    return nbformat.v4.new_code_cell(source)


# ---------------------------------------------------------------------------
# 01 - Cache Backend Protocol
# ---------------------------------------------------------------------------
def nb_cache_backend():
    write_notebook("01_cache_backend.ipynb", [
        md(
            "# Cache Backend Protocol\n"
            "> Understand the caching interface and implement a custom backend.\n"
            "\n"
            "`CacheBackend` is a protocol that defines four methods: `get()`, `set()`,\n"
            "`delete()`, and `clear()`.  Any object implementing these methods can serve\n"
            "as a cache in Anchor \u2014 no inheritance required.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.protocols import CacheBackend"
        ),
        md(
            "## The CacheBackend Protocol\n"
            "The protocol defines a minimal key-value cache interface:\n"
            "- `get(key)` \u2013 retrieve a cached value (or `None`)\n"
            "- `set(key, value, ttl=None)` \u2013 store a value with optional TTL\n"
            "- `delete(key)` \u2013 remove a single entry\n"
            "- `clear()` \u2013 wipe the entire cache"
        ),
        code(
            "# Inspect the protocol methods\n"
            "print(\"CacheBackend protocol methods:\")\n"
            "for method in [\"get\", \"set\", \"delete\", \"clear\"]:\n"
            "    print(f\"  - {method}()\")"
        ),
        md(
            "## Build a Custom Redis-Like Cache\n"
            "A stub implementation showing the required method signatures."
        ),
        code(
            "class RedisCache:\n"
            "    \"\"\"Stub that mimics a Redis-backed cache.\"\"\"\n"
            "\n"
            "    def __init__(self):\n"
            "        self._store = {}\n"
            "\n"
            "    def get(self, key: str):\n"
            "        value = self._store.get(key)\n"
            "        print(f\"GET '{key}' -> {value}\")\n"
            "        return value\n"
            "\n"
            "    def set(self, key: str, value, ttl=None):\n"
            "        self._store[key] = value\n"
            "        ttl_msg = f\" (ttl={ttl}s)\" if ttl else \"\"\n"
            "        print(f\"SET '{key}' = {value}{ttl_msg}\")\n"
            "\n"
            "    def delete(self, key: str) -> bool:\n"
            "        removed = self._store.pop(key, None) is not None\n"
            "        print(f\"DEL '{key}' -> {'removed' if removed else 'not found'}\")\n"
            "        return removed\n"
            "\n"
            "    def clear(self):\n"
            "        count = len(self._store)\n"
            "        self._store.clear()\n"
            "        print(f\"CLEAR -> removed {count} entries\")\n"
            "\n"
            "\n"
            "cache = RedisCache()\n"
            "print(f\"Type: {type(cache).__name__}\")"
        ),
        md(
            "## Verify Protocol Conformance\n"
            "`isinstance()` confirms the class satisfies the protocol."
        ),
        code(
            "is_valid = isinstance(cache, CacheBackend)\n"
            "print(f\"Satisfies CacheBackend protocol: {is_valid}\")\n"
            "\n"
            "assert is_valid, \"RedisCache must implement CacheBackend\""
        ),
        md(
            "## Exercise the Custom Cache\n"
            "Run through a typical set/get/delete/clear cycle."
        ),
        code(
            "# Store values\n"
            "cache.set(\"user:1\", {\"name\": \"Alice\", \"role\": \"admin\"})\n"
            "cache.set(\"user:2\", {\"name\": \"Bob\", \"role\": \"viewer\"}, ttl=300)\n"
            "\n"
            "# Retrieve\n"
            "print()\n"
            "cache.get(\"user:1\")\n"
            "cache.get(\"user:99\")  # miss\n"
            "\n"
            "# Delete one entry\n"
            "print()\n"
            "cache.delete(\"user:2\")\n"
            "cache.delete(\"user:99\")  # no-op\n"
            "\n"
            "# Clear everything\n"
            "print()\n"
            "cache.clear()"
        ),
        md(
            "## Key Takeaways\n"
            "- `CacheBackend` is a four-method protocol: `get`, `set`, `delete`, `clear`\n"
            "- Any class matching the signature is accepted \u2014 no base class needed\n"
            "- Use `isinstance()` for runtime protocol verification\n"
            "- Custom backends can wrap Redis, Memcached, DynamoDB, or any store\n"
            "\n"
            "**Next:** [In-Memory Cache](02_in_memory_cache.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - In-Memory Cache
# ---------------------------------------------------------------------------
def nb_in_memory_cache():
    write_notebook("02_in_memory_cache.ipynb", [
        md(
            "# In-Memory Cache\n"
            "> A batteries-included cache with TTL and size limits.\n"
            "\n"
            "`InMemoryCacheBackend` provides a thread-safe, dictionary-backed cache with\n"
            "configurable default TTL and maximum entry count.  It implements the\n"
            "`CacheBackend` protocol and is ready to use out of the box.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.cache import InMemoryCacheBackend"
        ),
        md(
            "## Create the Cache\n"
            "Specify a default TTL (in seconds) and maximum number of entries."
        ),
        code(
            "cache = InMemoryCacheBackend(default_ttl=300.0, max_size=1000)\n"
            "\n"
            "print(f\"Default TTL: {cache.default_ttl}s\")\n"
            "print(f\"Max size:    {cache.max_size} entries\")"
        ),
        md(
            "## Store and Retrieve Values\n"
            "`.set()` accepts any serializable value.  `.get()` returns `None` on a miss."
        ),
        code(
            "# Store a dictionary\n"
            "cache.set(\"key1\", {\"data\": \"value\"})\n"
            "print(f\"Stored 'key1'\")\n"
            "\n"
            "# Store a list with a custom TTL\n"
            "cache.set(\"key2\", [1, 2, 3], ttl=60.0)\n"
            "print(f\"Stored 'key2' with 60s TTL\")\n"
            "\n"
            "# Retrieve\n"
            "result = cache.get(\"key1\")\n"
            "print(f\"\\nGET 'key1': {result}\")\n"
            "\n"
            "# Miss\n"
            "miss = cache.get(\"nonexistent\")\n"
            "print(f\"GET 'nonexistent': {miss}\")"
        ),
        md(
            "## Delete Entries\n"
            "Remove a single key with `.delete()`."
        ),
        code(
            "deleted = cache.delete(\"key2\")\n"
            "print(f\"Deleted 'key2': {deleted}\")\n"
            "\n"
            "# Confirm it's gone\n"
            "print(f\"GET 'key2' after delete: {cache.get('key2')}\")"
        ),
        md(
            "## Bulk Operations\n"
            "Store multiple entries, then clear the entire cache."
        ),
        code(
            "# Populate\n"
            "for i in range(5):\n"
            "    cache.set(f\"item:{i}\", f\"value-{i}\")\n"
            "    print(f\"  Stored 'item:{i}'\")\n"
            "\n"
            "# Verify one exists\n"
            "print(f\"\\nGET 'item:3': {cache.get('item:3')}\")\n"
            "\n"
            "# Clear all\n"
            "cache.clear()\n"
            "print(f\"\\nAfter clear, GET 'item:3': {cache.get('item:3')}\")"
        ),
        md(
            "## Typical Usage Pattern\n"
            "Cache expensive computations like embeddings or API responses."
        ),
        code(
            "import hashlib\n"
            "\n"
            "\n"
            "def compute_embedding(text: str) -> list:\n"
            "    \"\"\"Simulate an expensive embedding call.\"\"\"\n"
            "    print(f\"  Computing embedding for: '{text[:30]}...'\")\n"
            "    return [0.1, 0.2, 0.3]  # mock\n"
            "\n"
            "\n"
            "def get_embedding(text: str, cache: InMemoryCacheBackend) -> list:\n"
            "    key = hashlib.sha256(text.encode()).hexdigest()[:16]\n"
            "    cached = cache.get(key)\n"
            "    if cached is not None:\n"
            "        print(f\"  Cache HIT for '{text[:30]}...'\")\n"
            "        return cached\n"
            "    result = compute_embedding(text)\n"
            "    cache.set(key, result)\n"
            "    return result\n"
            "\n"
            "\n"
            "embed_cache = InMemoryCacheBackend(default_ttl=600.0, max_size=500)\n"
            "\n"
            "# First call: cache miss\n"
            "print(\"Call 1:\")\n"
            "emb1 = get_embedding(\"What is machine learning?\", embed_cache)\n"
            "\n"
            "# Second call: cache hit\n"
            "print(\"\\nCall 2 (same input):\")\n"
            "emb2 = get_embedding(\"What is machine learning?\", embed_cache)\n"
            "\n"
            "print(f\"\\nSame result: {emb1 == emb2}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `InMemoryCacheBackend` supports default TTL and max size\n"
            "- `.set()` stores any value; `.get()` returns `None` on a miss\n"
            "- `.delete()` removes one key; `.clear()` wipes everything\n"
            "- Great for caching embeddings, API responses, and computed results\n"
            "\n"
            "**Previous:** [Cache Backend Protocol](01_cache_backend.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Caching notebooks in {OUTPUT_DIR} ...")
    nb_cache_backend()
    nb_in_memory_cache()
    print("Done \u2714")


if __name__ == "__main__":
    main()
