# Host-agent tests

A sibling of `tests/`, not a subdirectory of it, for one reason: pytest loads
every `conftest.py` between a test file and the rootdir, and `tests/conftest.py`
imports `torch` and `fastapi` at module scope for the coordinator fixtures. A
host test living under `tests/` therefore cannot be collected without the whole
coordinator dependency set installed.

That would quietly undo the rule these tests exist to hold up. `ganymede/host`
imports **nothing outside the standard library** — that is what lets a
contributor install the agent by copying a directory onto a machine that has
only a Python interpreter. If verifying it required torch, nobody would ever
find out when the rule broke, because the environment that could notice would
never exist.

So:

```sh
python -m pip install pytest        # and nothing else
python -m pytest tests_host/ -q
```

is the whole setup, on all three platforms. That is also what makes the macOS
and Windows CI jobs cheap enough to run on every push — see
`.github/workflows/ci.yml`, which additionally exercises each platform's real
scheduler and real idle probe, neither of which exists on a Linux container.
