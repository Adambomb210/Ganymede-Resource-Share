"""The Ganymede worker: the package a contributor installs (docs/02-architecture-v2.md 4).

Broad hardware compatibility is a project goal, so this is a pip-installable
package first and a container second (4.1). Every delivery path -- Linux
container, native Linux, macOS launchd, Windows Scheduled Task -- wraps this same
code.

``worker-core`` depends on **torch and the standard library only**. The trainer's
stack (`transformers`, `peft`, `datasets`) lives in ``worker-llm`` on top of it,
because the loop, the client and the probe have no use for it and a ~2 GB layer
that most of the code never touches is a layer nobody should pay for.
"""

__all__ = ["probe", "client", "control", "loop"]
