# Developer Guide

This document explains how to extend the repository safely. The most important principle is that PreceptualAI is **not** a single-mode codebase. It contains a primary UHCI system together with benchmark, deployment, and specialized research surfaces, so changes should be scoped, tested, and documented according to the layer they touch.[1]

## Development Principles

Good contributions to this repository do four things. They preserve clarity, reduce accidental coupling, keep benchmark claims honest, and document any new surface area introduced by the change.

| Principle | What it means in practice |
|---|---|
| Respect architectural boundaries | Do not blur core modeling, environment, serving, and federated layers unnecessarily |
| Prefer explicit interfaces | Add scripts, configs, or protocol definitions when creating user-facing behavior |
| Protect reproducibility | Do not make claims the repository artifacts cannot support |
| Update the docs with the code | New features should come with clear Markdown documentation |

## Recommended Contribution Flow

A sensible contribution workflow is to understand the UHCI system and the specific surface you are modifying first, then isolate the change to one layer, then validate the narrowest credible test surface, and finally update the relevant documentation files.

| Step | Recommended action |
|---|---|
| 1 | Read `README.md`, `ARCHITECTURE.md`, and `REPOSITORY_MAP.md` |
| 2 | Identify which package layer your change affects |
| 3 | Run the minimal relevant tests or validation commands |
| 4 | Update or add docs under `docs/` |
| 5 | Re-run validation and prepare a clean commit |

## Working with Optional Surfaces

Many advanced files are intentionally outside the default lightweight lint and type-check scopes. This means you should be explicit when changing GPU-heavy, UHCI-heavy, or deployment-heavy modules. Those changes may require targeted validation beyond the default local path.[2]

## Documentation Responsibility

The repository is deliberately documented as a system. If you add a new script, service, benchmark, or data path, update the documentation set, not just code comments. Readers should be able to understand the new behavior without reverse-engineering it from implementation details.

## References

[1]: [Architecture guide](ARCHITECTURE.md)
[2]: [Validation boundaries in `pyproject.toml`](../pyproject.toml)
