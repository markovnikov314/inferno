# ADR-0001: P0 Scope Lock

## Status

Accepted

## Decision

P0 implements only control-plane files, durable state, redaction policy, and
GPU/container preflight evidence.

## Consequence

Benchmark runners, serving adapters, dashboards, planners, routers, model
downloads, and fake product evidence are deferred until their named phases.
