# OR-Clarify data card

## Overview

OR-Clarify is a 100-case benchmark for evaluating whether an agent can recover missing formulation-critical facts through interaction before optimization modeling.

## Case contents

Each canonical TOML case contains:

- an incomplete public business brief;
- numbered problem units and private case facts;
- formulation-critical hidden slots with severity labels;
- fact-bounded simulator answers;
- semantic judge rubrics and acceptable-question guidance.

## Size and evaluation role

- Cases: 100
- Hidden slots: 178
- Severity: 75 P0, 83 P1, 20 P2
- Core-eligible cases containing at least one P0/P1 slot: 94

## Information boundary

The tested agent receives only the public brief and public transcript. Hidden slots, simulator-private facts, and judge rubrics are never provided to the tested agent. The simulator receives private case facts but must answer only the asked question. The post-hoc judge receives the final public transcript and the frozen hidden-slot rubrics.

## Canonical form

The TOML files are the canonical machine-readable cases.
