"""Planner: Compositional World Model (CWM, paradigm H2).

Explicit physics modules + learned residual, instead of monolithic neural
world model. Operators trust ITU-R / 3GPP physics; the neural network is
responsible only for the residual that physics misses.

See PARADIGMS.md §H2 for rationale.
"""
