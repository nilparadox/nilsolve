"""
nilsolve: Lab-grade nonlinear dynamics toolkit.

Public API (stable):
- Data models used by user-defined ODE systems

Internal APIs (solver logic, CLI) are intentionally not exported here.
"""

from __future__ import annotations

# Stable data models for user systems
from .models import ProblemSpec, GoalSpec

__all__ = [
    "ProblemSpec",
    "GoalSpec",
]
