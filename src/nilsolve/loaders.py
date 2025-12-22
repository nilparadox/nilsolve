from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

from .spec import GoalSpec, ProblemSpec


@dataclass(frozen=True)
class LoadedSystem:
    problem: ProblemSpec
    goal: GoalSpec
    module_name: str
    module_path: str


def _stable_module_name(path: str) -> str:
    abspath = os.path.abspath(os.path.expanduser(path))
    h = hashlib.sha1(abspath.encode("utf-8")).hexdigest()[:16]
    return f"nilsolve_user_{h}"


def _load_module_from_path(path: str) -> ModuleType:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"System file not found: {path}")
    if not path.endswith(".py"):
        raise ValueError(f"System file must be a .py file: {path}")

    module_name = _stable_module_name(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_system(path: str) -> LoadedSystem:
    """
    Load a user-defined system from a Python file.

    Contract:
      - Must provide either:
          (a) build_problem() -> ProblemSpec
          (b) PROBLEM: ProblemSpec
      - Optional:
          build_goal() -> GoalSpec
          GOAL: GoalSpec

    If goal not provided, defaults to GoalSpec().
    """
    abspath = os.path.abspath(os.path.expanduser(path))
    mod = _load_module_from_path(abspath)

    # Problem
    problem: Optional[ProblemSpec] = None

    if hasattr(mod, "build_problem"):
        bp = getattr(mod, "build_problem")
        if not callable(bp):
            raise TypeError("Found 'build_problem' but it is not callable.")
        problem = bp()
    elif hasattr(mod, "PROBLEM"):
        problem = getattr(mod, "PROBLEM")

    if problem is None:
        raise ValueError(
            "System file must define either build_problem() -> ProblemSpec or PROBLEM = ProblemSpec(...)."
        )
    if not isinstance(problem, ProblemSpec):
        raise TypeError(f"Loaded problem is not a ProblemSpec. Got type: {type(problem)}")

    # Goal
    goal: GoalSpec
    if hasattr(mod, "build_goal"):
        bg = getattr(mod, "build_goal")
        if not callable(bg):
            raise TypeError("Found 'build_goal' but it is not callable.")
        goal_obj = bg()
        if not isinstance(goal_obj, GoalSpec):
            raise TypeError(f"build_goal() must return GoalSpec. Got: {type(goal_obj)}")
        goal = goal_obj
    elif hasattr(mod, "GOAL"):
        goal_obj = getattr(mod, "GOAL")
        if not isinstance(goal_obj, GoalSpec):
            raise TypeError(f"GOAL must be GoalSpec. Got: {type(goal_obj)}")
        goal = goal_obj
    else:
        goal = GoalSpec()

    return LoadedSystem(problem=problem, goal=goal, module_name=mod.__name__, module_path=abspath)
