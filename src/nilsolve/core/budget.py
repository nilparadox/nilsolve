from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


BudgetName = Literal["fast", "robust", "thorough"]


@dataclass(frozen=True)
class BudgetStage:
    # Hard caps per candidate attempt
    time_limit_s: float
    max_nfev: int

    # Pilot fraction of the full time span used for this stage
    pilot_frac: float

    # Tolerance policy name to apply in this stage
    policy: str


@dataclass(frozen=True)
class BudgetPlan:
    name: BudgetName
    stages: Tuple[BudgetStage, ...]
    max_wall_s: float
    top_k_escalate: int


def get_budget_plan(name: BudgetName, max_wall_s: float | None = None) -> BudgetPlan:
    # Conservative, lab-safe defaults.
    if name == "fast":
        plan = BudgetPlan(
            name="fast",
            stages=(
                BudgetStage(time_limit_s=0.25, max_nfev=30_000, pilot_frac=0.03, policy="loose"),
            ),
            max_wall_s=8.0,
            top_k_escalate=3,
        )
    elif name == "robust":
        plan = BudgetPlan(
            name="robust",
            stages=(
                BudgetStage(time_limit_s=0.30, max_nfev=50_000, pilot_frac=0.03, policy="loose"),
                BudgetStage(time_limit_s=1.50, max_nfev=250_000, pilot_frac=0.10, policy="standard"),
            ),
            max_wall_s=25.0,
            top_k_escalate=4,
        )
    else:  # thorough
        plan = BudgetPlan(
            name="thorough",
            stages=(
                BudgetStage(time_limit_s=0.40, max_nfev=80_000, pilot_frac=0.03, policy="loose"),
                BudgetStage(time_limit_s=2.00, max_nfev=400_000, pilot_frac=0.12, policy="standard"),
                BudgetStage(time_limit_s=7.00, max_nfev=1_500_000, pilot_frac=0.35, policy="tight"),
            ),
            max_wall_s=60.0,
            top_k_escalate=5,
        )

    if max_wall_s is not None:
        return BudgetPlan(
            name=plan.name,
            stages=plan.stages,
            max_wall_s=float(max_wall_s),
            top_k_escalate=plan.top_k_escalate,
        )
    return plan
