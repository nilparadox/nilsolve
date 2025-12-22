from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EventFlags:
    fold: bool
    hopf: bool
    note: str


def detect_events(
    *,
    prev_dmu_ds: Optional[float],
    dmu_ds: float,
    jac_smin: float,
    prev_max_real: Optional[float],
    max_real: float,
    eigvals: np.ndarray,
    svd_sing_tol: float,
    hopf_imag_min: float,
) -> EventFlags:
    # Fold heuristic: near singular Jacobian AND tangent reversal in dmu/ds
    fold = False
    if prev_dmu_ds is not None:
        if (jac_smin < svd_sing_tol) and (np.sign(prev_dmu_ds) != np.sign(dmu_ds)):
            fold = True

    # Hopf heuristic: max real part crosses 0 and there is a complex pair
    hopf = False
    if prev_max_real is not None:
        crossed = (prev_max_real < 0.0 and max_real > 0.0) or (prev_max_real > 0.0 and max_real < 0.0)
        if crossed:
            imag = np.abs(np.imag(eigvals))
            has_complex = bool(np.any(imag > hopf_imag_min))
            if has_complex:
                hopf = True

    note_parts = []
    if fold:
        note_parts.append("FOLD?")
    if hopf:
        note_parts.append("HOPF?")
    note = ",".join(note_parts) if note_parts else ""
    return EventFlags(fold=fold, hopf=hopf, note=note)
