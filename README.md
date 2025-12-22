# nilsolve

A lab-grade nonlinear dynamics toolkit for:

- Phase 1: stiffness-aware ODE solving + reproducible solver ranking + 1D/2D parameter scans
- Phase 2: equilibrium continuation + stability + localized Hopf (and fold infrastructure)

Design principles:
- Rank once → reuse decision across scans
- Failures are recorded, not fatal
- System-agnostic (user ODEs live outside the library)
- Terminal-first (no UI assumptions)
- Clean separation: CLI parsing vs numerics vs I/O

## Install (editable)
From project root:

```bash
source .venv/bin/activate
pip install -e .
which nilsolve

