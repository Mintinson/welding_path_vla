# Use one Pixi workspace with explicit environments

The project uses one lock file and composable `sim`, `data`, `real`, `train`, `deploy`, and `dev` environments. This keeps laptop collection dependencies separate from server training without allowing their shared package versions to drift.

