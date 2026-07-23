# Keep framework-independent episodes as the source of truth

Simulation writes complete NPZ, MP4, and JSON episodes before exporting LeRobot datasets. This preserves command and execution trajectories and permits future action horizons or training frameworks without recollecting data.

