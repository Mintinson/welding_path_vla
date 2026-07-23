# Package the Elfin5 model with the simulator

The Elfin5 URDF, source MJCF, visual meshes, and welding scene live under the Python package assets. This makes model lookup independent of the current working directory and ensures laptop and server installations use the same robot revision.

The welding scene uses source inertial parameters and visual STL meshes, while low-complexity primitive collision proxies remain separate. The source LIBERO scene is retained for provenance but is not used by data collection.
