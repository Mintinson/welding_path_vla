import numpy as np

from welding_path_vla.core.domain import Seam


def test_seam_is_defined_in_workpiece_coordinates() -> None:
    seam = Seam(
        "straight", np.array([0.0, -0.1, 0.0]), np.array([0.0, 0.1, 0.0]), np.array([1.0, 0.0, 1.0])
    )
    assert seam.length == 0.2
    np.testing.assert_allclose(seam.tangent_local, [0, 1, 0])
