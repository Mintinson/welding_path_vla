"""cps_client — HansRobot CPS TCP protocol client.

Usage::

    from cps_client import CPSClient

    cps = CPSClient()
    cps.HRIF_Connect(0, "192.168.56.101", 10003)
    ...
"""

from .cps import CPSClient

__all__ = ["CPSClient"]
