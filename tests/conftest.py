"""Allow offline unit tests to import vehicle code without simulator transports.

Only absent RflySim SDK modules are replaced. Tests supply their own vehicle
fakes; no simulator or flight connection is opened by these placeholders.
"""

import importlib.util
import sys
from types import ModuleType

for name in ("PX4MavCtrlV4", "ReqCopterSim", "UE4CtrlAPI"):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        sys.modules[name] = ModuleType(name)
