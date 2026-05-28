"""Check Articulation API for velocity targeting."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    from isaacsim.core.prims import Articulation
    methods = sorted(m for m in dir(Articulation) if not m.startswith("_"))
    rel = [m for m in methods if "velocity" in m.lower() or "effort" in m.lower() or "target" in m.lower()]
    print("FOUND:", rel, flush=True)
finally:
    app.close()
