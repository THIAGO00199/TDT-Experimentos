import json
import ising4state_matched_fidelity_optimization_T4 as m
out={p:{str(a):m.run("ref",a,p) for a in [24,48,96]} for p in ["linear","ramp_hold"]}
print(json.dumps(out,indent=2))
