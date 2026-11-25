import json
import ising4state_matched_fidelity_optimization_T4 as m
out={}
for p in ["linear","ramp_hold"]:
 out[p]={str(a):m.run("ref",a,p) for a in [8,10,12,14,16,20,24]}
print(json.dumps(out,indent=2))
