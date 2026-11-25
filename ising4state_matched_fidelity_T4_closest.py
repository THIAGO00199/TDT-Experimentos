import json
from scipy.optimize import brentq
import ising4state_matched_fidelity_optimization_T4 as m
out={}
for prof in ["linear","ramp_hold"]:
 ref=m.run("ref",12.0,prof)
 target=ref["fidelity"]
 obsA=brentq(lambda a:m.run("obs",a,prof)["fidelity"]-target,0.0,12.0,xtol=2e-5)
 obs=m.run("obs",obsA,prof)
 delta=ref["dissipated_work"]-obs["dissipated_work"]
 out[prof]={"matched_fidelity":target,"ref_amplitude":12.0,"ref":ref,"obs_amplitude":obsA,"obs":obs,
 "dissipation_saving_before_memory":delta,
 "net_after_conditional_Landauer":delta-m.KT*__import__("numpy").log(2)*m.Hcond,
 "net_after_dynamic_reset":delta-5.364432689279152,
 "net_after_dynamic_reset_plus_MT_reset":delta-5.364432689279152-m.KT*__import__("numpy").log(2)*m.HM}
print(json.dumps(out,indent=2))
with open("ising4state_matched_fidelity_T4_closest.json","w") as f:json.dump(out,f,indent=2)
