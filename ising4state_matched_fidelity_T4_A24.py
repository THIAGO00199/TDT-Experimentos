import json, numpy as np
from scipy.optimize import brentq
import ising4state_matched_fidelity_optimization_T4 as m
out={}
for prof in ["linear","ramp_hold"]:
 ref=m.run("ref",24.0,prof); target=ref["fidelity"]
 obsA=brentq(lambda a:m.run("obs",a,prof)["fidelity"]-target,0,24,xtol=2e-5)
 obs=m.run("obs",obsA,prof); delta=ref["dissipated_work"]-obs["dissipated_work"]
 out[prof]={"matched_fidelity":target,"reference_amplitude":24.0,"reference":ref,"history_amplitude":obsA,"history":obs,
 "dissipation_saving_before_memory":delta,
 "net_after_Landauer_conditional_reset":delta-m.KT*np.log(2)*m.Hcond,
 "net_after_dynamic_reset_tau1000":delta-5.364432689279152}
print(json.dumps(out,indent=2))
with open("ising4state_matched_fidelity_T4_A24.json","w") as f:json.dump(out,f,indent=2)
