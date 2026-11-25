import numpy as np, json
from scipy.integrate import solve_ivp

beta, kT, nu, B0 = 0.4, 2.5, 1.0, 0.5
symbols = [-4,-2,0,2,4]
codes = {m:i for i,m in enumerate(symbols)}
pm = np.array([0.22201093570274258,0.17929293950047626,
               0.1973922495935625,0.1792929395004762,0.22201093570274258])
delta = 1e-6
p = np.zeros(8); p[:5]=pm
p = (p + delta/8); p /= p.sum()
# Stage A: write/read logic uses a controlled-XOR permutation on a degenerate 3-bit register.
# Stage B: prepare a Gibbs-compatible energy landscape while the logical state is frozen.
E0 = -kT*np.log(p); E0 -= E0.min()
W_prepare = float(np.dot(p,E0))
# Stage C: quasistatic-style reset ramp; target excitation 30 makes residual error tiny.
E1 = np.full(8,30.0); E1[0]=0.0
pairs=[(i,j) for i in range(8) for j in range(i+1,8)]
def generator(E):
    Q=np.zeros((8,8))
    for i,j in pairs:
        saddle=max(E[i],E[j])+B0
        Q[i,j]=nu*np.exp(-beta*(saddle-E[i]))
        Q[j,i]=nu*np.exp(-beta*(saddle-E[j]))
    Q[np.diag_indices(8)]=-Q.sum(axis=1)
    return Q
def Feq(E):
    z=np.exp(-beta*(E-np.min(E)))
    return float(np.min(E)-kT*np.log(z.sum()))
def run_reset(tau):
    dE=E1-E0
    def rhs(t,z):
        E=E0+(t/tau)*dE
        prob=z[:8]
        return np.r_[prob@generator(E), float(np.dot(prob,dE))/tau]
    sol=solve_ivp(rhs,(0,tau),np.r_[p,0.0],method="DOP853",rtol=2e-10,atol=2e-12)
    pf=sol.y[:8,-1]; Wreset=float(sol.y[8,-1])
    dF=Feq(E1)-Feq(E0)
    return {"tau":tau,"work_dynamic_reset":Wreset,"deltaF_reset_stage":dF,
            "dissipation_reset_stage":Wreset-dF,
            "work_landscape_preparation":W_prepare,
            "total_work_prepare_plus_reset":W_prepare+Wreset,
            "P_000_final":float(pf[0]),
            "final_entropy_bits":float(-np.sum(pf[pf>0]*np.log2(pf[pf>0]))),
            "final_distribution":pf.tolist(),"solver_nfev":int(sol.nfev)}
# Logical controlled-XOR is an involution: applying twice restores all 3 register bits.
gate_checks=[]
for m in symbols:
    c=codes[m]
    for r in range(8):
        gate_checks.append(((r^c)^c)==r)
H_M=-float(np.sum(pm*np.log2(pm)))
H_reg=-float(np.sum(p[p>0]*np.log2(p[p>0])))
W_landauer=kT*np.log(2)*H_M
Q0=generator(E0)
eig=np.linalg.eigvals(Q0); nz=np.sort(np.abs(np.real(eig[np.abs(eig)>1e-9])))
tau_relax=float(1/nz[0])
results=[run_reset(t) for t in [10.0,100.0,1000.0]]
out={
 "model":"3-bit register with 8 physical states; five M_T codewords and three unused patterns regularized at 1e-6",
 "parameters":{"beta":beta,"kT":kT,"nu":nu,"barrier_offset":B0,"final_excitation":30.0,"regularization_per_state":delta},
 "encoding":{str(m):format(c,"03b") for m,c in codes.items()},
 "write_read":{"controlled_XOR_roundtrip_all_inputs":all(gate_checks),"failures":sum(not x for x in gate_checks),
               "ideal_gate_work_degenerate_register":0.0,
               "qualification":"Logical reversibility only. A physical gate Hamiltonian/coupling and finite-time gate dissipation are not modeled."},
 "reset_protocol":{"initial_register_distribution":p.tolist(),"initial_energies_after_frozen_landscape_preparation":E0.tolist(),
                   "final_energies":E1.tolist(),"tau_relax_initial":tau_relax,
                   "work_landscape_preparation":W_prepare,
                   "H_M_bits_original":H_M,"H_register_bits_regularized":H_reg,
                   "Landauer_perfect_reset_bound_original":W_landauer,
                   "Landauer_perfect_reset_bound_regularized":kT*np.log(2)*H_reg,
                   "equilibrium_deltaF_reset_stage":Feq(E1)-Feq(E0),"runs":results},
 "checks":{"generator_row_sum_error":float(np.max(np.abs(Q0.sum(axis=1)))),
           "detailed_balance_max":float(max(abs(p[i]*Q0[i,j]-p[j]*Q0[j,i]) for i,j in pairs))},
 "limitations":["The landscape-preparation stage assumes the register is kinetically frozen while its energy levels are raised; energetic cost W_prepare is included, but the physical mechanism and cost of barrier switching are not modeled.",
                "Write/read are reversible controlled-XOR logic gates, not finite-time physical gate simulations.",
                "Reset uses an 8-state continuous-time master equation with Arrhenius rates satisfying detailed balance; work is integrated from the time-dependent state energies.",
                "Three unused bit patterns have tiny nonzero regularization probability to permit finite Gibbs energies.",
                "Input-source preparation, detector, controller electronics, and their reset costs are outside this isolated register test."]
}
with open("landauer_3bit_register_dynamic_validation.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
