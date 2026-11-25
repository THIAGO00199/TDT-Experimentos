import numpy as np, json
from scipy.integrate import solve_ivp

# Classical 3-bit CTMC: source spin s, register r, readout ancilla a.
# bit 0 -> spin +1; bit 1 -> spin -1. Source is frozen during the cycle.
beta, kT, nu = 0.4, 2.5, 1.0
states = [(s,r,a) for s in (0,1) for r in (0,1) for a in (0,1)]
N = len(states)
idx = {x:i for i,x in enumerate(states)}
xs = np.array([1-2*x[0] for x in states],float)
xr = np.array([1-2*x[1] for x in states],float)
xa = np.array([1-2*x[2] for x in states],float)
rbit = np.array([x[1] for x in states],float)
abit = np.array([x[2] for x in states],float)
edges = []
for i,x in enumerate(states):
    for bit in (1,2):
        y=list(x); y[bit]=1-y[bit]; j=idx[tuple(y)]
        if i<j: edges.append((i,j,bit))
p0=np.full(N,1/N)

def energy(g,J,er,ea):
    return -g*xs*xr - J*xr*xa + er*rbit + ea*abit

def generator(E,B,allowed_bits):
    Q=np.zeros((N,N))
    for i,j,b in edges:
        if b not in allowed_bits: continue
        # Symmetric transition-state convention: k_ij/k_ji=exp[-beta(Ej-Ei)].
        kij=nu*np.exp(-beta*(B+0.5*(E[j]-E[i])))
        kji=nu*np.exp(-beta*(B+0.5*(E[i]-E[j])))
        Q[i,j]=kij; Q[j,i]=kji
    Q[np.diag_indices(N)]=-Q.sum(axis=1)
    return Q

def entropy(p):
    q=p[p>0]
    return float(-np.sum(q*np.log(q)))

def stage(p, name, duration, start, end, allowed_bits, n_eval=1201):
    # start/end = (g,J,er,ea,B), all controls linear in time.
    d=np.array(end,float)-np.array(start,float)
    xgrid=np.linspace(0,duration,n_eval)
    def rhs(t,z):
        q=z[:N]
        f=t/duration
        pars=np.array(start,float)+f*d
        g,J,er,ea,B=pars
        E=energy(g,J,er,ea)
        Q=generator(E,B,allowed_bits)
        dp=q@Q
        dE_dtheta=np.array([
            -xs*xr, -xr*xa, rbit, abit, np.zeros(N)
        ])
        Edot=(d[0]*dE_dtheta[0]+d[1]*dE_dtheta[1]+d[2]*dE_dtheta[2]+d[3]*dE_dtheta[3])/duration
        power=float(np.dot(q,Edot))
        sigma=0.0
        for i,j,b in edges:
            if b not in allowed_bits: continue
            fij=q[i]*Q[i,j]; fji=q[j]*Q[j,i]
            net=fij-fji
            if abs(net)>0:
                # nonnegative pairwise entropy production rate
                sigma += net*np.log(max(fij,1e-300)/max(fji,1e-300))
        return np.r_[dp,power,sigma]
    sol=solve_ivp(rhs,(0,duration),np.r_[p,0.,0.],method="DOP853",
                  t_eval=[duration],rtol=2e-10,atol=2e-12)
    if not sol.success: raise RuntimeError(sol.message)
    z=sol.y[:,-1]; pf=z[:N]
    return pf, {"stage":name,"duration":duration,"work":float(z[N]),
                "entropy_production_nats":float(z[N+1]),
                "probability_sum_error":float(abs(pf.sum()-1))}

def run():
    p=p0.copy(); ledger=[]
    # Initial controls all zero; random source, register and ancilla.
    # Initial and final Hamiltonians use a high kinetic barrier; the initial uniform bits are prepared as a metastable ensemble.
    p,x=stage(p,"lower_initial_barrier",2.0,(0,0,0,0,40),(0,0,0,0,0.5),{1,2}); ledger.append(x)
    # 1) Dissipative writing: register responds to source spin.
    p,x=stage(p,"write_ramp_g",2.0,(0,0,0,0,0.5),(2,0,0,0,0.5),{1}); ledger.append(x)
    p,x=stage(p,"write_relax",8.0,(2,0,0,0,0.5),(2,0,0,0,0.5),{1}); ledger.append(x)
    p_write=p.copy()
    write_accuracy=float(np.sum(p_write[xs==xr]))
    # 2) Read: ancilla relaxes conditional on register; then ideal observation.
    p,x=stage(p,"read_ramp_J",2.0,(2,0,0,0,0.5),(2,2,0,0,0.5),{2}); ledger.append(x)
    p,x=stage(p,"read_relax",8.0,(2,2,0,0,0.5),(2,2,0,0,0.5),{2}); ledger.append(x)
    read_accuracy=float(np.sum(p[xr==xa]))
    # 3) Raise barrier (kinetic control only), then decouple interactions while frozen.
    p,x=stage(p,"raise_barrier_before_decoupling",2.0,(2,2,0,0,0.5),(2,2,0,0,40),{1,2}); ledger.append(x)
    p,x=stage(p,"decouple_frozen",2.0,(2,2,0,0,40),(0,0,0,0,40),{1,2}); ledger.append(x)
    # Lower barrier at zero interaction; this changes kinetics, not energy.
    p,x=stage(p,"lower_barrier",2.0,(0,0,0,0,40),(0,0,0,0,0.5),{1,2}); ledger.append(x)
    # 4) Reset register and ancilla with energy penalties; then freeze and restore
    # the Hamiltonian while both bits are held at zero.
    p,x=stage(p,"reset_bias_ramp",100.0,(0,0,0,0,0.5),(0,0,30,30,0.5),{1,2}); ledger.append(x)
    p_reset=p.copy()
    p000=float(np.sum(p_reset[(rbit==0)&(abit==0)]))
    p,x=stage(p,"raise_barrier_after_reset",2.0,(0,0,30,30,0.5),(0,0,30,30,40),{1,2}); ledger.append(x)
    p,x=stage(p,"restore_bias_frozen",2.0,(0,0,30,30,40),(0,0,0,0,40),{1,2}); ledger.append(x)

    W=sum(v["work"] for v in ledger)
    Sigma=sum(v["entropy_production_nats"] for v in ledger)
    U0=float(np.dot(p0,energy(0,0,0,0)))
    Uf=float(np.dot(p,energy(0,0,0,0)))
    S0=entropy(p0); Sf=entropy(p)
    F0=U0-kT*S0; Ff=Uf-kT*Sf
    dF=Ff-F0
    balance_residual=beta*(W-dF)-Sigma
    # Exact equilibrium Landauer bound for resetting two initially independent fair bits.
    landauer_two_bits=2*kT*np.log(2)
    return {
      "model":"8-state classical continuous-time Markov chain: source spin, register bit, readout ancilla",
      "parameters":{"beta":beta,"kT":kT,"nu":nu,"initial_distribution":"uniform over source/register/ancilla",
                    "transition_state_rate":"k_ij=nu*exp[-beta*(B+(E_j-E_i)/2)]",
                    "energy":"E=-g*s*x_r-J*x_r*x_a+epsilon_r*r+epsilon_a*a",
                    "source_spin":"held fixed during cycle; initially unbiased"},
      "write_accuracy":write_accuracy,"read_accuracy_before_reset":read_accuracy,
      "P_register_and_ancilla_zero_after_reset_bias":p000,
      "final_register_zero_probability":float(np.sum(p[rbit==0])),
      "final_ancilla_zero_probability":float(np.sum(p[abit==0])),
      "work_total":W,"entropy_production_total_nats":Sigma,
      "entropy_production_energy_equivalent":kT*Sigma,
      "initial_nonequilibrium_free_energy":F0,"final_nonequilibrium_free_energy":Ff,
      "deltaF_nonequilibrium":dF,"beta_work_minus_deltaF":beta*(W-dF),
      "dimensionless_first_second_law_residual":balance_residual,
      "energy_balance_residual":W-dF-kT*Sigma,
      "landauer_two_fair_bits_reference":landauer_two_bits,
      "initial_entropy_bits":S0/np.log(2),"final_entropy_bits":Sf/np.log(2),
      "stages":ledger,
      "limitations":["The source spin is an ideal frozen input, not a thermally reset source.",
        "The ancilla observation is ideal and its detector/controller memory and reset are not modeled.",
        "Barrier changes affect rates only; their energetic work is zero by construction. The final barrier remains high so the reset memory is metastably retained. Entropy production is computed from Markov probability currents.",
        "The symmetric transition-state rate model is a classical phenomenological kinetics, not a fabricated hardware implementation.",
        "The 3-bit five-symbol M_T register is not yet represented; this is a one-spin/one-register-bit/one-ancilla pilot cycle."]}
out=run()
with open("classical_stochastic_full_cycle.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
