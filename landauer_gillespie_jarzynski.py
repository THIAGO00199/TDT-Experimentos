import numpy as np
from numba import njit
import json, time, math

BETA=0.4
KT=1.0/BETA
NU=1.0
EPS_F=15.0
B=20.0
N=200_000
SEED=20261113

@njit
def simulate_one(rng_seed, tau, B, ntraj):
    np.random.seed(rng_seed)
    beta=0.4; nu=1.0; epsf=15.0
    k01=nu*np.exp(-beta*B)
    c=beta*epsf/tau
    epsdot=epsf/tau
    works=np.empty(ntraj)
    finals=np.empty(ntraj, dtype=np.int8)
    for j in range(ntraj):
        # Initially canonical equilibrium at epsilon=0: p(r=0)=p(r=1)=1/2.
        state=0 if np.random.random()<0.5 else 1
        t=0.0
        w=0.0
        while True:
            u=np.random.random()
            z=-math.log(u)
            if state==0:
                dt=z/k01
            else:
                # k10(t+s)=nu*exp[-beta*(B-epsdot*(t+s))]
                rate_now=nu*math.exp(-beta*(B-epsdot*t))
                if c < 1e-14:
                    dt=z/rate_now
                else:
                    dt=math.log1p(c*z/rate_now)/c
            if t+dt >= tau:
                if state==1:
                    w += epsdot*(tau-t)
                t=tau
                break
            if state==1:
                w += epsdot*dt
            t += dt
            state=1-state
        works[j]=w
        finals[j]=state
    return works, finals

def summary(works, finals, tau, label):
    beta=BETA
    weights=np.exp(-beta*works)
    meanw=float(np.mean(works))
    sew=float(np.std(works,ddof=1)/np.sqrt(len(works)))
    je=float(np.mean(weights))
    seje=float(np.std(weights,ddof=1)/np.sqrt(len(weights)))
    ess=float(np.sum(weights)**2/np.sum(weights**2))
    dF= -KT*np.log(1+np.exp(-beta*EPS_F)) + KT*np.log(2.0)
    target=float(np.exp(-beta*dF))
    return {
        "protocol":label,"N":len(works),"tau":tau,
        "mean_work":meanw,"mean_work_SE":sew,
        "jarzynski_mean_exp_minus_betaW":je,
        "jarzynski_SE_naive":seje,
        "jarzynski_target_exp_minus_betaDeltaF":target,
        "jarzynski_z_score_naive":(je-target)/seje,
        "relative_jarzynski_error_percent":100*(je-target)/target,
        "weight_ESS":ess,"weight_ESS_fraction":ess/len(works),
        "final_p1":float(np.mean(finals)),
        "work_min":float(np.min(works)),"work_max":float(np.max(works)),
        "work_std":float(np.std(works,ddof=1)),
        "DeltaF":dF
    }

if __name__=="__main__":
    tau_relax=1.0/(2.0*NU*np.exp(-BETA*B))
    cases=[("FAST_eta_0.01",0.01*tau_relax),("SLOW_eta_1000",1000.0*tau_relax)]
    results=[]
    t0=time.time()
    for i,(label,tau) in enumerate(cases):
        a=time.time()
        w,f=simulate_one(SEED+i,int(tau*0+0)+tau,B,N)
        res=summary(w,f,tau,label)
        res["runtime_s"]=time.time()-a
        results.append(res)
        print(json.dumps(res,indent=2),flush=True)
    out={"parameters":{"beta":BETA,"kT":KT,"nu":NU,"B":B,"epsilon_final":EPS_F,
                       "N_per_protocol":N,"seed":SEED,"tau_relax_initial":tau_relax},
         "results":results,"total_runtime_s":time.time()-t0}
    with open("landauer_gillespie_jarzynski.json","w") as f:
        json.dump(out,f,indent=2)
    print("SAVED landauer_gillespie_jarzynski.json",flush=True)
