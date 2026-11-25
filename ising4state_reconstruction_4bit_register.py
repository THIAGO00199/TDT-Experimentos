#!/usr/bin/env python3
"""Four-spin Ising reconstruction with 4-bit X0 register and explicit conditional reset."""
import json, time
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

BETA=0.4
KT=1.0/BETA
J=1.0
GAMMA=1.0
T_OBS=1.0
T_CTRL=1.0
NSTEPS=20
HMAX=3.0
NSAMP=30000
SEED=20261113
BARRIER_OFFSET=0.5
RESET_TAUS=[10.0,100.0,1000.0]
FINAL_GAP=30.0
N=4
NS=1<<N
IDS=np.arange(NS,dtype=int)
BITS=((IDS[:,None] >> np.arange(N)) & 1)
SPINS=(2*BITS-1).astype(np.int8)
MAG=SPINS.sum(axis=1)
E0=-J*np.array([sum(SPINS[x,i]*SPINS[x,(i+1)%N] for i in range(N)) for x in range(NS)],dtype=float)
P0=np.exp(-BETA*(E0-E0.min())); P0/=P0.sum()
MVALS=np.array([-4,-2,0,2,4],dtype=int)

def neighbors(x):
    return [x^(1<<i) for i in range(N)]

def rates_for_state(x,h):
    s=SPINS[x].astype(float)
    rr=np.empty(N,dtype=float)
    for i in range(N):
        loc=s[(i-1)%N]+s[(i+1)%N]
        de=2.0*s[i]*(J*loc+h[i])
        rr[i]=GAMMA/(1.0+np.exp(np.clip(BETA*de,-700,700)))
    return rr

def evolve_gillespie(x,h,duration,rng):
    t=0.0
    while True:
        rr=rates_for_state(x,h)
        total=float(rr.sum())
        wait=rng.exponential(1.0/total)
        if t+wait>=duration: return x
        z=rng.random()*total
        i=int(np.searchsorted(np.cumsum(rr),z,side="right"))
        if i>=N: i=N-1
        x ^= (1<<i)
        t += wait

def generator(h):
    Q=np.zeros((NS,NS),dtype=float)
    for x in range(NS):
        rr=rates_for_state(x,h)
        for i in range(N):
            y=x^(1<<i)
            Q[x,y]+=rr[i]
        Q[x,x]=-Q[x].sum()
    return Q

def Hbits(p):
    q=np.asarray(p)
    q=q[q>0]
    return float(-np.sum(q*np.log2(q)))

def build_joint():
    Q=generator(np.zeros(N))
    P=expm(Q*T_OBS)
    P=np.maximum(P,0.0); P/=P.sum(axis=1,keepdims=True)
    joint=np.zeros((NS,len(MVALS)))
    for x in range(NS):
        for y in range(NS):
            j=int(np.where(MVALS==MAG[y])[0][0])
            joint[x,j]+=P0[x]*P[x,y]
    pm=joint.sum(axis=0)
    post=np.divide(joint,pm[None,:],out=np.zeros_like(joint),where=pm[None,:]>0)
    map_ids=np.argmax(post,axis=0)
    HX=Hbits(joint.sum(axis=1))
    HM=Hbits(pm)
    Hcond=sum(pm[j]*Hbits(post[:,j]) for j in range(len(MVALS)) if pm[j]>0)
    I=HX-Hcond
    return Q,P,joint,pm,post,map_ids,HX,HM,Hcond,I

def controller_one(x_start,target,rng):
    x=int(x_start)
    work=0.0
    dt=T_CTRL/NSTEPS
    target_spins=SPINS[int(target)].astype(float)
    for k in range(NSTEPS):
        h_old=target_spins*(HMAX*k/NSTEPS)
        h_new=target_spins*(HMAX*(k+1)/NSTEPS)
        work += -float(np.dot(SPINS[x].astype(float),h_new-h_old))
        x=evolve_gillespie(x,h_new,dt,rng)
    return work,x

def run_reconstruction(map_ids):
    rng_obs=np.random.default_rng(SEED)
    x0s=rng_obs.choice(NS,size=NSAMP,p=P0)
    xts=np.empty(NSAMP,dtype=int)
    ms=np.empty(NSAMP,dtype=int)
    for k,x0 in enumerate(x0s):
        xt=evolve_gillespie(int(x0),np.zeros(N),T_OBS,rng_obs)
        xts[k]=xt; ms[k]=MAG[xt]
    rng_ref=np.random.default_rng(SEED+1)
    rng_obsctrl=np.random.default_rng(SEED+2)
    wref=np.empty(NSAMP); wobs=np.empty(NSAMP)
    fref=np.empty(NSAMP,dtype=bool); fobs=np.empty(NSAMP,dtype=bool)
    for k,(x0,xt,m) in enumerate(zip(x0s,xts,ms)):
        j=int(np.where(MVALS==m)[0][0])
        wr,xr=controller_one(int(xt),int(map_ids[j]),rng_ref)
        wo,xo=controller_one(int(xt),int(x0),rng_obsctrl)
        wref[k]=wr; wobs[k]=wo
        fref[k]=(xr==x0); fobs[k]=(xo==x0)
    delta=wref-wobs
    return {
      "sampling":{"trajectories":NSAMP,"seed":SEED,"same_sampled_X0_XT_M_for_both_controllers":True},
      "reference_M_only":{"mean_actuator_work":float(wref.mean()),"se_work":float(wref.std(ddof=1)/np.sqrt(NSAMP)),"exact_X0_reconstruction_fidelity":float(fref.mean()),"fidelity_se":float(np.sqrt(fref.mean()*(1-fref.mean())/NSAMP))},
      "observed_M_plus_4bit_X0":{"mean_actuator_work":float(wobs.mean()),"se_work":float(wobs.std(ddof=1)/np.sqrt(NSAMP)),"exact_X0_reconstruction_fidelity":float(fobs.mean()),"fidelity_se":float(np.sqrt(fobs.mean()*(1-fobs.mean())/NSAMP))},
      "paired_delta_W_ref_minus_obs":{"mean":float(delta.mean()),"se":float(delta.std(ddof=1)/np.sqrt(NSAMP)),"ci95_normal":[float(delta.mean()-1.96*delta.std(ddof=1)/np.sqrt(NSAMP)),float(delta.mean()+1.96*delta.std(ddof=1)/np.sqrt(NSAMP))]},
      "controller_spec":{"fields":"four independently controlled local fields; open-loop linear ramp aligned to MAP target or true X0","HMAX":HMAX,"control_time":T_CTRL,"steps":NSTEPS,"dynamics":"4-spin periodic Ising ring, single-spin Glauber Gillespie","work_convention":"W=-sum_i integral <s_i> dh_i"}
    }

def reg_generator(E):
    Q=np.zeros((NS,NS),dtype=float)
    for x in range(NS):
        for i in range(N):
            y=x^(1<<i)
            saddle=max(E[x],E[y])+BARRIER_OFFSET
            Q[x,y]=np.exp(-BETA*(saddle-E[x]))
        Q[x,x]=-Q[x].sum()
    return Q

def reset_register(post,pm):
    # Conditional Hamiltonian E0(x|m)=-kT ln p(x|m), so each conditional
    # register ensemble is Gibbs at beta. M is retained as an external side-information label.
    E1=np.full(NS,FINAL_GAP,dtype=float); E1[0]=0.0
    prep_work=0.0
    by_tau={str(tau):{"reset_stage_work":0.0,"total_prep_plus_reset_work":0.0,"weighted_final_entropy_bits":0.0,"weighted_P000":0.0,"weighted_dissipation":0.0} for tau in RESET_TAUS}
    details=[]
    for j,m in enumerate(MVALS):
        p=post[:,j].copy()
        if pm[j]<=0: continue
        Einit=-KT*np.log(np.maximum(p,1e-300))
        Einit-=np.min(Einit)
        wp=float(np.dot(p,Einit))
        prep_work += float(pm[j])*wp
        Q0=reg_generator(Einit)
        # Detailed balance check against p(x|m).
        db=max(abs(p[x]*Q0[x,y]-p[y]*Q0[y,x]) for x in range(NS) for y in range(NS) if x!=y and Q0[x,y]>0)
        for tau in RESET_TAUS:
            dE=E1-Einit
            def rhs(t,z):
                prob=z[:NS]
                u=t/tau
                E=Einit+u*dE
                return np.r_[prob@reg_generator(E),float(np.dot(prob,dE))/tau]
            sol=solve_ivp(rhs,(0,tau),np.r_[p,0.0],method="DOP853",rtol=2e-9,atol=2e-11)
            pf=np.maximum(sol.y[:NS,-1],0); pf/=pf.sum()
            wr=float(sol.y[NS,-1])
            ent=Hbits(pf)
            rec=by_tau[str(tau)]
            rec["reset_stage_work"] += float(pm[j])*wr
            rec["weighted_final_entropy_bits"] += float(pm[j])*ent
            rec["weighted_P000"] += float(pm[j])*pf[0]
            # F_eq(Einit)=0 because sum exp(-beta Einit)=1; final F_eq:
            zf=float(np.sum(np.exp(-BETA*(E1-E1.min()))))
            dF=float(-KT*np.log(zf) - KT*np.log(float(np.max(p)))) # F(E1)-F(Einit), with Einit shifted to min zero
            rec["weighted_dissipation"] += float(pm[j])*(wr-dF)
            details.append({"M_T":int(m),"tau":tau,"P000_final":float(pf[0]),"final_entropy_bits":ent,"reset_stage_work":wr,"deltaF_reset_stage":dF,"detailed_balance_residual":float(db),"solver_success":bool(sol.success)})
    for tau in RESET_TAUS:
        rec=by_tau[str(tau)]
        rec["preparation_work"] = prep_work
        rec["total_prep_plus_reset_work"] = prep_work+rec["reset_stage_work"]
        rec["Landauer_conditional_bound"] = KT*np.log(2)*Hbits(np.array([1.0])) # overwritten below
        rec["Landauer_conditional_bound"] = KT*np.log(2)*sum(pm[j]*Hbits(post[:,j]) for j in range(len(MVALS)))
        rec["Landauer_gap_total_work_minus_bound"] = rec["total_prep_plus_reset_work"]-rec["Landauer_conditional_bound"]
    Hcond=sum(pm[j]*Hbits(post[:,j]) for j in range(len(MVALS)))
    HM=Hbits(pm)
    return {"register_bits":4,"register_states":16,"encoding":"integer state index 0..15 mapped to 4-bit binary; reset target 0000","side_information_during_reset":"M_T retained and used to select conditional energy landscape E(x|M_T)","preparation_work_weighted":prep_work,"H_X0_given_M_T_bits":Hcond,"H_M_T_bits":HM,"runs":by_tau,"M_T_reset_ideal_if_closed_cycle":KT*np.log(2)*HM,"conditional_register_reset_details":details,"caveat":"Conditional reset uses an externally retained M_T label and a side-information-dependent Hamiltonian. Barrier switching is assumed not to cost work; finite-time gate/write/read and detector costs are not modeled."}

def main():
    start=time.time()
    Q,P,joint,pm,post,map_ids,HX,HM,Hcond,I=build_joint()
    recon=run_reconstruction(map_ids)
    reset=reset_register(post,pm)
    kTln2=KT*np.log(2)
    out={
      "title":"4-spin Ising physical reconstruction with 4-bit X0 register",
      "model":{"microstates":16,"register_bits":4,"beta":BETA,"kT":KT,"J":J,"gamma":GAMMA,"observation_time":T_OBS,"magnetization_values":MVALS.tolist(),"prior":"Gibbs equilibrium at zero field","observation_dynamics":"single-spin Glauber CTMC"},
      "exact_information":{"H_X0_bits":HX,"H_M_T_bits":HM,"H_X0_given_M_T_bits":Hcond,"I_X0_M_T_bits":I,"kTln2_times_I":kTln2*I,"kTln2_times_H_X0":kTln2*HX,"kTln2_times_H_cond":kTln2*Hcond,"kTln2_times_H_M":kTln2*HM},
      "reconstruction":recon,
      "four_bit_register_erasure":reset,
      "interpretation":"The register contains X0, so it supplies information beyond M_T. Actuator-work difference is a finite-time control result, not an automatic Landauer prediction. The conditional register reset bound applies only while M_T is retained. If M_T is also reset, add its own reset cost. No residual is evidence of new physics unless all omitted preparation, measurement, gate, controller, and barrier costs are closed.",
      "runtime_seconds":time.time()-start
    }
    out["checks"]={"Q_row_sum_error":float(np.max(np.abs(Q.sum(axis=1)))),"transition_min":float(P.min()),"joint_normalization_error":float(abs(joint.sum()-1)),"H_identity_error":float(abs(HX-Hcond-I))}
    with open("ising4state_reconstruction_4bit_register.json","w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
