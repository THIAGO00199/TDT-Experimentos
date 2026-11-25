import numpy as np, json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
N=8; beta=.4; T=1/beta; J=1.; gamma=1.; n=1<<N
states=np.array([[1 if (x>>i)&1 else -1 for i in range(N)] for x in range(n)],dtype=np.int8)
M=states.sum(1); bond=np.sum(states*np.roll(states,-1,axis=1),axis=1); Ebase=-J*bond
# unique undirected edges: flip bit i from 0 to 1
xs=[]; ys=[]; ds=[]
for x in range(n):
 for i in range(N):
  if ((x>>i)&1)==0:
   y=x^(1<<i)
   de=2*states[x,i]*(J*(states[x,(i-1)%N]+states[x,(i+1)%N]))
   xs.append(x); ys.append(y); ds.append(de)
xs=np.array(xs); ys=np.array(ys); ds=np.array(ds)
# all directed transitions indexed by source and flipped spin
src=np.repeat(np.arange(n),N); bit=np.tile(np.arange(N),n); dst=src^(1<<bit)
de0=2*states[src,bit]*J*(states[src,(bit-1)%N]+states[src,(bit+1)%N])
def Qmake(h):
 de=de0+2*states[src,bit]*h
 rates=gamma/(1+np.exp(np.clip(beta*de,-700,700)))
 diag=-np.bincount(src,weights=rates,minlength=n)
 return csr_matrix((np.r_[rates,diag],(np.r_[src,np.arange(n)],np.r_[dst,np.arange(n)])),shape=(n,n))
def gibbs(h):
 e=Ebase-h*M; z=logsumexp(-beta*e); return np.exp(-beta*e-z)
def entropy(p):
 pp=np.maximum(p,1e-300); return -float(np.sum(p*np.log(pp)))
def sigma_dot(p,h):
 # Schnakenberg sum over each undirected edge exactly once
 qf=gamma/(1+np.exp(np.clip(beta*(ds+2*states[xs,0]*0),-700,700))) # overwritten below with actual field contribution
 # recompute deltaE for each edge using spin index inferred from x,y
 inds=np.log2(xs^ys).astype(int)
 de=ds+2*states[xs,inds]*h
 kxy=gamma/(1+np.exp(np.clip(beta*de,-700,700)))
 kyx=gamma/(1+np.exp(np.clip(-beta*de,-700,700)))
 a=p[xs]*kxy; b=p[ys]*kyx
 return float(np.sum((a-b)*np.log(np.maximum(a,1e-300)/np.maximum(b,1e-300))))
p0=gibbs(0.)
h0=1.
p_quench=p0.copy()
Q0=Qmake(h0)
relax_times=[.05,.1,.2,.5,1.,2.,4.,8.,16.,32.]
ctrl=10.
results=[]
for tr in relax_times:
 def relax_rhs(t,y):
  p=y[:n]; return np.r_[Q0.T.dot(p),sigma_dot(p,h0)]
 y0=np.r_[p_quench,0.]
 solr=solve_ivp(relax_rhs,(0,tr),y0,method="DOP853",rtol=2e-10,atol=2e-12)
 pr=solr.y[:n,-1]; Sigma=float(solr.y[n,-1])
 # independent endpoint check: Delta S_system + Delta S_bath, Qbath=-Delta E_system at fixed h
 dS=entropy(pr)-entropy(p_quench)
 E=lambda p,h: float(np.dot(p,Ebase-h*M))
 dEb=E(pr,h0)-E(p_quench,h0)
 dSbath=-beta*dEb
 Sigma_endpoint=dS+dSbath
 # recovery ramp h:1->0 over ctrl, finite-time; work = integral -<M> dh
 def rec_rhs(t,y):
  h=h0*(1-t/ctrl); return Qmake(h).T.dot(y)
 sol=solve_ivp(rec_rhs,(0,ctrl),pr,method="DOP853",rtol=2e-9,atol=2e-11,dense_output=True)
 ts=np.linspace(0,ctrl,1001); ps=sol.sol(ts)
 mags=M@ps
 Wrec=float(np.trapezoid(-mags*(-h0/ctrl),ts))
 pf=ps[:,-1]
 KL=float(np.sum(pf*np.log(np.maximum(pf,1e-300)/p0)))
 F=lambda p,h: E(p,h)-T*entropy(p)
 dF=F(pf,0)-F(pr,h0)
 results.append(dict(t_relax=tr,Sigma_Schnakenberg=Sigma,Sigma_endpoint_check=Sigma_endpoint,
  Sigma_check_error=Sigma-Sigma_endpoint,deltaS_system=dS,deltaS_bath=dSbath,
  Wrec_system=Wrec,deltaF_recovery=dF,DeltaF_plus_TSigma=dF+T*Sigma,
  final_KL_to_initial=KL,final_magnetization=float(M@pf),
  KL_to_quench_equilibrium=float(np.sum(pr*np.log(np.maximum(pr,1e-300)/gibbs(h0))))))
out={"model":"8-spin periodic Ising ring; 256 states; single-spin Glauber bath",
"parameters":{"beta":beta,"T":T,"J":J,"gamma":gamma,"quench_field":h0,"recovery_duration":ctrl},
"protocol":"initial Gibbs at h=0; instantaneous quench to h=1; relax at fixed h=1; recover with finite-time linear field ramp h=1->0",
"units":"kB=1; entropy nats; energy/work in J units",
"interpretation":"Sigma is total Schnakenberg entropy production during forward relaxation. Wrec is finite-time system-only recovery work. DeltaF+T*Sigma is reported as the proposed TDT quantity, not assumed equal to Wrec and not a computed full bath-reset cost.",
"results":results}
path="/home/iknowyou/TDT_Experimentos/tdt_v4_ising8_testB_pilot"
with open(path+".json","w") as f: json.dump(out,f,indent=2)
x=np.array(relax_times)
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(x,[r["Sigma_Schnakenberg"] for r in results],"o-",label="Σ Schnakenberg (relaxação)")
ax.plot(x,[r["Wrec_system"] for r in results],"s-",label="Wrec do sistema (rampa)")
ax.plot(x,[r["DeltaF_plus_TSigma"] for r in results],"^-",label="ΔF + TΣ (quantidade proposta)")
ax.set_xscale("log"); ax.set_xlabel("Tempo de relaxação"); ax.set_ylabel("Energia / entropia (kB=1)")
ax.set_title("TDT v4 — Ising 8 spins: piloto")
ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path+".png",dpi=180)
print(json.dumps(out,indent=2))
