#!/usr/bin/env python3
"""First controlled Ising 4x4 benchmarks: global-field erasure and MAP inference."""
import json, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

L=4; N=16; NS=1<<N; J=1.; beta=.4; gamma=1.; T=10.
ids=np.arange(NS,dtype=np.uint32)
bits=(ids[:,None] >> np.arange(N,dtype=np.uint32)) & 1
s=(2*bits.astype(np.int8)-1).astype(np.int8)
M=s.sum(1).astype(float)
bonds=[]
for y in range(L):
 for x in range(L):
  i=y*L+x
  bonds.extend([(i,y*L+(x+1)%L),(i,((y+1)%L)*L+x)])
bs=np.zeros(NS,dtype=np.int16)
for i,j in bonds: bs += s[:,i]*s[:,j]
E0=-J*bs
p0=np.exp(-beta*(E0-E0.min())); p0/=p0.sum()
flip_ids=[ids ^ np.uint32(1<<i) for i in range(N)]
neighbors=[]
for i in range(N):
 y,x=divmod(i,L)
 neighbors.append([y*L+(x-1)%L,y*L+(x+1)%L,((y-1)%L)*L+x,((y+1)%L)*L+x])

def generator(h):
 rows=[]; cols=[]; vals=[]
 for i in range(N):
  loc=s[:,neighbors[i]].sum(1)
  dE=2*s[:,i]*(J*loc+h)
  rate=gamma/(1+np.exp(np.clip(beta*dE,-700,700)))
  rows.append(ids.astype(np.int32)); cols.append(flip_ids[i].astype(np.int32)); vals.append(rate)
 r=np.concatenate(rows); c=np.concatenate(cols); v=np.concatenate(vals)
 ex=np.bincount(r,weights=v,minlength=NS)
 return coo_matrix((np.r_[v,-ex],(np.r_[r,np.arange(NS)],np.r_[c,np.arange(NS)])),shape=(NS,NS)).tocsr()

def entropy(p):
 q=p[p>0]
 return float(-np.sum(q*np.log2(q)))

def erasure():
 # 20 isothermal Glauber relaxations, followed by field jumps 0 -> 5.
 nstep=50; dt=T/nstep; hs=np.linspace(0.,5.,nstep+1)
 p=p0.copy(); work=0.; work_parts=[]
 for k in range(nstep):
  Q=generator(hs[k])
  p=expm_multiply(Q.T*dt,p)
  p=np.maximum(p,0); p/=p.sum()
  dW=-float(np.dot(p,M))*(hs[k+1]-hs[k])
  work+=dW; work_parts.append(dW)
 # Final distribution is immediately after last field jump; report probability all-up.
 allup=NS-1
 return {"protocol":"20-step linear ramp h: 0 -> 5, with Glauber evolution at each pre-jump field",
  "T":T,"h_final":5.,"work_mean_J_units":work,
  "H_initial_bits":entropy(p0),"H_final_bits":entropy(p),
  "entropy_reduction_bits":entropy(p0)-entropy(p),
  "P_all_up_final":float(p[allup]),"P_not_all_up_final":float(1-p[allup]),
  "mean_M_final":float(np.dot(p,M)),
  "work_per_step":work_parts,
  "warning":"This is a fixed benchmark protocol, not an optimized minimum-work solution. The field jump work uses the pre-jump magnetization."}

def reconstruction_inference():
 # Uncontrolled reference channel at h=0; calculate P(M_T|X0) exactly via backward indicators.
 Q=generator(0.)
 mvals=np.arange(-16,17,2)
 cond=np.empty((NS,len(mvals)))
 for k,m in enumerate(mvals):
  cond[:,k]=expm_multiply(Q*T,(M==m).astype(float))
 cond=np.clip(cond,0,1)
 joint=p0[:,None]*cond
 pm=joint.sum(0)
 posterior=np.divide(joint,pm[None,:],out=np.zeros_like(joint),where=pm[None,:]>0)
 map_success=float(np.sum(np.max(joint,axis=0)))
 # Bayes posterior entropy
 hcond=0.
 for k in range(len(mvals)):
  pk=posterior[:,k]; z=pk>0
  hcond += pm[k]*(-np.sum(pk[z]*np.log2(pk[z])))
 return {"reference_channel":"Uncontrolled Glauber h=0 for T=1; observation M_T",
  "prior":"Gibbs beta=0.4, J=1, h_initial=0",
  "H_X0_given_M_T_bits":float(hcond),
  "best_possible_exact_guess_probability_without_auxiliary_info":map_success,
  "exact_guess_probability_with_X0_disclosed_to_controller":1.0,
  "information_advantage_bits":float(hcond),
  "warning":"This is an inference benchmark, not physical reconstruction work. Physical B requires specifying the post-observation starting state and a local-field preparation protocol."}

if __name__=="__main__":
 t=time.time()
 out={"model":{"L":L,"N":N,"states":NS,"boundary":"periodic","J":J,"beta":beta,"gamma":gamma},
      "erasure_global_field":erasure(),
      "reconstruction_information_benchmark":reconstruction_inference(),
      "runtime_seconds":None}
 out["runtime_seconds"]=time.time()-t
 with open("ising4x4_first_controls.json","w") as f: json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
