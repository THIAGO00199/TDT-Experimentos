#!/usr/bin/env python3
"""TDT v3 / Ising 4x4: exact state enumeration + Glauber baseline.

All 65,536 configurations are enumerated. The continuous-time Markov
semigroup is evaluated numerically with scipy.sparse.linalg.expm_multiply.
This is an infrastructure/validation run, not yet an optimized protocol.
"""
import json, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

L=4
N=L*L
NS=1<<N
J=1.0
BETA=0.4
GAMMA=1.0
T=1.0
H=0.0
OUT="ising4x4_baseline.json"

def main():
    t0=time.time()
    # Bit i represents spin i: 0 -> -1, 1 -> +1.
    ids=np.arange(NS,dtype=np.uint32)
    bits=((ids[:,None] >> np.arange(N,dtype=np.uint32)) & 1)
    spins=(2*bits.astype(np.int8)-1).astype(np.int8)
    mag=spins.sum(axis=1).astype(np.int16)

    # Periodic square lattice; each undirected bond counted once (right/down).
    bonds=[]
    for y in range(L):
        for x in range(L):
            i=y*L+x
            bonds.append((i,y*L+(x+1)%L))
            bonds.append((i,((y+1)%L)*L+x))
    bond_sum=np.zeros(NS,dtype=np.int16)
    for i,j in bonds:
        bond_sum += spins[:,i]*spins[:,j]
    energy=(-J*bond_sum-H*mag).astype(np.float64)

    # Single-spin-flip Glauber generator Q (row convention).
    rows=[]; cols=[]; vals=[]
    for i in range(N):
        neigh=[]
        y,x=divmod(i,L)
        neigh=[y*L+(x-1)%L,y*L+(x+1)%L,
               ((y-1)%L)*L+x,((y+1)%L)*L+x]
        local=spins[:,neigh].sum(axis=1)
        dE=2.0*spins[:,i]*(J*local+H)
        # Stable logistic form.
        z=np.clip(BETA*dE,-700,700)
        rate=GAMMA/(1.0+np.exp(z))
        flipped=ids ^ np.uint32(1<<i)
        rows.append(ids.astype(np.int32))
        cols.append(flipped.astype(np.int32))
        vals.append(rate)
    r=np.concatenate(rows); c=np.concatenate(cols); v=np.concatenate(vals)
    exit_rate=np.bincount(r,weights=v,minlength=NS)
    Q=coo_matrix((np.r_[v,-exit_rate],(np.r_[r,np.arange(NS)],np.r_[c,np.arange(NS)])),
                 shape=(NS,NS)).tocsr()
    Q.sum_duplicates()

    # Validate generator and detailed balance against Gibbs distribution.
    row_sum=np.asarray(Q.sum(axis=1)).ravel()
    p_gibbs=np.exp(-BETA*(energy-energy.min()))
    p_gibbs/=p_gibbs.sum()
    # For each directed edge, compare pi_i q_ij and pi_j q_ji.
    coo=Q.tocoo()
    off=coo.row!=coo.col
    rr=coo.row[off]; cc=coo.col[off]; qq=coo.data[off]
    rev=Q[cc,rr].A1
    flux_f=p_gibbs[rr]*qq
    flux_b=p_gibbs[cc]*rev
    db_rel=np.max(np.abs(flux_f-flux_b))/max(np.max(flux_f),1e-300)

    # Conditional entropy H(X0 | M_T) for two explicitly different priors.
    # Backward equation: P(M_T=m | X0=x) = [exp(Q T) 1_{M=m}]_x.
    mvals=np.arange(-N,N+1,2,dtype=np.int16)
    masks=[(mag==m).astype(np.float64) for m in mvals]
    cond=np.empty((NS,len(mvals)),dtype=np.float64)
    for k,indicator in enumerate(masks):
        cond[:,k]=expm_multiply(Q*T,indicator)
    cond=np.clip(cond,0.0,1.0)
    cond_sum=cond.sum(axis=1)
    partition_error=float(np.max(np.abs(cond_sum-1.0)))

    priors={"uniform":np.full(NS,1.0/NS),"gibbs_beta_0p4":p_gibbs}
    results={}
    for name,p0 in priors.items():
        joint=p0[:,None]*cond
        pm=joint.sum(axis=0)
        nz=joint>0
        h_joint=-np.sum(joint[nz]*np.log2(joint[nz]))
        nzm=pm>0
        h_m=-np.sum(pm[nzm]*np.log2(pm[nzm]))
        h_x=-np.sum(p0[p0>0]*np.log2(p0[p0>0]))
        results[name]={
            "H_X0_bits":float(h_x),
            "H_M_T_bits":float(h_m),
            "H_X0_given_M_T_bits":float(h_joint-h_m),
            "mutual_information_bits":float(h_x-(h_joint-h_m)),
            "P_M_T":{str(int(m)):float(prob) for m,prob in zip(mvals,pm)}
        }

    output={
      "model":{"L":L,"N_spins":N,"microstates":NS,"boundary":"periodic",
       "J":J,"beta":BETA,"gamma_per_spin":GAMMA,"field_h":H,"time":T,
       "dynamics":"continuous-time single-spin-flip Glauber",
       "energy":"E=-J sum_(right/down bonds) s_i s_j - h M"},
      "numerics":{"generator_shape":[NS,NS],"generator_nnz":int(Q.nnz),
       "max_abs_generator_row_sum":float(np.max(np.abs(row_sum))),
       "max_relative_detailed_balance_flux_error":float(db_rel),
       "max_conditional_probability_partition_error":partition_error,
       "elapsed_seconds":time.time()-t0,
       "propagation":"scipy expm_multiply; numerical Krylov action of matrix exponential"},
      "priors":results,
      "interpretation":"Baseline uncontrolled dynamics only. Conditional entropy is uncertainty about X0 after observing M_T; it is not automatically an operational erasure cost or a reconstruction work."
    }
    with open(OUT,"w") as f: json.dump(output,f,indent=2)
    print(json.dumps({k:v for k,v in output.items() if k!="priors"},indent=2))
    for name,res in results.items():
        print(name, "H(X0)=",res["H_X0_bits"],
              "H(X0|M_T)=",res["H_X0_given_M_T_bits"],
              "I=",res["mutual_information_bits"])
    print("Saved",OUT)

if __name__=="__main__": main()
