import json, numpy as np
from scipy.linalg import expm
p="/home/iknowyou/TDT_Experimentos/ising4state_full_80control_optimization.json"
o=json.load(open(p))
B=.4; KT=2.5; J=1.; T=4.; dt=T/20
ids=np.arange(16); bits=((ids[:,None]>>np.arange(4))&1); S=(2*bits-1).astype(float); mag=S.sum(1)
E=np.array([-J*sum(S[x,i]*S[x,(i+1)%4] for i in range(4)) for x in range(16)])
p0=np.exp(-B*(E-E.min())); p0/=p0.sum()
def qmat(h):
 Q=np.zeros((16,16))
 for x in range(16):
  for i in range(4):
   loc=S[x,(i-1)%4]+S[x,(i+1)%4]; de=2*S[x,i]*(J*loc+h[i])
   Q[x,x^(1<<i)]+=1/(1+np.exp(B*de))
  Q[x,x]=-Q[x].sum()
 return Q
joint=p0[:,None]*expm(qmat(np.zeros(4))*1.)
Ms=np.array([-4,-2,0,2,4]); pm=np.array([joint[:,mag==m].sum() for m in Ms])
post=np.array([joint[:,mag==m].sum(1)/pm[j] for j,m in enumerate(Ms)]); maps=np.argmax(post,axis=1)
def grp(pol):
 if pol=="reference": return maps,pm,np.array([joint[:,mag==m].sum(1)/pm[j] for j,m in enumerate(Ms)])
 return np.arange(16),p0,np.array([joint[x]/p0[x] for x in range(16)])
out={}
for pol in ["reference","informed"]:
 tg,w,ps=grp(pol); A=np.array(o[pol]["best"]["controls"])
 W=F0=Ff=fid=0.
 for g,target in enumerate(tg):
  p=ps[g].copy()
  f0=np.dot(p,E)+KT*np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0))
  ww=0.; prev=np.zeros(4)
  for k in range(20):
   h=A[k]*S[target]
   ww-=np.dot(p@S,h-prev*S[target])
   p=p@expm(qmat(h)*dt)
   prev=A[k]
  ww+=np.dot(p@S,prev*S[target])
  ff=np.dot(p,E)+KT*np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0))
  W+=w[g]*ww; F0+=w[g]*f0; Ff+=w[g]*ff; fid+=w[g]*p[target]
 out[pol]={"fidelity_exact_expm":fid,"actuator_work_exact_expm":W,"deltaF_exact_expm":Ff-F0,"dissipated_work_exact_expm":W-(Ff-F0)}
print(json.dumps(out,indent=2))
json.dump(out,open("/home/iknowyou/TDT_Experimentos/ising4state_full_80control_validation_expm.json","w"),indent=2)
