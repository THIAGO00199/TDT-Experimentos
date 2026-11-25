p="/home/iknowyou/TDT_Experimentos/tdt_quantum_chain3_control_return.py"
s=open(p).read()
old='res=minimize(objective,c0,method="Powell",bounds=[(-1.5,1.5)]*12,options={"maxiter":90,"xtol":1e-3,"ftol":1e-7,"disp":False})\noptpsi,optw,opte=evolve(res.x)'
new='''rng=np.random.default_rng(20261113)
x=np.zeros(12); best=x.copy(); bestf=objective(x)
for k in range(1,101):
    ak=0.18/(k**0.602); ck=0.12/(k**0.101)
    delta=rng.choice([-1.0,1.0],size=12)
    fp=objective(np.clip(x+ck*delta,-1.5,1.5))
    fm=objective(np.clip(x-ck*delta,-1.5,1.5))
    grad=((fp-fm)/(2*ck))*delta
    x=np.clip(x-ak*grad,-1.5,1.5)
    fx=objective(x)
    if fx<bestf: bestf=fx; best=x.copy()
    if k%20==0: print("SPSA",k,"current",fx,"best",bestf,flush=True)
res=type("Result",(),{"x":best,"success":True,"message":"SPSA completed","nit":100,"fun":bestf})()
optpsi,optw,opte=evolve(res.x)'''
if old not in s: raise SystemExit("target optimizer line not found")
s=s.replace(old,new)
s=s.replace('"optimizer":"Powell deterministic derivative-free; same objective and bounds for all channels; no SPSA"','"optimizer":"SPSA 100 iterations, fixed seed, 12 coefficients clipped to [-1.5,1.5]"')
open(p,"w").write(s)
print("patched SPSA")
