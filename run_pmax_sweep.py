from pathlib import Path
p=Path('/home/iknowyou/TDT_Experimentos/tdt_v5_pmax_and_work_target085.py')
src=p.read_text()
prefix=src.split('# Work-minimization under P>=.85')[0]
prefix=prefix.replace('for seed in range(8):','for seed in range(6):').replace('20261113+seed,140,.18','20261114+seed,100,.18')
prefix=prefix.replace('np.zeros(20)','np.zeros(4*K)').replace('normal(0,.35,20)','normal(0,.35,4*K)')
out=Path('/home/iknowyou/TDT_Experimentos')
for knots in (5,9,17):
    code=prefix.replace('K=5; BOUND=1.5',f'K={knots}; BOUND=1.5')
    code += """
result={'K':K,'T':T,'NT':NT,'dt':dt,'control_bound':BOUND,'seeds':pmax_records,'best_found':best_pmax,'certified_upper_bound':False}
with open(OUT+f'/tdt_v5_pmax_sweep_K{K}.json','w') as f: json.dump(result,f,indent=2)
print('SWEEP_RESULT',json.dumps(result),flush=True)
"""
    exec(compile(code, f'<sweep_K{knots}>','exec'),{})
