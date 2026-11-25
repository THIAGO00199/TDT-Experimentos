src='/home/iknowyou/TDT_Experimentos/tdt_v5_work_and_markov_reference.py'
raw=open(src).read()
prefix=raw.split('# Original optimization grid dt=.1 and identical RNG stream/algorithm')[0]
defs=raw.split('# Original optimization grid dt=.1 and identical RNG stream/algorithm')[1].split('def evolve_entropy')[0]
ns={}
exec(prefix+ '# Original optimization grid dt=.1 and identical RNG stream/algorithm'+defs,ns)
import json
r=json.load(open('/home/iknowyou/TDT_Experimentos/tdt_v5_work_and_markov_reference_result.json'))
c=ns['np'].array(r['nonmarkovian']['control_coefficients'])
nt=18000; us=ns['controls'](c,nt)
x,W,dE,res=ns['run_closed'](nt,us)
out={'NT':nt,'dt':ns['T']/nt,'W_pre_quench':W,'delta_E_total':dE,'residual':res,'purity':ns['pur'](ns['rq'](x)),'entropy_nats':ns['ent'](ns['rq'](x))}
json.dump(out,open('/home/iknowyou/TDT_Experimentos/tdt_v5_dt_1e-3_result.json','w'),indent=2)
print(json.dumps(out,indent=2))
