"""Original arithmetic controls and explicit rational normalization of frozen gates."""
import json,sys
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moelab.capacity import audit
from moelab.capverify import verify

def build():
    def p(q,c,a=None,v=None,t=None):
        n,e=len(q),len(c)
        return {'probabilities':q,'capacities':c,'eligible':a if a is not None else [list(range(e))for _ in q],
                'expert_values':v if v is not None else [[1]*e for _ in q],'targets':t if t is not None else [0]*n}
    f={
      'order_loss':p([['3/5','2/5'],['99/100','1/100']],[1,1]),
      'confidence_loss':p([['1/2','49/100','1/100'],['49/100','1/100','1/2']],[1,1,0]),
      'eligibility_trap':p([['9/10','1/10'],['9/10','1/10']],[1,1],[[0,1],[0]]),
      'coverage_before_mass':p([['9/10','1/10'],['1/10','9/10']],[1,1],[[0,1],[0]]),
      'quality_not_mass':p([['3/5','2/5'],['99/100','1/100']],[1,1],v=[[1,100],[1,100]],t=['3/5',1]),
      'tied_quality':p([['1/2','1/2']]*2,[1,1],v=[[0,10],[10,0]]),
      'tied_family':p([['1/2','1/2']]*3,[1,1]),
      'zero_capacity':p([['1/2','1/2']]*2,[0,0]),
      'unused_capacity':p([['9/10','1/10']]*2,[1,2],[[0],[0]]),
      'zero_gate_dispatch':p([[1,0],[1,0]],[1,1]),
      'empty_eligibility':p([['1/2','1/2'],['1/3','2/3']],[2,2],[[],[0,1]]),
    }
    bridge=json.loads((ROOT/'moelab/route_data/capacity-bridge.json').read_text())
    for name,rows in bridge['models'].items():
        q=[];v=[];t=[]
        for row in rows:
            gate=list(map(F,row['probabilities']));den=sum(gate)
            q.append([str(x/den)for x in gate]);v.append([str(F(x))for x in row['expert_values']]);t.append(str(F(row['target'])))
        f['frozen_'+name]=p(q,[2]*4,v=v,t=t)
    return [{'id':name,'input':payload,'report':audit(payload)}for name,payload in f.items()]

if __name__=='__main__':
    fixtures=build()
    for item in fixtures:verify(item['input'],item['report'])
    target=ROOT/'moelab/route_data/capacity-fixtures.json';target.write_text(json.dumps(fixtures,separators=(',',':'),sort_keys=True)+'\n')
    (ROOT/'docs/capacity-data.js').write_text('globalThis.CAPACITY_FIXTURES='+json.dumps(fixtures,separators=(',',':'),sort_keys=True)+';\n')
    for f in fixtures:
        r=f['report'];print(f['id'],r['optimum']['dispatched'],r['optimum']['retained_probability'],r['optimum']['count'],r['optimum']['canonical']['update_mse'],len(r['states']))
