import copy
import inspect
import itertools
import json
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from moelab.capacity import audit,validate
from moelab.capverify import verify,reference
ROOT=Path(__file__).resolve().parents[1]
FIXTURES=json.loads((ROOT/'moelab/route_data/capacity-fixtures.json').read_text())


def payload(q,c,a=None):
    return {'probabilities':q,'capacities':c,'eligible':a if a is not None else [list(range(len(c)))for _ in q],
            'expert_values':[[1]*len(c)for _ in q],'targets':[0]*len(q)}


class Capacity(unittest.TestCase):
    def test_all_fourteen_complete_controls(self):
        self.assertEqual(len(FIXTURES),14)
        for item in FIXTURES:
            self.assertEqual(audit(item['input']),item['report']);self.assertTrue(verify(item['input'],item['report'])['verified'])

    def test_reference_is_independent(self):
        import moelab.capverify as module
        source=inspect.getsource(module)
        self.assertNotIn('from .capacity',source);self.assertNotIn('import capacity',source)
        self.assertIn('itertools.product',source)

    def test_225_quarter_grid_pairs(self):
        # All 15 ternary quarter-grid probabilities, including exact zeros.
        rows=[[str(F(i,4)),str(F(j,4)),str(F(4-i-j,4))]for i in range(5)for j in range(5-i)]
        self.assertEqual(len(rows),15)
        for q in itertools.product(rows,repeat=2):
            p=payload(list(q),[1,1,1]);self.assertEqual(audit(p),reference(p))

    def test_1024_eligibility_capacity_controls(self):
        possible=[[],[0],[1],[0,1]];rows=[['3/4','1/4'],['1/4','3/4'],['1/2','1/2']];total=0
        for eligible in itertools.product(possible,repeat=3):
            for c in itertools.product(range(4),repeat=2):
                p=payload(rows,list(c),list(eligible));r=audit(p);self.assertEqual(r,reference(p));self.assertEqual(r['minimum_drops'],3-r['optimum']['dispatched']);total+=1
        self.assertEqual(total,1024)

    def test_bound_and_tiny_probability(self):
        tiny=F(1,2**128);row=[str(tiny),str(tiny),str(tiny),str(1-3*tiny)]
        p=payload([row]*7,[2,2,2,2]);r=audit(p);self.assertEqual(r,reference(p));self.assertEqual(r['optimum']['dispatched'],7)
        self.assertEqual(sum(map(int,r['optimum']['marginal_counts'][0])),int(r['optimum']['count']))

    def test_capacity_and_batch_coupling(self):
        p=payload([['9/10','1/10'],['1/10','9/10']],[1,0],[[0,1],[0]])
        low=audit(p);p['capacities']=[1,1];high=audit(p)
        self.assertEqual(low['optimum']['dispatched'],1);self.assertEqual(high['optimum']['dispatched'],2)
        self.assertGreater(F(low['optimum']['retained_probability']),F(high['optimum']['retained_probability']))
        p=FIXTURES[0]['input'];batch=audit(p);alone={k:v[:1] if k!='capacities' else v for k,v in p.items()}
        self.assertEqual(audit(alone)['optimum']['canonical']['assignment'],[0]);self.assertEqual(batch['optimum']['canonical']['assignment'],[1,0])

    def test_known_mechanism_controls(self):
        rs={i['id']:i['report']for i in FIXTURES}
        r=rs['order_loss'];self.assertEqual(r['optimum']['retained_probability'],'139/100');self.assertEqual(r['baselines']['ordered_best_available']['retained_probability'],'61/100')
        self.assertEqual(r['baselines']['confidence_best_available'],r['optimum']['canonical'])
        r=rs['confidence_loss'];self.assertEqual(r['optimum']['retained_probability'],'49/50');self.assertEqual(r['baselines']['confidence_best_available']['retained_probability'],'51/100')
        r=rs['eligibility_trap'];self.assertEqual(r['optimum']['dispatched'],2);self.assertEqual(r['baselines']['confidence_best_available']['dispatched'],1)
        r=rs['coverage_before_mass'];self.assertEqual(r['optimum']['retained_probability'],'1/5');self.assertEqual(r['baselines']['ordered_best_available']['retained_probability'],'9/10')
        r=rs['quality_not_mass'];self.assertGreater(F(r['optimum']['canonical']['update_mse']),F(r['baselines']['ordered_best_available']['update_mse']))
        r=rs['unused_capacity'];self.assertEqual(r['minimum_drops'],1);self.assertEqual(sum(r['input']['capacities']),3)
        r=rs['zero_gate_dispatch'];self.assertEqual(r['optimum']['count'],'2');self.assertEqual(r['optimum']['canonical']['weighted_updates'],['1','0'])

    def test_tied_task_quality_range(self):
        r=next(i['report']for i in FIXTURES if i['id']=='tied_quality')
        self.assertEqual(r['optimum']['update_mse_range']['minimum'],'0');self.assertEqual(r['optimum']['update_mse_range']['maximum'],'25')

    def test_whole_tied_family(self):
        r=next(i['report']for i in FIXTURES if i['id']=='tied_family');self.assertEqual(r['optimum']['count'],'6')
        self.assertEqual(r['optimum']['marginal_counts'],[['2','2','2']]*3)
        self.assertEqual(r['optimum']['canonical']['assignment'],[0,1,-1])

    def test_permutation_equivariance(self):
        rng=random.Random(17)
        for _ in range(30):
            q=[[str(F(x,10))for x in row]for row in [[6,3,1],[2,5,3],[3,3,4],[0,5,5]]]
            p=payload(q,[1,2,1]);r=audit(p);tokens=list(range(4));experts=list(range(3));rng.shuffle(tokens);rng.shuffle(experts)
            pp=payload([[q[i][j]for j in experts]for i in tokens],[p['capacities'][j]for j in experts]);rr=audit(pp)
            for k in ['dispatched','retained_probability','count']:self.assertEqual(r['optimum'][k],rr['optimum'][k])
            for ii,i in enumerate(tokens):
                for jj,j in enumerate(experts):self.assertEqual(r['optimum']['marginal_counts'][i][j],rr['optimum']['marginal_counts'][ii][jj])

    def test_all_leaf_tampering_and_shape(self):
        p=FIXTURES[0]['input'];r=audit(p);paths=[]
        def walk(x,path):
            if isinstance(x,dict):
                for k,v in x.items():walk(v,path+[k])
            elif isinstance(x,list):
                for k,v in enumerate(x):walk(v,path+[k])
            else:paths.append(path)
        walk(r,[])
        for path in paths:
            z=copy.deepcopy(r);x=z
            for k in path[:-1]:x=x[k]
            old=x[path[-1]];x[path[-1]]=not old if type(old)is bool else old+1 if type(old)is int else str(old)+'!'
            with self.assertRaises(ValueError):verify(p,z)
        self.assertGreater(len(paths),150)
        for z in [dict(r,unknown=1),{k:v for k,v in r.items()if k!='states'}]:
            with self.assertRaises(ValueError):verify(p,z)
        z=copy.deepcopy(r);z['optimum']['dispatched']=True
        with self.assertRaises(ValueError):verify(p,z)

    def test_invalid_contract(self):
        base=FIXTURES[0]['input'];bad=[]
        for key,value in [('capacities',[True,1]),('capacities',[-1,1]),('capacities',[3,1]),('eligible',[[0,0],[1]]),('eligible',[[False],[1]]),('eligible',[[2],[0]]),('probabilities',[[.5,.5],[1,0]]),('probabilities',[['1/3','1/3'],[1,0]]),('probabilities',[['-1','2'],[1,0]]),('probabilities',[['1/0','0'],[1,0]]),('targets',[True,1]),('expert_values',[[1],[1]])]:
            p=copy.deepcopy(base);p[key]=value;bad.append(p)
        bad.extend([{},dict(base,unknown=1),payload([['1/2','1/2']]*8,[1,1])])
        for p in bad:
            for fn in [audit,reference]:
                with self.assertRaises(ValueError):fn(p)

    def test_strict_cli(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'input.json';r=Path(d)/'report.json';p.write_text(json.dumps(FIXTURES[0]['input']))
            def cli(*args):return subprocess.run([sys.executable,'-m','moelab.capacity_cli',*map(str,args)],capture_output=True,text=True)
            run=cli('audit',p);self.assertEqual(run.returncode,0,run.stderr);r.write_text(run.stdout)
            self.assertTrue(json.loads(cli('verify',p,r).stdout)['verified'])
            p.write_text('{"x":1,"x":2}');run=cli('audit',p);self.assertEqual(run.returncode,2);self.assertIn('duplicate',run.stderr)
            p.write_text('NaN');self.assertEqual(cli('audit',p).returncode,2)
            p.write_text('['*50+'0'+']'*50);self.assertEqual(cli('audit',p).returncode,2)
            self.assertEqual(cli('audit',Path(d)/'missing').returncode,2)

    def test_original_frozen_gate_bridge(self):
        from moelab.routes import _inference
        frozen=json.loads((ROOT/'moelab/route_data/frozen.json').read_text())
        for model in frozen['models']:
            item=next(i for i in FIXTURES if i['id']=='frozen_'+model)
            for i,row in enumerate(frozen['splits']['original_train'][:6]):
                native=_inference(frozen['models'][model],row['x']);gate=list(map(F,native['probabilities']));z=sum(gate)
                self.assertEqual(item['input']['probabilities'][i],[str(x/z)for x in gate]);self.assertEqual(item['input']['expert_values'][i],[str(F(x))for x in native['experts']])
                self.assertEqual(item['input']['targets'][i],str(F(row['target'])))
