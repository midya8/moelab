import copy,json,math,random,unittest
from decimal import Decimal,localcontext
from pathlib import Path
from moelab.routes import audit,validate,batch_audit,_inference
from moelab.moe import MoE


def control():
    return {'model':{'W':[[0],[0]],'b':[-1,1],'R':[[-1],[1]],'c':[.5,-.5]},'origin':[0],'destination':[1]}

class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=json.loads((Path(__file__).resolve().parents[1]/'moelab/route_data/frozen.json').read_text())
    def test_visible_jump_and_point_tie(self):
        r=audit(control());points=[p for p in r['topology']['points']if p['route_changes']];self.assertEqual(len(points),1)
        p=points[0];self.assertEqual(p['at']['value'],.5);self.assertEqual(p['state']['winner'],0);self.assertEqual(p['left_limit'],-.5);self.assertEqual(p['right_limit'],.5);self.assertEqual(p['jump'],1)
    def test_route_change_need_not_jump(self):
        p=control();p['model']['b']=[3,3];r=audit(p);self.assertEqual([p['jump']for p in r['topology']['points']if p['route_changes']],[0])
    def test_point_can_differ_from_both_limits(self):
        p={'model':{'W':[[0],[0],[0]],'b':[9,-1,1],'R':[[0],[-1],[1]],'c':[0,.5,-.5]},'origin':[0],'destination':[1]}
        r=audit(p);point=r['topology']['points'][1];self.assertEqual(point['state']['co_maximal'],[0,1,2]);self.assertEqual(point['state']['winner'],0);self.assertEqual(point['state']['sparse'],3);self.assertAlmostEqual(point['left_limit'],-1/3);self.assertAlmostEqual(point['right_limit'],1/3)
    def test_endpoint_and_constant_path(self):
        p=control();p['origin']=[.5];p['destination']=[.5];r=audit(p);self.assertEqual(len(r['topology']['cells']),1);self.assertEqual(r['topology']['cells'][0]['co_maximal'],[0,1]);self.assertEqual(r['probe']['branch_derivative'],0);self.assertIsNone(r['nearest_transition'])
    def test_original_forward_all_rows_three_models(self):
        for name,values in self.f['models'].items():
            m=MoE(16,4,random.Random(0))
            for k,v in values.items():setattr(m,k,v)
            for rows in self.f['splits'].values():
                for row in rows:
                    s=_inference(values,row['x']);self.assertEqual(s['sparse'],m.predict_top1(row['x']));self.assertEqual(s['soft'],m.predict_soft(row['x']));self.assertEqual(s['winner'],m.top_expert(row['x']))
    def test_original_metrics_and_noninterference(self):
        f=self.f;rows=f['splits']['original_train'];p={'model':f['models']['aux'],'origin':rows[0]['x'],'destination':rows[1]['x'],'evaluation':f['splits']};r=audit(p)
        self.assertAlmostEqual(r['evaluation']['original_test']['sparse_mse'],f['original_report']['moe_mse_top1'],delta=1e-12);self.assertAlmostEqual(r['evaluation']['original_test']['soft_mse'],f['original_report']['moe_mse_soft'],delta=1e-12)
        p=copy.deepcopy(p)
        for ds in p['evaluation'].values():
            for row in ds:row['target']+=1
        changed=audit(p)
        self.assertEqual(r['topology'],changed['topology']);self.assertEqual(r['probe'],changed['probe']);self.assertEqual(r['samples'],changed['samples']);self.assertNotEqual(r['evaluation']['original_test']['sparse_mse'],changed['evaluation']['original_test']['sparse_mse'])
    def test_branch_derivative_centered_away_from_boundaries(self):
        for name,m in self.f['models'].items():
            rows=self.f['splits']['original_train'];p={'model':m,'origin':rows[0]['x'],'destination':rows[1]['x']}
            for t in [.1,.2,.3,.6,.9]:
                r=audit(p,t,3);a=audit(p,t-1e-5,3)['probe'];b=audit(p,t+1e-5,3)['probe']
                if a['rational_extension']['winner']==b['rational_extension']['winner']:
                    fd=(b['rational_extension']['sparse']-a['rational_extension']['sparse'])/2e-5;self.assertAlmostEqual(fd,r['probe']['branch_derivative'],delta=1e-6)
    def test_three_penalties_not_interchangeable(self):
        m={'W':[[0],[0]],'b':[0,0],'R':[[20],[-20]],'c':[0,0]}
        rows=[{'x':[1],'target':0},{'x':[-1],'target':0}];r=batch_audit(m,rows)
        self.assertEqual(r['shares'],[.5,.5]);self.assertAlmostEqual(r['mean_per_example_penalty'],2);self.assertAlmostEqual(r['batch_mean_square_penalty'],1);self.assertAlmostEqual(r['switch_frequency_probability_penalty'],1)
        m['R']=[[0],[0]];r=batch_audit(m,rows)
        self.assertEqual(r['shares'],[1,0]);self.assertEqual(r['switch_frequency_probability_penalty'],1);self.assertEqual(r['mean_per_example_penalty'],1)
    def test_aux_gap_equals_variance(self):
        for m in self.f['models'].values():
            rows=self.f['splits']['original_test'];r=batch_audit(m,rows);means=r['mean_probabilities'];var=4*sum(sum((s['probabilities'][e]-means[e])**2 for e in range(4))for s in r['rows'])/len(rows)
            self.assertAlmostEqual(r['mean_per_example_penalty']-r['batch_mean_square_penalty'],var,places=12)
    def test_stable_large_logits_and_signed_zero(self):
        p=control();p['model']['R']=[[100],[-100]];p['origin']=[20];p['destination']=[-20];r=audit(p);json.dumps(r,allow_nan=False);self.assertEqual(r['samples'][0]['original_binary64']['probabilities'],[1,0])
        p['origin']=[-0.];audit(p)
    def test_reject_schema_and_limits(self):
        bad=[]
        for key,value in [('extra',1),('origin',[True]),('destination',[21]),('evaluation',{'x':[]})]:
            p=control();p[key]=value;bad.append(p)
        for key,value in [('R',[[0]]),('b',[0,float('inf')]),('W',[[0],[101]])]:
            p=control();p['model'][key]=value;bad.append(p)
        p=control();p['evaluation']={'x':[{'x':[0],'target':0,'regime':True}]};bad.append(p)
        p=control();p['evaluation']={'x':[{'x':[0],'target':0}]*513};bad.append(p)
        for p in bad:
            with self.assertRaises(ValueError):audit(p)
        for t in [-.1,1.1,True,float('nan')]:
            with self.assertRaises(ValueError):audit(control(),t)
        for samples in [1,130,2.5,True]:
            with self.assertRaises(ValueError):audit(control(),samples=samples)
    def test_160_independent_decimal_predictions_and_derivatives(self):
        rng=random.Random(917)
        with localcontext()as ctx:
            ctx.prec=70
            for _ in range(160):
                a=[rng.randint(-4,4)for _ in range(3)];b=[rng.randint(-4,4)for _ in range(3)];u=[rng.randint(-4,4)for _ in range(3)];v=[rng.randint(-4,4)for _ in range(3)];t=rng.randint(0,8)/8
                p={'model':{'W':[[x]for x in v],'b':u,'R':[[x]for x in b],'c':a},'origin':[0],'destination':[1]};r=audit(p,t,2)['probe'];dt=Decimal(t);z=[Decimal(ai)+dt*Decimal(bi)for ai,bi in zip(a,b)];ex=[x.exp()for x in z];g=[x/sum(ex)for x in ex];y=[Decimal(ui)+dt*Decimal(vi)for ui,vi in zip(u,v)];e=max(range(3),key=lambda i:z[i]);mean=sum(gi*Decimal(bi)for gi,bi in zip(g,b));dy=g[e]*(Decimal(v[e])+y[e]*(Decimal(b[e])-mean));self.assertAlmostEqual(r['rational_extension']['sparse'],float(g[e]*y[e]),delta=1e-12);self.assertAlmostEqual(r['branch_derivative'],float(dy),delta=1e-12)

    def test_rounded_softmax_owner_is_not_exact_owner(self):
        p={'model':{'W':[[0],[0]],'b':[1,2],'R':[[0],[0]],'c':[0,5e-324]},'origin':[0],'destination':[1]}
        r=audit(p);self.assertEqual(r['probe']['original_binary64']['winner'],0);self.assertEqual(r['probe']['rational_extension']['winner'],1);self.assertTrue(r['probe']['routing_disagrees']);self.assertEqual(r['routing_disagreement_samples'],65)
    def test_null_evaluation_and_integer_valued_regime(self):
        p=control();p['evaluation']=None
        with self.assertRaises(ValueError):audit(p)
        p['evaluation']={'x':[{'x':[0],'target':0,'regime':1.0}]};self.assertEqual(audit(p)['evaluation']['x']['rows'][0]['regime'],1)
