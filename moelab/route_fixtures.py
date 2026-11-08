"""Reproduce original models and data without changing their training loop."""
import json
import random
from pathlib import Path
from .data import DIM,N_REGIMES,make_split
from .moe import MoE
from .eval import DenseMLP,train_moe,run_evaluation,mse


def build():
    seed=11
    train=make_split(60,seed)
    test=make_split(25,seed+1000)
    models={}
    for name,aux in [('aux',.05),('no_aux',0),('initialization',None)]:
        model=MoE(DIM,N_REGIMES,random.Random(seed))
        if aux is not None:
            train_moe(model,train,80,.1,aux)
        models[name]={key:getattr(model,key)for key in ['W','b','R','c']}
    dense=DenseMLP(DIM,8,random.Random(seed))
    for _ in range(80):
        for x,t,_ in train:
            dense.train_step(x,t,.02)
    dense_model={key:getattr(dense,key)for key in ['W1','b1','w2','b2']}
    original=run_evaluation()
    # Independently tie serialized models back to the entire original report.
    restored={}
    for name,values in models.items():
        m=MoE(DIM,N_REGIMES,random.Random(0))
        for key,v in values.items():setattr(m,key,v)
        restored[name]=m
    aux,na=restored['aux'],restored['no_aux']
    def shares(m):
        return [sum(m.top_expert(x)==e for x,_,_ in test)/len(test)for e in range(N_REGIMES)]
    purity=[]
    for regime in range(N_REGIMES):
        tops=[aux.top_expert(x)for x,_,r in test if r==regime]
        purity.append(max(tops.count(e)for e in range(N_REGIMES))/len(tops))
    report={
        'moe_mse_soft':mse([aux.predict_soft(x)for x,_,_ in test],test),
        'moe_mse_top1':mse([aux.predict_top1(x)for x,_,_ in test],test),
        'moe_noaux_mse_top1':mse([na.predict_top1(x)for x,_,_ in test],test),
        'mlp_mse':mse([dense.predict(x)for x,_,_ in test],test),
        'moe_params':aux.param_count(),'mlp_params':dense.param_count(),
        'routing_purity_min':min(purity),'expert_shares':shares(aux),
        'noaux_max_share':max(shares(na)),'aux_max_share':max(shares(aux))}
    if original!=report:raise AssertionError('frozen original report differs')
    def rows(data):return [{'x':x,'target':t,'regime':r}for x,t,r in data]
    controls=controls_fixture()
    return {'controls':controls,'schema':'moelab-route-frozen-v1','source_main':'ea54b17c90c14a266b926b2894e61a30e9fcf501',
            'models':models,'dense_model':dense_model,
            'splits':{'original_train':rows(train),'original_test':rows(test)},
            'original_report':report}


def controls_fixture():
    def p(R,c,b,W=None):
        return {'model':{'R':[[x]for x in R],'c':c,'b':b,'W':[[0]for _ in R] if W is None else [[x]for x in W]},'origin':[0],'destination':[1],
                'evaluation':{'two_positions':[{'x':[0],'target':0},{'x':[1],'target':0}]}}
    return {
        'jump':p([-1,1],[.5,-.5],[-1,1]),
        'zero_jump':p([-1,1],[.5,-.5],[3,3]),
        'triple_tie':p([0,-1,1],[0,.5,-.5],[9,-1,1]),
        'persistent_tie':p([0,0],[0,0],[-1,1]),
        'hidden_crossing':p([1,-1,0],[0,1,5],[1,2,3]),
        'thin_cell':p([1,0,-1],[0,.5+2**-40,1],[1,2,3]),
        'endpoint_tie':p([1,0],[0,0],[1,-1]),
        'sharp_balanced':p([40,-40],[-20,20],[-1,1])}


def witnesses(f):
    from .routes import audit
    result=[]
    for name in f['models']:
        for start,end in [(0,1),(0,17),(0,79),(2,60)]:
            for probe in [.25,.75]:
                meta={'model':name,'start':start,'end':end,'probe':probe}
                result.append({'input':meta,'report':audit(payload(f,meta),probe)})
    for name in f['controls']:
        meta={'control':name,'probe':.5}
        result.append({'input':meta,'report':audit(payload(f,meta),.5)})
    return result


def payload(f,meta):
    if 'control'in meta:return f['controls'][meta['control']]
    rows=f['splits']['original_train']
    return {'model':f['models'][meta['model']],'origin':rows[meta['start']]['x'],
            'destination':rows[meta['end']]['x'],'evaluation':f['splits']}


def write(path):
    import gzip
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);f=build()
    p.write_text(json.dumps(f,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n')
    w=witnesses(f)
    (p.parent/'witnesses.json.gz').write_bytes(gzip.compress(json.dumps(w,separators=(',',':'),sort_keys=True,allow_nan=False).encode(),mtime=0))
    return {'models':len(f['models']),'witnesses':len(w),'original_report':f['original_report']}


if __name__=='__main__':
    import sys
    print(json.dumps(write(sys.argv[1]),sort_keys=True))
