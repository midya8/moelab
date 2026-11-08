"""Bounded path routing audit: exact affine topology, floating-point predictions.

This is an intervention in input space, not model training or a robustness
certificate. Exact rational topology and original binary64 inference are
reported separately; only the former decides mathematical point ties.
"""
from fractions import Fraction as F
import math
from .envelope import rational_envelope, _rational


def _keys(value,required,optional=()):
    if not isinstance(value,dict) or not set(required)<=set(value) or set(value)-set(required)-set(optional):
        raise ValueError('missing or unknown object keys')


def _num(value,bound):
    if type(value) not in (int,float) or abs(value)>bound or not math.isfinite(value):
        raise ValueError('expected bounded finite number, not bool')
    return float(value)


def _vec(v,n,bound):
    if not isinstance(v,list) or len(v)!=n:raise ValueError('vector dimension mismatch')
    return [_num(x,bound) for x in v]


def validate(p):
    _keys(p,['model','origin','destination'],['evaluation'])
    m=p['model'];_keys(m,['W','b','R','c'])
    if not isinstance(m['W'],list) or not 2<=len(m['W'])<=16:raise ValueError('2..16 experts required')
    e=len(m['W'])
    if not isinstance(m['W'][0],list) or not 1<=len(m['W'][0])<=32:raise ValueError('1..32 inputs required')
    d=len(m['W'][0]);v={}
    for k in ['W','R']:
        if not isinstance(m[k],list) or len(m[k])!=e:raise ValueError('expert dimension mismatch')
        v[k]=[_vec(row,d,100)for row in m[k]]
    for k in ['b','c']:v[k]=_vec(m[k],e,100)
    origin=_vec(p['origin'],d,20);destination=_vec(p['destination'],d,20)
    datasets=p.get('evaluation',{});out={};total=0
    if not isinstance(datasets,dict) or len(datasets)>4:raise ValueError('at most four evaluation splits')
    for name,rows in datasets.items():
        if not isinstance(name,str) or not name or len(name)>40 or not isinstance(rows,list) or not rows:raise ValueError('bad split')
        total+=len(rows)
        if total>512:raise ValueError('at most 512 evaluation rows')
        result=[]
        for row in rows:
            _keys(row,['x','target'],['regime']);item={'x':_vec(row['x'],d,20),'target':_num(row['target'],1000)}
            if 'regime'in row:
                r=row['regime']
                r=_num(r,15)
                if not r.is_integer() or r<0:raise ValueError('regime must be integer-valued 0..15')
                item['regime']=int(r)
            result.append(item)
        out[name]=result
    return {'model':v,'origin':origin,'destination':destination,'evaluation':out}


def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _fdot(a,b):return sum((F(x)*F(y)for x,y in zip(a,b)),F(0))
def _softmax(z):
    peak=max(z);ex=[math.exp(x-peak)for x in z];den=sum(ex)
    return [x/den for x in ex]

def _inference(m,x):
    z=[_dot(r,x)+c for r,c in zip(m['R'],m['c'])]
    p=_softmax(z);e=p.index(max(p));v=[_dot(w,x)+b for w,b in zip(m['W'],m['b'])]
    return {'logits':z,'probabilities':p,'experts':v,'winner':e,
            'sparse':p[e]*v[e],'soft':sum(pi*vi for pi,vi in zip(p,v)),
            'per_example_penalty':len(p)*sum(pi*pi for pi in p)}


def batch_audit(m,rows):
    e=len(m['W']);n=len(rows);means=[0.]*e;counts=[0]*e;items=[]
    for i,row in enumerate(rows):
        state=_inference(m,row['x']);counts[state['winner']]+=1
        for k in range(e):means[k]+=state['probabilities'][k]/n
        item={'position':i,'target':row['target'],**state,
              'squared_error':(state['sparse']-row['target'])**2,
              'soft_squared_error':(state['soft']-row['target'])**2}
        if 'regime'in row:item['regime']=row['regime']
        items.append(item)
    shares=[c/n for c in counts]
    return {'count':n,'counts':counts,'shares':shares,'mean_probabilities':means,
            'sparse_mse':sum(r['squared_error']for r in items)/n,
            'soft_mse':sum(r['soft_squared_error']for r in items)/n,
            'mean_per_example_penalty':sum(r['per_example_penalty']for r in items)/n,
            'batch_mean_square_penalty':e*sum(p*p for p in means),
            'switch_frequency_probability_penalty':e*sum(f*p for f,p in zip(shares,means)),
            'rows':items}


def audit(payload,probe=.5,samples=65):
    p=validate(payload);probe=_num(probe,1)
    if probe<0:raise ValueError('probe must be 0..1')
    if type(samples)!=int or not 2<=samples<=129:raise ValueError('2..129 display samples')
    m=p['model'];origin=p['origin'];end=p['destination'];d=len(origin);e=len(m['W'])
    # No rounded dot products or rounded endpoint subtraction enter topology.
    x0=list(map(F,origin));direction=[F(b)-F(a)for a,b in zip(origin,end)]
    a=[sum((F(rj)*xj for rj,xj in zip(r,x0)),F(c))for r,c in zip(m['R'],m['c'])]
    b=[sum((F(rj)*xj for rj,xj in zip(r,direction)),F(0))for r in m['R']]
    u=[sum((F(wj)*xj for wj,xj in zip(w,x0)),F(bias))for w,bias in zip(m['W'],m['b'])]
    v=[sum((F(wj)*xj for wj,xj in zip(w,direction)),F(0))for w in m['W']]
    topo=rational_envelope(a,b,F(0),F(1))
    def rational_state(t):
        logits=[ai+t*bi for ai,bi in zip(a,b)];best=max(logits)
        tied=[i for i,z in enumerate(logits)if z==best]
        gates=_softmax(list(map(float,logits)));experts=[float(ui+t*vi)for ui,vi in zip(u,v)]
        weighted=[pi*yi for pi,yi in zip(gates,experts)]
        return {'winner':tied[0],'co_maximal':tied,'logits':list(map(float,logits)),
                'probabilities':gates,'experts':experts,'weighted_experts':weighted,
                'sparse':weighted[tied[0]],'soft':sum(weighted)}
    for point in topo['points']:
        t=F(int(point['at']['numerator']),int(point['at']['denominator']));state=rational_state(t)
        point['state']=state
        left=point['left_winner'];right=point['right_winner']
        point['left_limit']=None if left is None else state['weighted_experts'][left]
        point['right_limit']=None if right is None else state['weighted_experts'][right]
        point['jump']=None if left is None or right is None else point['right_limit']-point['left_limit']
    def sample(t):
        # Preserve original inference summation, gate argmax, and rounded path.
        x=[float(o)+float(t)*(float(dest)-float(o))for o,dest in zip(origin,end)]
        native=_inference(m,x);ideal=rational_state(F(t));owner=ideal['winner'];g=ideal['probabilities'];ys=ideal['experts']
        mean_slope=sum(pi*float(bi)for pi,bi in zip(g,b))
        derivative=g[owner]*(float(v[owner])+ys[owner]*(float(b[owner])-mean_slope))
        return {'lambda':float(t),'x':x,'original_binary64':native,'rational_extension':ideal,
                'branch_derivative':derivative,'routing_disagrees':native['winner']!=owner}
    displayed=[sample(i/(samples-1))for i in range(samples)]
    selected=sample(probe)
    transitions=[point for point in topo['points']if point['route_changes']]
    nearest=min(transitions,key=lambda point:abs(point['at']['value']-probe))if transitions else None
    return {'schema':'moelab-route-v1','dimensions':{'inputs':d,'experts':e,'parameters':2*e*(d+1)},
            'probe':selected,'topology':topo,'samples':displayed,
            'nearest_transition':None if nearest is None else {'at':nearest['at'],'distance_in_lambda':abs(nearest['at']['value']-probe)},
            'routing_disagreement_samples':sum(s['routing_disagrees']for s in displayed),
            'coefficients':{'router_intercepts':list(map(_rational,a)),'router_slopes':list(map(_rational,b)),
                            'expert_intercepts':list(map(_rational,u)),'expert_slopes':list(map(_rational,v))},
            'evaluation':{name:batch_audit(m,rows)for name,rows in p['evaluation'].items()}}
