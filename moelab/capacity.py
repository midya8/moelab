"""Exact finite capacity dispatch. No training, floating arithmetic or network."""
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
import re


def rat(value):
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or len(value)>160 or not re.fullmatch(r'-?\d+(?:/[1-9]\d*)?',value):
        raise ValueError('rational must be an integer or fraction string')
    v=F(value)
    if max(v.numerator.bit_length(),v.denominator.bit_length())>256:
        raise ValueError('rational exceeds 256 bits')
    return v


def validate(p):
    if not isinstance(p,dict) or set(p)!={'probabilities','capacities','eligible','expert_values','targets'}:
        raise ValueError('expected probabilities, capacities, eligible, expert_values, targets only')
    rows=p['probabilities']
    if not isinstance(rows,list) or not 1<=len(rows)<=7:raise ValueError('1..7 tokens required')
    if not isinstance(rows[0],list) or not 2<=len(rows[0])<=4:raise ValueError('2..4 experts required')
    n,e=len(rows),len(rows[0]);q=[]
    for row in rows:
        if not isinstance(row,list) or len(row)!=e:raise ValueError('probability dimensions')
        r=list(map(rat,row))
        if min(r)<0 or sum(r)!=1:raise ValueError('nonnegative probability rows must sum to one exactly')
        q.append(r)
    c=p['capacities']
    if not isinstance(c,list) or len(c)!=e or any(type(v)is not int or not 0<=v<=n for v in c):raise ValueError('capacity integer 0..tokens required')
    eligible=p['eligible']
    if not isinstance(eligible,list) or len(eligible)!=n:raise ValueError('eligibility dimensions')
    a=[]
    for row in eligible:
        if not isinstance(row,list) or any(type(v)is not int or not 0<=v<e for v in row) or len(set(row))!=len(row):raise ValueError('eligible expert indices must be distinct')
        a.append(sorted(row))
    values=p['expert_values'];targets=p['targets']
    if not isinstance(values,list) or len(values)!=n or not isinstance(targets,list)or len(targets)!=n:raise ValueError('value/target dimensions')
    v=[]
    for row in values:
        if not isinstance(row,list) or len(row)!=e:raise ValueError('expert values dimensions')
        v.append(list(map(rat,row)))
    t=list(map(rat,targets))
    return q,c,a,v,t


def _key(i,u):return str(i)+':'+','.join(map(str,u))


def audit(payload):
    q,cap,eligible,values,targets=validate(payload);n,e=len(q),len(cap)
    def options(i,u):return [j for j in eligible[i]if u[j]<cap[j]]+[-1]
    def next_u(u,j):return tuple(x+(k==j)for k,x in enumerate(u))
    @lru_cache(None)
    def solve(i,u):
        if i==n:return (0,F(0)),1,()
        candidates=[]
        for j in options(i,u):
            tail,count,_=solve(i+1,next_u(u,j))
            objective=(tail[0]+(j!=-1),tail[1]+(q[i][j]if j!=-1 else 0))
            candidates.append((j,objective,count))
        best=max(x[1]for x in candidates)
        return best,sum(x[2]for x in candidates if x[1]==best),tuple(j for j,obj,count in candidates if obj==best)
    start=(0,)*e;best,count,_=solve(0,start)
    # Emit every reachable capacity state, including suboptimal incoming paths.
    layers=[{start}];states=[]
    for i in range(n+1):
        nxt=set()
        for u in sorted(layers[i]):
            obj,ways,optimal=solve(i,u);edges=[]
            if i<n:
                for j in options(i,u):
                    nu=next_u(u,j);nxt.add(nu);tail,_,_=solve(i+1,nu)
                    value=(tail[0]+(j!=-1),tail[1]+(q[i][j]if j!=-1 else 0))
                    edges.append({'expert':j,'next':_key(i+1,nu),'objective':[value[0],str(value[1])],'optimal':j in optimal})
            states.append({'id':_key(i,u),'token':i,'used':list(u),'objective':[obj[0],str(obj[1])],'count':str(ways),'edges':edges})
        if i<n:layers.append(nxt)
    # Uniform-over-optima counts, not an assertion of stochastic dispatch.
    prefixes={start:1};marginals=[[0]*(e+1)for _ in range(n)];canonical=[];cu=start
    for i in range(n):
        cs=solve(i,cu)[2][0];canonical.append(cs);cu=next_u(cu,cs)
        nxt={}
        for u,paths in prefixes.items():
            for j in solve(i,u)[2]:
                nu=next_u(u,j);completions=solve(i+1,nu)[1]
                marginals[i][e if j==-1 else j]+=paths*completions
                nxt[nu]=nxt.get(nu,0)+paths
        prefixes=nxt
    def dispatch(assignment):
        loads=[assignment.count(j)for j in range(e)]
        pred=[F(0)if j==-1 else q[i][j]*values[i][j]for i,j in enumerate(assignment)]
        mass=sum((q[i][j]for i,j in enumerate(assignment)if j!=-1),F(0))
        return {'assignment':assignment,'loads':loads,'dispatched':sum(loads),'retained_probability':str(mass),
                'weighted_updates':list(map(str,pred)),'update_mse':str(sum(((p-t)**2 for p,t in zip(pred,targets)),F(0))/n)}
    @lru_cache(None)
    def errors(i,u):
        if i==n:return (F(0),()),(F(0),())
        lows=[];highs=[]
        for j in solve(i,u)[2]:
            pred=F(0)if j<0 else q[i][j]*values[i][j]
            loss=(pred-targets[i])**2
            low,high=errors(i+1,next_u(u,j))
            lows.append((loss+low[0],(j,)+low[1]));highs.append((loss+high[0],(j,)+high[1]))
        return min(lows,key=lambda x:x[0]),max(highs,key=lambda x:x[0])
    low,high=errors(0,start)
    error_range={'minimum':str(low[0]/n),'maximum':str(high[0]/n),'minimum_witness':dispatch(list(low[1])),'maximum_witness':dispatch(list(high[1]))}
    def greedy(order,reroute):
        used=[0]*e;assignment=[-1]*n
        for i in order:
            ranked=sorted(eligible[i],key=lambda j:(-q[i][j],j))
            candidates=ranked if reroute else ranked[:1]
            j=next((j for j in candidates if used[j]<cap[j]),-1)
            if j!=-1:used[j]+=1
            assignment[i]=j
        return dispatch(assignment)
    confidence=sorted(range(n),key=lambda i:(-max((q[i][j]for j in eligible[i]),default=F(0)),i))
    # Complete capacitated Hall table proves the minimum drop count independently
    # of the score objective. Empty neighborhoods are intentional and visible.
    hall=[]
    for size in range(1,n+1):
        for subset in combinations(range(n),size):
            neighbors=sorted(set(j for i in subset for j in eligible[i]));slots=sum(cap[j]for j in neighbors)
            hall.append({'tokens':list(subset),'neighbors':neighbors,'slots':slots,'deficiency':max(0,size-slots)})
    deficiency=max([0]+[r['deficiency']for r in hall]);assert n-best[0]==deficiency
    return {'schema':'moelab-capacity-v1','input':{'probabilities':[[str(x)for x in row]for row in q],'capacities':cap,'eligible':eligible,'expert_values':[[str(x)for x in row]for row in values],'targets':list(map(str,targets))},
            'objective':'maximum dispatched count, then maximum sum of original gate probabilities',
            'optimum':{'dispatched':best[0],'retained_probability':str(best[1]),'count':str(count),'canonical':dispatch(canonical),
                       'marginal_counts':[[str(x)for x in row]for row in marginals],'drop_column':e,'update_mse_range':error_range},
            'baselines':{'ordered_top_choice':greedy(list(range(n)),False),'confidence_top_choice':greedy(confidence,False),
                         'ordered_best_available':greedy(list(range(n)),True),'confidence_best_available':greedy(confidence,True)},
            'confidence_order':confidence,'minimum_drops':deficiency,'hall':hall,'states':states}
