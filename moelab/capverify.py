"""Independent exhaustive dispatch reference. Never imports capacity producer."""
import itertools
import re
from fractions import Fraction


def _parse(p):
    if type(p)is not dict or sorted(p)!=sorted(['probabilities','capacities','eligible','expert_values','targets']):raise ValueError('input fields')
    def fraction(x):
        if type(x)is int:x=str(x)
        if type(x)is not str or len(x)>160 or re.fullmatch(r'-?\d+(?:/[1-9]\d*)?',x)is None:raise ValueError('rational syntax')
        z=Fraction(x)
        if z.numerator.bit_length()>256 or z.denominator.bit_length()>256:raise ValueError('rational size')
        return z
    ps=p['probabilities'];cs=p['capacities'];es=p['eligible'];vs=p['expert_values'];ts=p['targets']
    if type(ps)is not list or not 1<=len(ps)<=7 or type(ps[0])is not list or not 2<=len(ps[0])<=4:raise ValueError('bounds')
    n,m=len(ps),len(ps[0]);q=[];v=[]
    if type(cs)is not list or len(cs)!=m or any(type(c)is not int or c<0 or c>n for c in cs):raise ValueError('capacities')
    if type(es)is not list or len(es)!=n or type(vs)is not list or len(vs)!=n or type(ts)is not list or len(ts)!=n:raise ValueError('shapes')
    for i in range(n):
        if type(ps[i])is not list or len(ps[i])!=m or type(vs[i])is not list or len(vs[i])!=m:raise ValueError('shapes')
        row=list(map(fraction,ps[i]));q.append(row);v.append(list(map(fraction,vs[i])))
        if any(x<0 for x in row)or sum(row)!=1:raise ValueError('probabilities')
        if type(es[i])is not list or any(type(j)is not int or not 0<=j<m for j in es[i])or len(set(es[i]))!=len(es[i]):raise ValueError('eligibility')
    return q,cs,[sorted(x)for x in es],v,list(map(fraction,ts))


def reference(payload):
    q,c,a,v,t=_parse(payload);n,m=len(q),len(c)
    choices=[row+[-1]for row in a]
    def routes(i,u):
        # Full Cartesian enumeration, no Bellman solver or producer calls.
        result=[]
        for tail in itertools.product(*choices[i:]):
            loads=[tail.count(j)for j in range(m)]
            if any(loads[j]+u[j]>c[j]for j in range(m)):continue
            score=sum((q[i+k][j]for k,j in enumerate(tail)if j>=0),Fraction(0))
            result.append((tail,(sum(loads),score)))
        return result
    all_routes=routes(0,[0]*m);best=max(x[1]for x in all_routes);opt=[x[0]for x in all_routes if x[1]==best]
    def summary(route):
        loads=[route.count(j)for j in range(m)];pred=[Fraction(0)if j<0 else q[i][j]*v[i][j]for i,j in enumerate(route)]
        return {'assignment':list(route),'loads':loads,'dispatched':sum(loads),'retained_probability':str(sum((q[i][j]for i,j in enumerate(route)if j>=0),Fraction(0))),
                'weighted_updates':[str(x)for x in pred],'update_mse':str(sum(((pred[i]-t[i])**2 for i in range(n)),Fraction(0))/n)}
    state_keys=set()
    for route,_ in all_routes:
        u=[0]*m;state_keys.add((0,tuple(u)))
        for i,j in enumerate(route):
            if j>=0:u[j]+=1
            state_keys.add((i+1,tuple(u)))
    def key(i,u):return str(i)+':'+','.join(map(str,u))
    states=[]
    for i,u in sorted(state_keys):
        tails=routes(i,u);obj=max(x[1]for x in tails);besttails=[tail for tail,s in tails if s==obj];edges=[]
        if i<n:
            for j in choices[i]:
                relevant=[s for tail,s in tails if tail[0]==j]
                if not relevant:continue
                nu=[x+(k==j)for k,x in enumerate(u)];val=max(relevant)
                edges.append({'expert':j,'next':key(i+1,nu),'objective':[val[0],str(val[1])],'optimal':val==obj})
        states.append({'id':key(i,u),'token':i,'used':list(u),'objective':[obj[0],str(obj[1])],'count':str(len(besttails)),'edges':edges})
    low=min(opt,key=lambda route:Fraction(summary(route)['update_mse']));high=max(opt,key=lambda route:Fraction(summary(route)['update_mse']))
    error_range={'minimum':summary(low)['update_mse'],'maximum':summary(high)['update_mse'],'minimum_witness':summary(low),'maximum_witness':summary(high)}
    marg=[[str(sum(route[i]==j for route in opt))for j in list(range(m))+[-1]]for i in range(n)]
    order=sorted(range(n),key=lambda i:(-max([q[i][j]for j in a[i]]or[Fraction(0)]),i))
    baselines={}
    for name,ids,reroute in [('ordered_top_choice',list(range(n)),False),('confidence_top_choice',order,False),('ordered_best_available',list(range(n)),True),('confidence_best_available',order,True)]:
        assignment=[-1]*n;loads=[0]*m
        for i in ids:
            experts=sorted(a[i],key=lambda j:(-q[i][j],j))
            if not reroute:experts=experts[:1]
            for j in experts:
                if loads[j]<c[j]:assignment[i]=j;loads[j]+=1;break
        baselines[name]=summary(assignment)
    hall=[]
    for mask in range(1,2**n):
        tokens=[i for i in range(n)if mask&(1<<i)];neighbors=[j for j in range(m)if any(j in a[i]for i in tokens)]
        slots=sum(c[j]for j in neighbors);hall.append({'tokens':tokens,'neighbors':neighbors,'slots':slots,'deficiency':max(0,len(tokens)-slots)})
    hall.sort(key=lambda r:(len(r['tokens']),r['tokens']))
    return {'schema':'moelab-capacity-v1','input':{'probabilities':[[str(x)for x in row]for row in q],'capacities':c,'eligible':a,'expert_values':[[str(x)for x in row]for row in v],'targets':list(map(str,t))},
            'objective':'maximum dispatched count, then maximum sum of original gate probabilities',
            'optimum':{'dispatched':best[0],'retained_probability':str(best[1]),'count':str(len(opt)),'canonical':summary(opt[0]),'marginal_counts':marg,'drop_column':m,'update_mse_range':error_range},
            'baselines':baselines,'confidence_order':order,'minimum_drops':max(r['deficiency']for r in hall),'hall':hall,'states':states}


def verify(payload,report):
    expected=reference(payload)
    def same(a,b,path):
        if type(a)is not type(b):raise ValueError('certificate type mismatch at '+path)
        if type(a)is dict:
            if a.keys()!=b.keys():raise ValueError('certificate fields at '+path)
            for k in a:same(a[k],b[k],path+'/'+k)
        elif type(a)is list:
            if len(a)!=len(b):raise ValueError('certificate length at '+path)
            for i,(x,y)in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
        elif a!=b:raise ValueError('certificate mismatch at '+path)
    same(expected,report,'')
    return {'verified':True,'schema':expected['schema'],'states':len(expected['states']),'optimal_assignments':expected['optimum']['count'],'minimum_drops':expected['minimum_drops']}
