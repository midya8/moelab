/* Supplied gate-table regret. BigInt vertex enumeration, no native calls. */
(function(root){'use strict';
const arithmetic=typeof module!=='undefined'&&module.exports?require('./capacity-engine.js'):root.Capacity;
const {Q}=arithmetic;
function strictJSON(text){if(typeof text!=='string'||new TextEncoder().encode(text).length>8000000)throw Error('input size');return arithmetic.strictJSON(text);}
const internalRat=x=>{const parts=x.split('/');return new Q(BigInt(parts[0]),parts[1]?BigInt(parts[1]):1n);};
const zero=()=>new Q(0n),one=()=>new Q(1n),sum=xs=>xs.reduce((a,b)=>a.add(b),zero());
function validate(p){
 if(!p||Array.isArray(p)||typeof p!=='object'||Object.keys(p).sort().join('|')!=='capacities|eligible|lower|probabilities|upper')throw Error('five required input fields only');
 const rows=p.probabilities;
 if(!Array.isArray(rows)||rows.length<1||rows.length>7||!Array.isArray(rows[0])||rows[0].length<2||rows[0].length>4)throw Error('1..7 tokens, 2..4 experts required');
 const n=rows.length,m=rows[0].length;
 function matrix(rows){if(!Array.isArray(rows)||rows.length!==n)throw Error('matrix rows');return rows.map(row=>{if(!Array.isArray(row)||row.length!==m)throw Error('matrix columns');return row.map(Q.parse);});}
 const q=matrix(rows),lower=matrix(p.lower),upper=matrix(p.upper),cap=p.capacities;
 if(!Array.isArray(cap)||cap.length!==m||cap.some(c=>!Number.isInteger(c)||c<0||c>n))throw Error('capacities: integers in 0..tokens');
 if(!Array.isArray(p.eligible)||p.eligible.length!==n)throw Error('eligibility rows');
 const eligible=p.eligible.map(row=>{if(!Array.isArray(row)||row.some(j=>!Number.isInteger(j)||j<0||j>=m)||new Set(row).size!==row.length)throw Error('distinct eligible indices');return [...row].sort((a,b)=>a-b);});
 for(let i=0;i<n;i++){if(sum(q[i]).cmp(one()))throw Error('nominal rows must sum to one exactly');for(let j=0;j<m;j++)if(lower[i][j].cmp(zero())<0||lower[i][j].cmp(q[i][j])>0||q[i][j].cmp(upper[i][j])>0||upper[i][j].cmp(one())>0)throw Error('bounds must contain nominal probabilities in [0,1]');}
 return {q,lower,upper,cap:[...cap],eligible,n,m};
}
function support(lower,upper,c){
 // A vertex has all but at most one coordinate at an endpoint. Unlike the
 // native producer, enumerate that finite set instead of filling by score.
 const m=c.length,order=Array.from({length:m},(_,j)=>j).sort((a,b)=>c[b].cmp(c[a])||a-b);
 let point=null,value=null;
 for(let free=0;free<m;free++){const other=Array.from({length:m},(_,j)=>j).filter(j=>j!==free);
  for(let mask=0;mask<2**other.length;mask++){const row=Array.from({length:m},zero);other.forEach((j,k)=>row[j]=(mask&(1<<k))?upper[j]:lower[j]);row[free]=one().sub(sum(other.map(j=>row[j])));
   if(row[free].cmp(lower[free])<0||row[free].cmp(upper[free])>0)continue;
   const score=sum(row.map((x,j)=>x.mul(c[j])));let cmp=value?score.cmp(value):1;
   if(!cmp)for(const j of order){cmp=row[j].cmp(point[j]);if(cmp)break;}
   if(cmp>0){value=score;point=row;}
  }
 }
 if(!point)throw Error('empty simplex');let lambda=c[order[0]];
 for(const j of order)if(point[j].cmp(lower[j])>0)lambda=c[j];
 const terms=c.map((x,j)=>{const a=x.sub(lambda).mul(lower[j]),b=x.sub(lambda).mul(upper[j]);return a.cmp(b)>0?a:b;});
 const dual=lambda.add(sum(terms));if(dual.cmp(value))throw Error('support dual mismatch');
 return {coefficient:c.map(String),point:point.map(String),value:String(value),lambda:String(lambda),dual_terms:terms.map(String),dual_bound:String(dual)};
}
function dispatch(cap,eligible,weights){
 const n=eligible.length,m=cap.length,memo=new Map(),key=(i,u)=>i+':'+u.join(','),advance=(u,j)=>u.map((x,k)=>x+Number(j===k)),opts=(i,u)=>eligible[i].filter(j=>u[j]<cap[j]).concat(-1),compare=(x,y)=>x[0]-y[0]||x[1].cmp(y[1]);
 function best(i,u){const k=key(i,u);if(memo.has(k))return memo.get(k);let result;
  if(i===n)result={obj:[0,zero()],count:1n,choices:[]};else{let obj=null,count=0n,choices=[];
   for(const j of opts(i,u)){const tail=best(i+1,advance(u,j)),value=[tail.obj[0]+Number(j>=0),tail.obj[1].add(weights[i][j])],cmp=obj?compare(value,obj):1;
    if(cmp>0){obj=value;count=tail.count;choices=[j];}else if(!cmp){count+=tail.count;choices.push(j);}}
   result={obj,count,choices};}memo.set(k,result);return result;}
 const start=Array(m).fill(0),optimal=best(0,start),canonical=[],states=[];let used=start,layer=[start];
 for(let i=0;i<n;i++){const j=best(i,used).choices[0];canonical.push(j);used=advance(used,j);}
 for(let i=0;i<=n;i++){const next=new Map();layer.sort((a,b)=>{for(let k=0;k<m;k++)if(a[k]!==b[k])return a[k]-b[k];return 0;});
  for(const u of layer){const s=best(i,u),edges=[];
   if(i<n)for(const j of opts(i,u)){const nu=advance(u,j);next.set(nu.join(','),nu);const tail=best(i+1,nu),obj=[tail.obj[0]+Number(j>=0),tail.obj[1].add(weights[i][j])];edges.push({expert:j,next:key(i+1,nu),objective:[obj[0],String(obj[1])],optimal:s.choices.includes(j)});}
   states.push({id:key(i,u),token:i,used:[...u],objective:[s.obj[0],String(s.obj[1])],count:String(s.count),edges});}layer=[...next.values()];}
 return {dispatched:optimal.obj[0],score:String(optimal.obj[1]),count:String(optimal.count),canonical,states};
}
function audit(payload){
 const {q,lower,upper,cap,eligible,n,m}=validate(payload),choices=eligible.map(row=>row.concat(-1));
 const nominal=dispatch(cap,eligible,q.map((row,i)=>Object.fromEntries(choices[i].map(j=>[j,j<0?zero():row[j]])))),a=nominal.canonical;
 const rows=choices.map((actions,i)=>actions.map(j=>({expert:j,...support(lower[i],upper[i],Array.from({length:m},(_,k)=>new Q(BigInt(Number(k===j)-Number(k===a[i])))))})));
 const weights=rows.map(row=>Object.fromEntries(row.map(r=>[r.expert,internalRat(r.value)]))),worst=dispatch(cap,eligible,weights),b=worst.canonical;
 const table=b.map((j,i)=>rows[i].find(r=>r.expert===j).point),mass=route=>sum(route.map((j,i)=>j<0?zero():internalRat(table[i][j]))),referenceMass=mass(a),challengerMass=mass(b);
 const units=(j,sign=1)=>Array.from({length:m},(_,k)=>new Q(BigInt(sign*Number(k===j))));
 const upperWeights=choices.map((actions,i)=>Object.fromEntries(actions.map(j=>[j,internalRat(support(lower[i],upper[i],units(j)).value)])));
 const separate=dispatch(cap,eligible,upperWeights),bestTable=separate.canonical.map((j,i)=>support(lower[i],upper[i],units(j)).point),minimum=a.map((j,i)=>support(lower[i],upper[i],units(j,-1)));
 const referenceMin=zero().sub(sum(minimum.map(r=>internalRat(r.value)))),bound=internalRat(separate.score).sub(referenceMin);
 if(challengerMass.sub(referenceMass).cmp(internalRat(worst.score))||bound.cmp(internalRat(worst.score))<0)throw Error('regret discrepancy');
 return {schema:'moelab-gate-regret-v1',input:{probabilities:q.map(row=>row.map(String)),capacities:cap,eligible,lower:lower.map(row=>row.map(String)),upper:upper.map(row=>row.map(String))},nominal,row_supports:rows,worst,adversarial_probabilities:table,reference_mass:String(referenceMass),challenger_mass:String(challengerMass),regret:worst.score,separate_extrema_bound:String(bound),separate_extrema_gap:String(bound.sub(internalRat(worst.score))),separate_extrema:{challenger:separate,best_probabilities:bestTable,reference_worst_probabilities:minimum.map(r=>r.point),best_mass:separate.score,reference_worst_mass:String(referenceMin)}};
}
const api={audit,validate,support,strictJSON};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.GateRegret=api;
})(typeof globalThis!=='undefined'?globalThis:this);
