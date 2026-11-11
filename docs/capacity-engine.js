/* Exact finite capacity dispatch. Independent BigInt arithmetic, no native calls. */
(function(root){'use strict';
const gcd=(a,b)=>{a=a<0n?-a:a;while(b){[a,b]=[b,a%b];}return a;};
class Q{
 constructor(a,b=1n){if(!b)throw Error('zero denominator');if(b<0n){a=-a;b=-b;}const g=gcd(a,b);this.a=a/g;this.b=b/g;}
 static parse(x){if(Number.isSafeInteger(x))x=String(x);if(typeof x!=='string'||x.length>160||!(/^-?\d+(\/[1-9]\d*)?$/).test(x))throw Error('integer or rational string required');const s=x.split('/'),q=new Q(BigInt(s[0]),s[1]?BigInt(s[1]):1n);if(q.a.toString(2).replace('-','').length>256||q.b.toString(2).length>256)throw Error('rational exceeds 256 bits');return q;}
 add(q){return new Q(this.a*q.b+q.a*this.b,this.b*q.b);}sub(q){return new Q(this.a*q.b-q.a*this.b,this.b*q.b);}mul(q){return new Q(this.a*q.a,this.b*q.b);}div(q){return new Q(this.a*q.b,this.b*q.a);}cmp(q){const x=this.a*q.b-q.a*this.b;return x>0n?1:x<0n?-1:0;}toString(){return this.b===1n?String(this.a):this.a+'/'+this.b;}
}
const zero=()=>new Q(0n),sum=xs=>xs.reduce((a,b)=>a.add(b),zero());
function validate(p){
 if(!p||Array.isArray(p)||typeof p!=='object'||Object.keys(p).sort().join('|')!=='capacities|eligible|expert_values|probabilities|targets')throw Error('five required input fields only');
 const rows=p.probabilities;if(!Array.isArray(rows)||rows.length<1||rows.length>7||!Array.isArray(rows[0])||rows[0].length<2||rows[0].length>4)throw Error('1..7 tokens, 2..4 experts required');
 const n=rows.length,e=rows[0].length,vec=(r,len)=>{if(!Array.isArray(r)||r.length!==len)throw Error('dimension mismatch');return r.map(Q.parse);};
 const q=rows.map(r=>{r=vec(r,e);if(r.some(x=>x.cmp(zero())<0)||sum(r).cmp(new Q(1n)))throw Error('nonnegative rows sum exactly to one');return r;});
 const c=p.capacities;if(!Array.isArray(c)||c.length!==e||c.some(x=>!Number.isInteger(x)||x<0||x>n))throw Error('capacity integer in 0..tokens');
 if(!Array.isArray(p.eligible)||p.eligible.length!==n)throw Error('eligibility dimension');
 const a=p.eligible.map(r=>{if(!Array.isArray(r)||r.some(j=>!Number.isInteger(j)||j<0||j>=e)||new Set(r).size!==r.length)throw Error('distinct eligible expert indices');return [...r].sort((a,b)=>a-b);});
 if(!Array.isArray(p.expert_values)||p.expert_values.length!==n)throw Error('expert value dimensions');
 return {q,c:[...c],a,v:p.expert_values.map(r=>vec(r,e)),t:vec(p.targets,n),n,e};
}
function audit(payload){
 const {q,c,a,v,t,n,e}=validate(payload),memo=new Map(),key=(i,u)=>i+':'+u.join(','),advance=(u,j)=>u.map((x,k)=>x+Number(j===k)),opts=(i,u)=>a[i].filter(j=>u[j]<c[j]).concat(-1),compare=(x,y)=>x[0]-y[0]||x[1].cmp(y[1]);
 function solve(i,u){const k=key(i,u);if(memo.has(k))return memo.get(k);let result;
  if(i===n)result={obj:[0,zero()],count:1n,choices:[]};else{let best=null,count=0n,choices=[];
   for(const j of opts(i,u)){const tail=solve(i+1,advance(u,j)),obj=[tail.obj[0]+Number(j>=0),tail.obj[1].add(j>=0?q[i][j]:zero())],cmp=best?compare(obj,best):1;
    if(cmp>0){best=obj;count=tail.count;choices=[j];}else if(!cmp){count+=tail.count;choices.push(j);}}
   result={obj:best,count,choices};}memo.set(k,result);return result;}
 const start=Array(e).fill(0),optimal=solve(0,start),states=[];let layer=[start];
 for(let i=0;i<=n;i++){const next=new Map();layer.sort((a,b)=>{for(let k=0;k<e;k++)if(a[k]!==b[k])return a[k]-b[k];return 0;});
  for(const u of layer){const s=solve(i,u),edges=[];
   if(i<n)for(const j of opts(i,u)){const nu=advance(u,j);next.set(nu.join(','),nu);const tail=solve(i+1,nu),obj=[tail.obj[0]+Number(j>=0),tail.obj[1].add(j>=0?q[i][j]:zero())];edges.push({expert:j,next:key(i+1,nu),objective:[obj[0],String(obj[1])],optimal:s.choices.includes(j)});}
   states.push({id:key(i,u),token:i,used:[...u],objective:[s.obj[0],String(s.obj[1])],count:String(s.count),edges});}layer=[...next.values()];}
 const marginal=Array.from({length:n},()=>Array(e+1).fill(0n)),canonical=[];let cu=start,prefix=new Map([[start.join(','),{u:start,count:1n}]]);
 for(let i=0;i<n;i++){const j=solve(i,cu).choices[0];canonical.push(j);cu=advance(cu,j);const next=new Map();
  for(const {u,count}of prefix.values())for(const j of solve(i,u).choices){const nu=advance(u,j),k=nu.join(',');marginal[i][j<0?e:j]+=count*solve(i+1,nu).count;if(!next.has(k))next.set(k,{u:nu,count:0n});next.get(k).count+=count;}prefix=next;}
 function summary(assignment){const loads=Array.from({length:e},(_,j)=>assignment.filter(x=>x===j).length),pred=assignment.map((j,i)=>j<0?zero():q[i][j].mul(v[i][j])),mass=sum(assignment.map((j,i)=>j<0?zero():q[i][j]));
  return {assignment,loads,dispatched:loads.reduce((a,b)=>a+b,0),retained_probability:String(mass),weighted_updates:pred.map(String),update_mse:String(sum(pred.map((x,i)=>{const d=x.sub(t[i]);return d.mul(d);})).div(new Q(BigInt(n))))};}
 const errorMemo=new Map();function errorRange(i,u){const k=key(i,u);if(errorMemo.has(k))return errorMemo.get(k);if(i===n)return [{loss:zero(),route:[]},{loss:zero(),route:[]}];let low=null,high=null;for(const j of solve(i,u).choices){const pred=j<0?zero():q[i][j].mul(v[i][j]),d=pred.sub(t[i]),cost=d.mul(d),tail=errorRange(i+1,advance(u,j)),lo={loss:cost.add(tail[0].loss),route:[j,...tail[0].route]},hi={loss:cost.add(tail[1].loss),route:[j,...tail[1].route]};if(!low||lo.loss.cmp(low.loss)<0)low=lo;if(!high||hi.loss.cmp(high.loss)>0)high=hi;}const out=[low,high];errorMemo.set(k,out);return out;}
 const [lo,hi]=errorRange(0,start),error_range={minimum:String(lo.loss.div(new Q(BigInt(n)))),maximum:String(hi.loss.div(new Q(BigInt(n)))),minimum_witness:summary(lo.route),maximum_witness:summary(hi.route)};
 const confidence=Array.from({length:n},(_,i)=>i).sort((i,j)=>{const maximum=i=>a[i].reduce((r,j)=>q[i][j].cmp(r)>0?q[i][j]:r,zero());return maximum(j).cmp(maximum(i))||i-j;});
 function greedy(order,reroute){const u=Array(e).fill(0),assignment=Array(n).fill(-1);for(const i of order){let ranked=[...a[i]].sort((j,k)=>q[i][k].cmp(q[i][j])||j-k);if(!reroute)ranked=ranked.slice(0,1);const j=ranked.find(j=>u[j]<c[j]);if(j!==undefined){u[j]++;assignment[i]=j;}}return summary(assignment);}
 const hall=[];for(let mask=1;mask<2**n;mask++){const tokens=Array.from({length:n},(_,i)=>i).filter(i=>mask&(1<<i)),neighbors=Array.from({length:e},(_,j)=>j).filter(j=>tokens.some(i=>a[i].includes(j))),slots=neighbors.reduce((s,j)=>s+c[j],0);hall.push({tokens,neighbors,slots,deficiency:Math.max(0,tokens.length-slots)});}
 hall.sort((a,b)=>a.tokens.length-b.tokens.length||(()=>{for(let i=0;i<a.tokens.length;i++)if(a.tokens[i]!==b.tokens[i])return a.tokens[i]-b.tokens[i];return 0;})());
 const minimum_drops=Math.max(0,...hall.map(r=>r.deficiency));if(minimum_drops!==n-optimal.obj[0])throw Error('Hall discrepancy');
 return {schema:'moelab-capacity-v1',input:{probabilities:q.map(r=>r.map(String)),capacities:c,eligible:a,expert_values:v.map(r=>r.map(String)),targets:t.map(String)},objective:'maximum dispatched count, then maximum sum of original gate probabilities',
  optimum:{dispatched:optimal.obj[0],retained_probability:String(optimal.obj[1]),count:String(optimal.count),canonical:summary(canonical),marginal_counts:marginal.map(r=>r.map(String)),drop_column:e,update_mse_range:error_range},
  baselines:{ordered_top_choice:greedy(Array.from({length:n},(_,i)=>i),false),confidence_top_choice:greedy(confidence,false),ordered_best_available:greedy(Array.from({length:n},(_,i)=>i),true),confidence_best_available:greedy(confidence,true)},confidence_order:confidence,minimum_drops,hall,states};
}
function strictJSON(text){
 if(typeof text!=='string'||text.length>8000000)throw Error('input size');
 // JSON.parse alone loses duplicate object keys. Scan key tokens before parsing.
 let at=0;const ws=()=>{while(/\s/.test(text[at]||'')&&at<text.length)at++;};
 function string(){const start=at++;while(at<text.length){if(text[at]==='\\'){at+=2;continue;}if(text[at++]==='"')return JSON.parse(text.slice(start,at));}throw Error('unterminated string');}
 function value(depth){if(depth>48)throw Error('JSON depth exceeds 48');ws();const c=text[at];
  if(c==='"'){string();return;}if(c==='{'||c==='['){at++;ws();const close=c==='{'?'}':']',seen=new Set();if(text[at]===close){at++;return;}while(at<text.length){if(c==='{'){ws();if(text[at]!=='"')throw Error('expected key');const k=string();if(seen.has(k))throw Error('duplicate JSON key');seen.add(k);ws();if(text[at++]!==':')throw Error('expected colon');}value(depth+1);ws();if(text[at]===close){at++;return;}if(text[at++]!==',')throw Error('expected comma');}throw Error('unterminated collection');}
  const match=text.slice(at).match(/^(?:null|true|false|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/);if(!match)throw Error('invalid JSON value');if(/^-?\d/.test(match[0])&&/[.eE]/.test(match[0]))throw Error('use rational strings, not decimal or exponent JSON numbers');at+=match[0].length;
 }value(0);ws();if(at!==text.length)throw Error('trailing JSON');return JSON.parse(text);
}
const api={audit,validate,Q,strictJSON};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.Capacity=api;
})(typeof globalThis!=='undefined'?globalThis:this);
