'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),{spawnSync}=require('node:child_process'),{audit,strictJSON}=require('../docs/gate-engine.js');
const fixtures=JSON.parse(fs.readFileSync('moelab/route_data/gate-fixtures.json'));
test('all ten complete native/vertex-browser reports and data bindings',()=>{
 assert.equal(fixtures.length,10);for(const f of fixtures)assert.deepEqual(audit(f.input),f.report,f.id);
 const vm=require('node:vm'),ctx={};vm.runInNewContext(fs.readFileSync('docs/gate-data.js','utf8'),ctx);assert.deepEqual(JSON.parse(JSON.stringify(ctx.GATE_FIXTURES)),fixtures);
});
test('125 declared three-token inputs independently replay every browser report in native verifier',()=>{
 const inputs=[];for(let i=0;i<5;i++)for(let j=0;j<5;j++)for(let k=0;k<5;k++)inputs.push({probabilities:[i,j,k].map(x=>[x+'/4',(4-x)+'/4']),capacities:[1+(i%2),1+(j%2)],eligible:[[0,1],k%2?[0]:[0,1],[0,1]],lower:[[0,0],[0,0],[0,0]],upper:[[1,1],[1,1],[1,1]]});
 const data=inputs.map(input=>({input,report:audit(input)}));
 const proc=spawnSync(process.env.PYTHON||'python3',['-c',"import sys,json;from moelab.gate_verify import verify;print(json.dumps([verify(f['input'],f['report']) for f in json.load(sys.stdin)]))"],{input:JSON.stringify(data),encoding:'utf8',maxBuffer:30e6});assert.equal(proc.status,0,proc.stderr);assert.equal(JSON.parse(proc.stdout).filter(r=>r.verified).length,125);
});
test('generated 256-bit input fractions may yield larger exact internal objectives',()=>{
 const d=2n**255n-19n,e=2n**253n-7n;
 const p={probabilities:[['1/2','1/2'],['1/2','1/2']],capacities:[1,1],eligible:[[0,1],[0,1]],lower:[[(d/2n)+'/'+d,(d/2n)+'/'+d],[(e/2n)+'/'+e,(e/2n)+'/'+e]],upper:[[((d+1n)/2n)+'/'+d,((d+1n)/2n)+'/'+d],[((e+1n)/2n)+'/'+e,((e+1n)/2n)+'/'+e]]};
 const report=audit(p);assert.ok(report.regret.length>160);
 const proc=spawnSync(process.env.PYTHON||'python3',['-c',"import sys,json;from moelab.gate_verify import verify;x=json.load(sys.stdin);print(verify(x['input'],x['report'])['verified'])"],{input:JSON.stringify({input:p,report}),encoding:'utf8'});assert.equal(proc.status,0,proc.stderr);assert.equal(proc.stdout.trim(),'True');
});
test('strict input structures and decoded duplicate JSON keys',()=>{
 for(const text of ['{"x":0,"x":1}','{"x":0,"\\u0078":1}','NaN','{"x":1.0}','{"x":1e0}','['.repeat(50)+'0'+']'.repeat(50),'{} trailing'])assert.throws(()=>strictJSON(text));
 for(const p of [{},{...fixtures[0].input,capacities:[true,1,0]},{...fixtures[0].input,eligible:[[0,0]]},{...fixtures[0].input,lower:[[.3,.2,.3]]}])assert.throws(()=>audit(p));
});
