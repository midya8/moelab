'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn,spawnSync}=require('node:child_process'),{waitForDebugger}=require('./chrome_support.js');const fixtures=JSON.parse(fs.readFileSync(path.resolve(__dirname,'../moelab/route_data/gate-fixtures.json'))),delay=ms=>new Promise(r=>setTimeout(r,ms));
test('actual Chrome gate inspector: tables, row proofs, three full searches, strict edits and native downloads',{timeout:180000},async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'moe-gate-')),downloads=fs.mkdtempSync(path.join(os.tmpdir(),'moe-gate-download-'));const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,diagnostics='';browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
 for(let i=0;i<1200&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),diagnostics);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0],tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`);ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(Error(JSON.stringify(m.error))):res(m.result);}};const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await call('Page.enable');await call('Page.navigate',{url:process.env.GATE_URL||'file://'+path.resolve(__dirname,'../docs/gate.html')});
 for(let i=0;i<300;i++){if(await js('Boolean(globalThis.GATE_CURRENT)'))break;await delay(20);}assert.equal(await js('GATE_CURRENT.schema'),'moelab-gate-regret-v1');
 let tables=0,rowProofs=0,states=0;
 for(const f of fixtures){
  await js(`document.getElementById('fixture').value=${JSON.stringify(f.id)};document.getElementById('fixture').dispatchEvent(new Event('change'))`);assert.deepEqual(await js('GATE_CURRENT'),f.report,f.id);
  assert.equal(await js(`document.getElementById('regret').dataset.exact`),f.report.regret);assert.equal(await js(`document.getElementById('bound').dataset.exact`),f.report.separate_extrema_bound);
  for(const kind of ['shared','best','minimum']){
   await js(`document.getElementById('table-kind').value=${JSON.stringify(kind)};document.getElementById('table-kind').dispatchEvent(new Event('change'))`);
   const table=kind==='shared'?f.report.adversarial_probabilities:kind==='best'?f.report.separate_extrema.best_probabilities:f.report.separate_extrema.reference_worst_probabilities;
   assert.deepEqual(await js(`Array.from(document.querySelectorAll('#allocation tbody tr')).map(r=>Array.from(r.querySelectorAll('.gate-value')).map(c=>c.dataset.exact))`),table);
   const routes=await js(`Array.from(document.querySelectorAll('#routes tbody tr')).map(r=>[r.cells[3].dataset.exact,r.cells[4].dataset.exact])`);
   routes.forEach((row,i)=>assert.deepEqual(row,[f.report.nominal.canonical[i]<0?'0':table[i][f.report.nominal.canonical[i]],f.report.worst.canonical[i]<0?'0':table[i][f.report.worst.canonical[i]]]));tables++;
  }
  for(let i=0;i<f.report.row_supports.length;i++)for(const support of f.report.row_supports[i]){
   await js(`document.getElementById('token').value=${i};document.getElementById('token').dispatchEvent(new Event('change'));document.getElementById('action').value=${support.expert};document.getElementById('action').dispatchEvent(new Event('change'))`);
   const cells=await js(`Array.from(document.querySelectorAll('#support tbody tr')).map(r=>Array.from(r.cells).slice(1).map(c=>c.dataset.exact))`);
   cells.forEach((row,j)=>assert.deepEqual(row,[f.report.input.lower[i][j],f.report.input.upper[i][j],support.coefficient[j],support.point[j],support.dual_terms[j]]));rowProofs++;
  }
  for(const search of ['nominal','worst','separate']){
   const expected=search==='separate'?f.report.separate_extrema.challenger:f.report[search];
   await js(`document.getElementById('search').value=${JSON.stringify(search)};document.getElementById('search').dispatchEvent(new Event('change'))`);
   for(const s of expected.states){
    await js(`document.getElementById('state').value=${JSON.stringify(s.id)};document.getElementById('state').dispatchEvent(new Event('change'))`);
    const edges=await js(`Array.from(document.querySelectorAll('#edges tbody tr')).map(r=>[r.cells[1].textContent,r.cells[2].textContent,r.cells[3].dataset.exact||null,r.cells[4].textContent])`);
    assert.deepEqual(edges,s.edges.length?s.edges.map(e=>[e.next,String(e.objective[0]),e.objective[1],e.optimal?'Yes':'No']):[['No next action','0',null,'Complete']]);states++;
   }
  }
 }
 await js(`document.getElementById('fixture').value='simplex_gap';document.getElementById('fixture').dispatchEvent(new Event('change'));document.getElementById('input').focus()`);
 await call('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',modifiers:2,windowsVirtualKeyCode:65});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',modifiers:2,windowsVirtualKeyCode:65});await call('Input.insertText',{text:'{"bad":true}'});
 assert.equal(await js('GATE_CURRENT'),null);assert.equal(await js(`document.getElementById('download-report').disabled`),true);await js(`document.getElementById('apply').click()`);assert.match(await js(`document.getElementById('status').textContent`),/Cannot replay/);await js(`document.getElementById('restore').click()`);assert.deepEqual(await js('GATE_CURRENT'),fixtures[0].report);
 for(const input of ['{"x":0,"\\u0078":1}','NaN','{"x":1.0}']){await js(`document.getElementById('input').value=${JSON.stringify(input)};document.getElementById('apply').click()`);assert.equal(await js('GATE_CURRENT'),null);}await js(`document.getElementById('restore').click()`);
 const custom=structuredClone(fixtures[0].input);custom.capacities=[0,0,0];await js(`document.getElementById('input').value=${JSON.stringify(JSON.stringify(custom))};document.getElementById('apply').click()`);assert.equal(await js('GATE_CURRENT.regret'),'0');assert.equal(await js('GATE_CURRENT.nominal.dispatched'),0);await js(`document.getElementById('restore').click()`);
 await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads});await js(`document.getElementById('download-input').click();document.getElementById('download-report').click()`);
 const input=path.join(downloads,'moelab-gate-simplex_gap-input.json'),report=path.join(downloads,'moelab-gate-simplex_gap-report.json');
 for(let i=0;i<200;i++){if(fs.existsSync(input)&&fs.existsSync(report))break;await delay(25);}assert.deepEqual(JSON.parse(fs.readFileSync(input)),fixtures[0].report.input);assert.deepEqual(JSON.parse(fs.readFileSync(report)),fixtures[0].report);
 const check=spawnSync(process.env.PYTHON||'python3',['-m','moelab.gate_cli','verify',input,report],{encoding:'utf8'});assert.equal(check.status,0,check.stderr);assert.equal(JSON.parse(check.stdout).verified,true);
 const screens=process.env.GATE_SCREEN_DIR;if(screens)fs.mkdirSync(screens,{recursive:true});let layouts=0;
 for(const width of [320,390,1280])for(const f of fixtures){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await js(`document.getElementById('fixture').value=${JSON.stringify(f.id)};document.getElementById('fixture').dispatchEvent(new Event('change'));document.getElementById('table-kind').value='shared';document.getElementById('table-kind').dispatchEvent(new Event('change'));document.activeElement?.blur();window.scrollTo(0,0)`);await delay(30);assert.equal(await js('document.documentElement.scrollWidth<=innerWidth'),true,'viewport '+f.id+width);assert.equal(await js(`Array.from(document.querySelectorAll('input,select,button')).every(e=>e.getBoundingClientRect().height>=40)`),true,'touch');if(screens&&['simplex_gap','shared_route','four_experts'].includes(f.id)){await js('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');const metrics=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:metrics.cssContentSize.height,scale:1}});fs.writeFileSync(path.join(screens,f.id+'-'+width+'.png'),Buffer.from(image.data,'base64'));}layouts++;}
 console.log(JSON.stringify({tables,rowProofs,allDispatchStates:states,layouts,realDownloads:2,independentNativeDownloadReplay:true}));
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(r=>browser.once('exit',r));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:100});await fs.promises.rm(downloads,{recursive:true,force:true,maxRetries:10,retryDelay:100});}
});
