'use strict';
const $ = id => document.getElementById(id);
const canvas = $('drawing');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const appState = { page:'recognize', version:0, busy:false, pending:false, ink:false, training:false, analysisImage:null, history:[], drawing:false, brush:18, analysisVersion:0 };
let predictionTimer, toastTimer, previousPoint;
const percent = n => (n * 100).toFixed(1) + '%';

async function api(path, data) {
  if(appState.closing && path!=='shutdown')throw new Error('服务正在关闭，请重新启动后使用。');
  const response = await fetch('/api/' + path, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  let result;
  try { result = await response.json(); } catch { throw new Error('服务返回异常，请确认后台程序仍在运行。'); }
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '参数不正确，请检查后再试。');
  return result;
}
function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6000);
}
function safeAction(fn) { return async (...args) => { try { await fn(...args); } catch(error) { toast(error.message); } }; }
function setPage(page) {
  appState.page = page;
  document.querySelectorAll('.page').forEach(el => { el.hidden = el.id !== 'page-' + page; });
  document.querySelectorAll('[data-page]').forEach(el => { el.classList.toggle('active', el.dataset.page === page); el.setAttribute('aria-current',el.dataset.page === page ? 'page' : 'false'); });
  $('breadcrumb').textContent = {recognize:'手写识别',training:'模型训练',analysis:'模型探索'}[page];
  if(page === 'analysis' && appState.ink && !appState.analysisImage) setAnalysisImage(canvas.toDataURL('image/png'));
  if(page === 'training') pollTraining();
}
document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click',()=>setPage(button.dataset.page)));
function imageElement(src, alt) { const image = new Image(); image.src = src; image.alt = alt; return image; }

function renderResult(data) {
  const blank = data.blank || data.prediction === null;
  $('answer').textContent = blank ? '?' : String(data.prediction);
  $('answer-state').textContent = blank ? '等待输入' : '模型预测';
  const p = blank ? 0 : data.probabilities[data.prediction];
  $('confidence').textContent = blank ? '写下你的第一个数字' : '置信度 ' + percent(p);
  const order = data.probabilities.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p);
  $('answer-tip').textContent = blank ? '模型的判断会出现在这里' : p < .6 ? '笔迹可能有歧义，试着再写清楚一些' : '次选 ' + order[1].i + ' · ' + percent(order[1].p);
  $('inference-time').textContent = blank ? '—' : data.elapsed_ms + ' ms';
  $('prediction-status').textContent = blank ? '准备就绪，等待笔迹' : '识别完成 · 仅在本机处理';
  $('probabilities').replaceChildren();
  data.probabilities.forEach((p,i)=>{
    const row = document.createElement('div'); row.className = 'probability-row' + (!blank && data.prediction === i ? ' winner' : '');
    const label = document.createElement('span'); label.className = 'probability-label'; label.textContent = i;
    const track = document.createElement('div'); track.className = 'probability-track';
    const fill = document.createElement('div'); fill.className = 'probability-fill'; fill.style.width = (p*100) + '%'; track.append(fill);
    const value = document.createElement('span'); value.className = 'probability-value'; value.textContent = blank ? '—' : percent(p);
    row.append(label, track, value); $('probabilities').append(row);
  });
  $('input-preview').replaceChildren();
  if(data.input) $('input-preview').append(imageElement(data.input,'模型实际输入的 28 × 28 图像'));
  else { const span=document.createElement('span'); span.textContent='28'; $('input-preview').append(span); }
}
function clearCanvas() {
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,320,320);
  appState.ink=false; $('canvas-guide').hidden=false;
}
function remember() {
  appState.history.push({image:ctx.getImageData(0,0,320,320),ink:appState.ink,caption:$('result-caption').textContent});
  if(appState.history.length>20) appState.history.shift();
  $('undo').disabled=false;
}
function schedulePrediction() {
  appState.version++; clearTimeout(predictionTimer);
  predictionTimer=setTimeout(()=>requestPrediction(),140);
}
async function requestPrediction() {
  clearTimeout(predictionTimer);
  if(appState.drawing) { appState.pending=true; return; }
  if(!appState.ink) { renderResult({blank:true,prediction:null,probabilities:Array(10).fill(0)}); return; }
  if(appState.busy) { appState.pending=true; return; }
  appState.busy=true; appState.pending=false;
  const version=appState.version;
  $('prediction-status').textContent='正在识别…';
  try {
    const result=await api('predict',{image:canvas.toDataURL('image/png'),auto_center:true});
    if(version === appState.version) renderResult(result);
  } catch(error) { if(version === appState.version) { $('prediction-status').textContent='识别失败，请重试'; toast(error.message); } }
  finally { appState.busy=false; if(appState.pending || version !== appState.version) { appState.pending=false; requestPrediction(); } }
}
function point(event) { const rect=canvas.getBoundingClientRect(); return {x:(event.clientX-rect.left)*320/rect.width,y:(event.clientY-rect.top)*320/rect.height}; }
canvas.addEventListener('pointerdown',event=>{
  if(event.button !== 0 && event.pointerType==='mouse') return;
  event.preventDefault(); canvas.setPointerCapture(event.pointerId); remember();
  appState.drawing=true; appState.version++; previousPoint=point(event);
  ctx.fillStyle='#22263d'; ctx.strokeStyle='#22263d'; ctx.lineWidth=appState.brush; ctx.lineCap='round'; ctx.lineJoin='round';
  ctx.beginPath(); ctx.arc(previousPoint.x,previousPoint.y,appState.brush/2,0,Math.PI*2); ctx.fill();
  appState.ink=true; $('canvas-guide').hidden=true;
});
canvas.addEventListener('pointermove',event=>{
  if(!appState.drawing) return;
  const current=point(event); ctx.beginPath(); ctx.moveTo(previousPoint.x,previousPoint.y); ctx.lineTo(current.x,current.y); ctx.stroke(); previousPoint=current;
});
function endStroke(event) { if(!appState.drawing)return; appState.drawing=false; if(canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); schedulePrediction(); }
canvas.addEventListener('pointerup',endStroke); canvas.addEventListener('pointercancel',endStroke);
$('brush').addEventListener('input',event=>{appState.brush=Number(event.target.value); $('brush-value').textContent=appState.brush;});
$('clear').addEventListener('click',()=>{if(appState.ink)remember();clearCanvas();$('result-caption').textContent='模型对每个候选数字的判断';schedulePrediction();});
$('undo').addEventListener('click',()=>{
  const saved=appState.history.pop(); if(!saved)return;
  ctx.putImageData(saved.image,0,0); appState.ink=saved.ink; $('result-caption').textContent=saved.caption; $('canvas-guide').hidden=saved.ink; $('undo').disabled=!appState.history.length; schedulePrediction();
});
$('predict').addEventListener('click',()=>requestPrediction());
async function loadImage(src) { return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('无法加载图片，请换一张再试。'));img.src=src;}); }
async function loadSample(digit) {
  $('sample').disabled=true;
  try {
    const sample=await api('sample'+(digit===undefined?'':'?digit='+digit));
    const image=await loadImage(sample.image); remember();clearCanvas();ctx.drawImage(image,0,0,320,320);
    appState.ink=true;$('canvas-guide').hidden=true;appState.version++;
    $('result-caption').textContent='测试样本 · 真实数字 '+sample.label;
    await requestPrediction();
  } finally { $('sample').disabled=false; }
}
$('sample').addEventListener('click',safeAction(()=>loadSample()));
for(let i=0;i<10;i++){const btn=document.createElement('button');btn.className='digit-chip';btn.textContent=i;btn.setAttribute('aria-label','尝试数字 '+i+' 的测试样本');btn.addEventListener('click',safeAction(()=>loadSample(i)));$('digit-shortcuts').append(btn);}
canvas.addEventListener('pointerdown',()=>{$('result-caption').textContent='模型对每个候选数字的判断';});
$('to-analysis').addEventListener('click',()=>{if(appState.ink)setAnalysisImage(canvas.toDataURL('image/png'));setPage('analysis');});

async function refreshStatus() {
  if(appState.closing)return;
  try {
    const status=await api('status'); $('connection').classList.remove('offline');
    $('connection').lastElementChild.textContent=status.ready?'模型已就绪 · '+status.device:'尚未加载模型';
    $('model-accuracy').replaceChildren(document.createTextNode(status.validation_accuracy===null?'—':percent(status.validation_accuracy)));
    const small=document.createElement('small');small.textContent='历史训练记录';$('model-accuracy').append(small);
    $('device-label').textContent=status.device+' 推理';
  } catch { $('connection').classList.add('offline');$('connection').lastElementChild.textContent='服务未连接'; }
}

function renderChart(container,series,isAccuracy=false) {
  const width=640,height=180,left=38,right=15,top=15,bottom=28;
  const values=series.flatMap(s=>s.values);const max=isAccuracy?100:Math.max(.1,...values)*1.15;
  const count=Math.max(1,...series.map(s=>s.values.length));
  const x=i=>left+(count===1?.5:i/(count-1))*(width-left-right);
  const y=v=>top+(1-v/max)*(height-top-bottom);
  const ns='http://www.w3.org/2000/svg'; const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 '+width+' '+height);svg.setAttribute('role','img');svg.setAttribute('aria-label',isAccuracy?'训练与验证准确率曲线':'训练与验证损失曲线');
  function el(name,attrs,text){const n=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.append(n);return n;}
  for(let i=0;i<4;i++){const value=max*i/3,yy=y(value);el('line',{x1:left,x2:width-right,y1:yy,y2:yy,stroke:'#edf0f6','stroke-width':1});el('text',{x:left-8,y:yy+3,'text-anchor':'end',fill:'#b5bbcb','font-size':9},isAccuracy?Math.round(value)+'%':value.toFixed(2));}
  if(!values.length)el('text',{x:width/2,y:height/2,'text-anchor':'middle',fill:'#bac0d0','font-size':11},'训练开始后，曲线将在这里呈现');
  const every=Math.max(1,Math.ceil(count/10));for(let i=0;i<count;i+=every)el('text',{x:x(i),y:height-9,'text-anchor':'middle',fill:'#b5bbcb','font-size':9},i+1);
  series.forEach(s=>{if(!s.values.length)return;el('polyline',{points:s.values.map((v,i)=>x(i)+','+y(v)).join(' '),fill:'none',stroke:s.color,'stroke-width':2.3,'stroke-linejoin':'round'});s.values.forEach((v,i)=>el('circle',{cx:x(i),cy:y(v),r:2.6,fill:s.color}));});
  container.replaceChildren(svg);
}
function renderTraining(state) {
  appState.training=state.running;$('start-training').disabled=state.running;$('stop-training').disabled=!state.running;
  $('training-form').querySelectorAll('input,select').forEach(el=>el.disabled=state.running);
  $('training-status').textContent=state.status;$('training-log').textContent=state.log||'训练开始后，日志将显示在这里。';$('csv-download').hidden=!state.csv;
  const h=state.history||{},count=(h.train_loss||[]).length;
  $('epoch-progress').textContent=count+' / '+state.epochs;$('epoch-count').textContent=count;
  $('train-loss').textContent=count?h.train_loss[count-1].toFixed(4):'—';$('val-accuracy').textContent=count?percent(h.val_acc[count-1]):'—';
  renderChart($('loss-chart'),[{values:h.train_loss||[],color:'#8574f1'},{values:h.val_loss||[],color:'#5dbda7'}]);
  renderChart($('accuracy-chart'),[{values:(h.train_acc||[]).map(n=>n*100),color:'#8574f1'},{values:(h.val_acc||[]).map(n=>n*100),color:'#5dbda7'}],true);
}
let pollingTraining=false;
async function pollTraining(){if(pollingTraining)return;pollingTraining=true;try{const previous=appState.training;renderTraining(await api('train/status'));if(previous&&!appState.training)refreshStatus();}catch(error){if(appState.training)toast(error.message);}finally{pollingTraining=false;}}
$('epochs').addEventListener('input',e=>{$('epochs-value').textContent=e.target.value+' 轮';});
$('training-form').addEventListener('submit',async event=>{
  event.preventDefault();if(appState.training)return;
  const data=Object.fromEntries(new FormData(event.currentTarget));['epochs','batch_size','lr'].forEach(k=>data[k]=Number(data[k]));
  $('start-training').disabled=true;
  try{await api('train/start',data);await pollTraining();}catch(error){$('start-training').disabled=false;toast(error.message);}
});
$('stop-training').addEventListener('click',safeAction(async()=>{const result=await api('train/stop',{});$('training-status').textContent=result.status;$('stop-training').disabled=true;}));
const trainingInterval=setInterval(()=>{if(appState.training||appState.page==='training')pollTraining();},1500);

function setAnalysisImage(src){appState.analysisVersion++; $('analysis-answer').hidden=true; $('confusion-result').hidden=true; $('network-summary').textContent='分析后显示模型结构。'; const empty=document.createElement('div');empty.className='empty-analysis';empty.textContent='图片已更新，开始分析以查看特征。';$('feature-maps').replaceChildren(empty);appState.analysisImage=src;$('analysis-source').replaceChildren(imageElement(src,'用于分析的数字图片'));$('analysis-status').textContent='图片已就绪，点击开始分析。';}
$('use-drawing').addEventListener('click',()=>{if(!appState.ink){toast('请先在手写识别页面写一个数字。');return;}setAnalysisImage(canvas.toDataURL('image/png'));});
$('analysis-sample').addEventListener('click',safeAction(async()=>{const sample=await api('sample');setAnalysisImage(sample.image);$('analysis-status').textContent='测试样本 · 真实数字 '+sample.label;}));
$('upload').addEventListener('change',safeAction(async event=>{
  const file=event.target.files[0];if(!file)return;
  if(file.size>10*1024*1024)throw new Error('图片过大，请使用小于 10 MB 的图片。');
  const src=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('图片读取失败'));reader.readAsDataURL(file);});
  const img=await loadImage(src);const temp=document.createElement('canvas'),scale=Math.min(1,800/Math.max(img.width,img.height));temp.width=Math.max(1,Math.round(img.width*scale));temp.height=Math.max(1,Math.round(img.height*scale));const context=temp.getContext('2d');context.fillStyle='white';context.fillRect(0,0,temp.width,temp.height);context.drawImage(img,0,0,temp.width,temp.height);setAnalysisImage(temp.toDataURL('image/png'));
}));
$('analyze').addEventListener('click',safeAction(async()=>{
  if(!appState.analysisImage){const sample=await api('sample');setAnalysisImage(sample.image);}
  const analysisVersion=appState.analysisVersion;
  $('analyze').disabled=true;$('analysis-status').textContent=$('confusion').checked?'正在提取特征并评估 1,000 张样本…':'正在提取卷积特征…';
  try{
    const result=await api('analyze',{image:appState.analysisImage,channels:Number($('channels').value),confusion:$('confusion').checked});
    if(analysisVersion!==appState.analysisVersion)return;
    $('analysis-answer').hidden=false;$('analysis-answer').replaceChildren();const digit=document.createElement('strong');digit.textContent=result.prediction;const info=document.createElement('span');info.textContent='模型预测 · 置信度 '+percent(result.probabilities[result.prediction]);$('analysis-answer').append(digit,info);
    $('feature-maps').replaceChildren();result.features.forEach((src,i)=>{const item=document.createElement('div');item.className='feature-item';item.append(imageElement(src,'卷积通道 '+(i+1)));const label=document.createElement('span');label.textContent='通道 '+String(i+1).padStart(2,'0');item.append(label);$('feature-maps').append(item);});
    $('network-summary').textContent=result.summary;$('confusion-result').hidden=!result.confusion;
    if(result.confusion){const {matrix,accuracy,samples}=result.confusion;$('confusion-caption').textContent='行：真实数字 · 列：预测数字 · '+samples+' 张均衡测试样本 · 准确率 '+percent(accuracy);$('confusion-matrix').replaceChildren();const max=Math.max(1,...matrix.flat());for(let row=-1;row<10;row++)for(let col=-1;col<10;col++){const cell=document.createElement('div');if(row===-1||col===-1){cell.className='matrix-label';cell.textContent=row===-1&&col===-1?'':row===-1?col:row;}else{const n=matrix[row][col];cell.textContent=n;cell.style.background='rgba(124,105,228,'+(.04+.76*n/max)+')';if(n>max*.6)cell.style.color='white';cell.title='真实 '+row+' → 预测 '+col+'：'+n+' 张';}$('confusion-matrix').append(cell);}}
    $('analysis-status').textContent='分析完成 · '+result.features.length+' 个卷积通道';
  }catch(error){if(analysisVersion===appState.analysisVersion)$('analysis-status').textContent='分析失败，请重试。';throw error;}finally{$('analyze').disabled=false;}
}));
clearCanvas();renderResult({blank:true,prediction:null,probabilities:Array(10).fill(0)});renderTraining({running:false,status:'准备就绪 · 尚未开始训练',history:{},epochs:5});refreshStatus();pollTraining();
const statusInterval=setInterval(refreshStatus,30000);

$('shutdown-service').addEventListener('click',()=>{
  $('shutdown-message').textContent=appState.training?'正在训练的任务会先停止并保存，随后退出服务。再次双击 start.bat 即可启动。':'关闭后，网页将无法继续识别。再次双击 start.bat 即可启动。';
  $('shutdown-dialog').showModal();
});
$('shutdown-cancel').addEventListener('click',()=>$('shutdown-dialog').close());
$('shutdown-dialog').addEventListener('cancel',event=>{if(appState.closing)event.preventDefault();});
$('shutdown-confirm').addEventListener('click',async()=>{
  $('shutdown-confirm').disabled=true;
  try{
    await api('shutdown',{});
    appState.closing=true;appState.version++;
    clearInterval(trainingInterval);clearInterval(statusInterval);clearTimeout(predictionTimer);
    document.querySelector('main').inert=true;document.querySelector('aside').inert=true;
    $('shutdown-cancel').hidden=true;
    $('shutdown-title').textContent='正在关闭服务…';
    $('shutdown-message').textContent=appState.training?'正在等待当前批次结束并保存模型，请稍候。':'正在释放本机服务，请稍候。';
    $('connection').lastElementChild.textContent='正在关闭服务';
    $('shutdown-confirm').textContent='正在关闭…';
    // A successful response only acknowledges shutdown; wait for the port to stop responding.
    const monitor=setInterval(async()=>{
      try{await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(3000)});}
      catch(error){
        if(error.name==='TimeoutError')return;
        clearInterval(monitor);
        $('shutdown-title').textContent='服务已关闭';
        $('shutdown-message').textContent='可以关闭此网页。下次双击 start.bat，服务就绪后会自动打开网页。';
        $('connection').classList.add('offline');$('connection').lastElementChild.textContent='服务已关闭';
        $('shutdown-confirm').textContent='已关闭';
      }
    },1000);
  }catch(error){$('shutdown-confirm').disabled=false;toast(error.message);}
});
