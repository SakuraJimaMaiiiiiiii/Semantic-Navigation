"""Shared node inspection controls for standalone and stacked map pages."""

CONTROLS = '''<div class="graph-inspector">
<label>圆圈显示 <select data-radius-mode><option value="selected">仅查看的节点</option><option value="all">全部圆圈轮廓</option><option value="none">隐藏圆圈</option></select></label>
<label>半径类型 <select data-radius-kind><option value="free">当前自由半径</option><option value="explored">历史探索半径</option></select></label>
<label>边显示 <select data-edge-mode><option value="all">全部边</option><option value="selected">仅查看节点的邻边</option><option value="none">隐藏边</option></select></label>
<button type="button" data-clear-node>清除选择</button>
<span data-node-status>悬停或点击节点，查看自由半径和相邻连接。</span></div>'''

SCRIPT = '''function wireGraphInspector(svg, root) {
  const mode=root.querySelector('[data-radius-mode]');if(!mode)return;
  const kind=root.querySelector('[data-radius-kind]');
  let selected=null, hovered=null, selectedObject=root.dataset.selectedObject||null;
  const objects=Array.from(svg.querySelectorAll('[data-object-id]'));
  const associations=Array.from(svg.querySelectorAll('[data-object-edge]'));
  const nodes=Array.from(svg.querySelectorAll('[data-node-id]'));
  const circles=Array.from(svg.querySelectorAll('[data-node-radius]'));
  const edges=Array.from(svg.querySelectorAll('[data-edge-source]'));
  const remembered=nodes.find(n=>n.dataset.stableId===root.dataset.selectedStableId);
  if(remembered)selected=remembered.dataset.nodeId;
  const edgeMode=root.querySelector('[data-edge-mode]');
  if(edgeMode&&edges.length>5000)edgeMode.value='selected';
  function render(){
    const selectedNode=nodes.find(n=>n.dataset.nodeId===selected);
    root.dataset.selectedStableId=selectedNode?selectedNode.dataset.stableId:'';
    root.dataset.selectedObject=selectedObject||'';
    const active=hovered===null?selected:hovered;
    circles.forEach(c=>{c.setAttribute('r',kind&&kind.value==='explored'?c.dataset.exploredRadius:c.dataset.freeRadius);c.style.display=mode.value==='all'||(mode.value==='selected'&&c.dataset.nodeRadius===active)?'':'none';});
    const associated=new Set(associations.filter(e=>e.dataset.objectEdge===selectedObject).map(e=>e.dataset.navTarget));
    nodes.forEach(n=>{n.setAttribute('stroke',n.dataset.nodeId===active||associated.has(n.dataset.nodeId)?'#f59e0b':'none');n.setAttribute('stroke-width','.06');});
    edges.forEach(e=>{const near=e.dataset.edgeSource===active||e.dataset.edgeTarget===active;e.style.display=edgeMode&&(edgeMode.value==='none'||edgeMode.value==='selected'&&!near)?'none':'';e.setAttribute('stroke-opacity',active===null?'.45':near?'1':'.12');e.setAttribute('stroke-width',near&&active!==null?'.065':'.025');});
    objects.forEach(o=>o.setAttribute('stroke-width',o.dataset.objectId===selectedObject?'.12':'.04'));
    associations.forEach(e=>e.setAttribute('stroke-opacity',selectedObject===null?'.5':e.dataset.objectEdge===selectedObject?'1':'.08'));
    svg.querySelectorAll('[data-object-outline]').forEach(e=>e.setAttribute('stroke-width',e.dataset.objectOutline===selectedObject?'.10':'.045'));
    const node=nodes.find(n=>n.dataset.nodeId===active);
    root.querySelector('[data-node-status]').textContent=node?'节点 '+(node.dataset.stableId||active)+' · 前沿点 '+(node.dataset.frontierCount||0)+' · 历史探索半径 '+Number(node.dataset.exploredRadius).toFixed(2)+' m · 自由半径 '+Number(node.dataset.freeRadius).toFixed(2)+' m · 相邻连接 '+edges.filter(e=>e.dataset.edgeSource===active||e.dataset.edgeTarget===active).length+' 条'+(node.dataset.aboveVehicles?' · 车辆上方（满足当前包围盒净空）: '+node.dataset.aboveVehicles:''):'悬停或点击节点，查看自由半径和相邻连接。';
    const object=objects.find(o=>o.dataset.objectId===selectedObject);
    if(object && active===null)root.querySelector('[data-node-status]').textContent=object.dataset.objectInfo;
  }
  objects.forEach(o=>{o.style.cursor='pointer';o.addEventListener('click',()=>{selectedObject=o.dataset.objectId;selected=null;hovered=null;render();});});
  nodes.forEach(node=>{
    node.style.cursor='pointer';
    node.addEventListener('mouseenter',()=>{hovered=node.dataset.nodeId;render();});
    node.addEventListener('mouseleave',()=>{hovered=null;render();});
    node.addEventListener('click',()=>{selected=node.dataset.nodeId;selectedObject=null;render();});
  });
  mode.addEventListener('change',render);
  if(kind)kind.addEventListener('change',render);
  if(edgeMode)edgeMode.addEventListener('change',render);
  const clear=root.querySelector('[data-clear-node]');
  const clearSelection=()=>{selected=null;hovered=null;selectedObject=null;render();};
  clear.addEventListener('click',clearSelection);
  svg._disposeInspector=()=>{
    mode.removeEventListener('change',render);
    if(kind)kind.removeEventListener('change',render);
    if(edgeMode)edgeMode.removeEventListener('change',render);
    clear.removeEventListener('click',clearSelection);
  };
  render();
}'''
