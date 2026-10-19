"use strict";
const report = window.REPORT;
const $ = (selector) => document.querySelector(selector);
const number = (n) => Number(n).toLocaleString("zh-CN", {maximumFractionDigits: 2});
const clusterNames = ["华东边缘集群", "华南算力主集群", "香港低延迟专线", "华北异构集群"];
const rules = ["no_action", "flag_all", "age_only", "slots_only", "global_or", "cluster_policy"];
const labels = ["不干预", "全部标记", "仅年龄 >15 分钟", "仅槽位 ≤3", "全局 OR", "分集群规则"];
function renderRisks(kind) {
  const age = kind === "age", flag = age ? "stale_gt_15m" : "slots_le_3";
  const source = age ? report.ageGroups : report.slotGroups;
  $("#risk-title").textContent = age ? "健康快照超过 15 分钟后，哪些集群更危险？" : "槽位不足的风险，是否对所有集群都一样？";
  $("#legend-normal").textContent = age ? "年龄 ≤15 分钟" : "槽位 >3";
  $("#legend-risk").textContent = age ? "年龄 >15 分钟" : "槽位 ≤3";
  $("#risk-chart").replaceChildren(...clusterNames.map((cluster) => {
    const element = document.createElement("div"); element.className = "risk-row";
    const title = document.createElement("div");title.className="risk-label";title.textContent=cluster;
    const small=document.createElement("small");small.textContent="全量已知结果";title.append(small);
    const bars = document.createElement("div"); bars.className="risk-bars";
    for(const risk of [false,true]) {
      const row=source.find((d)=>d.service_cluster===cluster&&d[flag]===risk);
      const line=document.createElement("div");line.className="bar-line";
      const track=document.createElement("div");track.className="bar-track";
      const fill=document.createElement("div");fill.className="bar-fill "+(risk?"orange":"blue");fill.style.width=(row.failure_rate_known*100)+"%";track.append(fill);
      const text=document.createElement("span");text.className="bar-number";
      const strong=document.createElement("strong");strong.textContent=(row.failure_rate_known*100).toFixed(2)+"%";
      text.append(strong,`  ${row.failures}/${row.known_outcomes}`);
      line.append(track,text);bars.append(line);
    }
    element.append(title,bars);return element;
  }));
  $("#risk-insight").textContent=age ? "华东、华南与香港的旧状态组失败率明显更高；华北为 10.14% 对 11.41%，并未呈现同样方向。统一按状态年龄判断，会掩盖这种差异。" : "华北槽位 ≤3 时失败率为 65.85%（27/41），槽位 >3 时为 0.47%（1/212）。其余集群的分组差异远没有这么突出，不能直接把低槽位规则扩展到全部集群。";
  document.querySelectorAll("[data-risk]").forEach((button)=>button.setAttribute("aria-pressed",String(button.dataset.risk===kind)));
}
function renderPolicies(split) {
  const rows=report.comparisons.filter((d)=>d.split===split), chosen=rows.find((d)=>d.rule==="cluster_policy"), global=rows.find((d)=>d.rule==="global_or");
  $("#summary-alerts").textContent=`${chosen.alerts} / ${chosen.requests} 次`;
  $("#summary-recall").textContent=`${chosen.alerted_failures} / ${chosen.failures} 次`;
  $("#summary-reduction").textContent=`${global.alerts-chosen.alerts} 次`;
  const prefix={validation:"后段验证",train:"前段规则选择",all_descriptive:"全量描述，含规则选择样本"}[split];
  $("#cohort-description").textContent=`${prefix}：${chosen.requests} 次请求，${chosen.known_outcomes-chosen.failures} 次成功，${chosen.failures} 次失败，${chosen.unknown_outcomes} 次未知。`;
  $("#policy-table").replaceChildren(...rules.map((rule,i)=>{
    const row=rows.find((d)=>d.rule===rule),tr=document.createElement("tr");if(rule==="cluster_policy")tr.className="highlight";
    [labels[i],row.alerts,row.alerted_failures,row.alerted_successes,row.alerted_unknowns,number(row.scenario_loss_yuan)].forEach((value)=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});return tr;
  }));
  document.querySelectorAll("[data-split]").forEach((button)=>button.setAttribute("aria-pressed",String(button.dataset.split===split)));
}
function renderCost() {
  const q=Number($("#rescue").value)/100,r=Number($("#harm").value)/100;
  const row=report.comparisons.find((d)=>d.split==="validation"&&d.rule==="cluster_policy");
  const action=report.costs.action*row.alerts, f=report.costs.failure;
  const saving=f*(row.alerted_failures*q-row.alerted_successes*r)-action;
  const breakEven=(action/f+row.alerted_successes*r)/row.alerted_failures;
  $("#rescue-label").textContent=number(q*100)+"%";$("#harm-label").textContent=number(r*100)+"%";
  $("#saving").textContent=number(saving);$("#saving").style.color=saving<0?"#f2b39a":"#ffffff";
  $("#break-even").textContent=(breakEven*100).toFixed(2)+"%";
  $("#result-note").textContent=saving>=0?"在当前假设与动作费代理下，测算净收益为正。":"在当前假设下，挽回收益尚不足以覆盖干预成本。";
  const x=(n)=>45+n*420,y=(v)=>155-(v+1400)/5400*135;
  const value=(a)=>f*(row.alerted_failures*a-row.alerted_successes*r)-action;
  const path=Array.from({length:11},(_,i)=>`${i?"L":"M"}${x(i/10)},${y(value(i/10))}`).join(" ");
  const pointX=x(q), pointY=y(saving);
  $("#cost-chart").innerHTML=`<title>当前 r=${number(r*100)}% 时，不同 q 下的情景净收益；当前净收益 ${number(saving)} 元</title><line x1="45" y1="${y(0)}" x2="465" y2="${y(0)}" stroke="#647f91" stroke-dasharray="3 4"/><text x="4" y="${y(0)+4}" fill="#a8bdcc" font-size="10">0元</text><path d="${path}" fill="none" stroke="#83bcdb" stroke-width="2.5"/><circle cx="${pointX}" cy="${pointY}" r="5" fill="#f0b38d"/><line x1="${pointX}" y1="${pointY+8}" x2="${pointX}" y2="167" stroke="#9cb0be" stroke-dasharray="2 5"/><text x="45" y="188" fill="#a8bdcc" font-size="10">q = 0%</text><text x="426" y="188" fill="#a8bdcc" font-size="10">100%</text>`;
}
document.querySelectorAll("[data-risk]").forEach((b)=>b.addEventListener("click",()=>renderRisks(b.dataset.risk)));
document.querySelectorAll("[data-split]").forEach((b)=>b.addEventListener("click",()=>renderPolicies(b.dataset.split)));
document.querySelectorAll("[data-preset]").forEach((b)=>b.addEventListener("click",()=>{const[q,r]=b.dataset.preset.split(",");$("#rescue").value=q;$("#harm").value=r;renderCost();}));
$("#rescue").addEventListener("input",renderCost);$("#harm").addEventListener("input",renderCost);
$("#source-hash").textContent=report.audit.sha256;
renderRisks("age");renderPolicies("validation");renderCost();
const observer=new IntersectionObserver((entries)=>{const current=entries.filter((e)=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(current)document.querySelectorAll("nav a").forEach((a)=>a.classList.toggle("active",a.hash==="#"+current.target.id));},{rootMargin:"-5% 0px -45% 0px",threshold:[0,.1,.3]});
document.querySelectorAll("section[id]").forEach((s)=>observer.observe(s));
