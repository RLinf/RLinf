# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Local episode controls for Tianji HIL; no SDK or Pico input access."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request

from rlinf.tianji_shadow import control as monitor_control
from rlinf.tianji_shadow import settings as shadow_settings, transport as shadow
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Tianji HIL</title><style>
body{font:18px sans-serif;max-width:780px;margin:40px auto;padding:0 20px}
button{font:inherit;padding:12px;margin:5px}pre{white-space:pre-wrap}
#shadow_thresholds{margin-top:.7rem;max-width:32rem}
#shadow_thresholds table{border-collapse:collapse}
#shadow_thresholds th,#shadow_thresholds td{padding:.2rem .35rem;text-align:left}
#shadow_thresholds input{width:4.5rem;font:inherit}
#shadow_threshold_message{margin-left:.5rem}
</style><h1>Tianji HIL 训练</h1>
<p>开始每条任务前摆好场景。按 H 键进入人工接管模式，按 P 键交回 policy。
人工模式下只有按住 Pico 握持键时才会实际操控机械臂并产生 BC 目标；释放握持键时保持不动。</p>
<p>实机点击“开始本条”后先返回已保存的复位姿态，完成后才运行模型。
复位中按暂停或 H 会取消本次启动。Stage2 中复位失败或控制器故障会结束本条并显示原因：在采集器恢复 Position/EEF 后再点“开始本条”；Stage1 复位失败需处理原因后重新启动程序。</p>
<p>policy 的目标连续无法执行（通常是关节到达限位，例如 B 臂 J5）时会自动暂停并显示原因：按 H 用 Pico 把腕部转回一些后按 P 交回；或点“恢复 policy”重试；或结束本条返回默认位姿。</p>
<button onclick="send('start')">开始本条</button>
<button onclick="send('human_start')">H：开始人工接管</button>
<button onclick="send('human_stop')">P：交回 policy</button>
<button onclick="send('pause')">暂停 policy</button>
<button onclick="send('resume')">恢复 policy</button>
<button onclick="send('success')">成功，结束本条（奖励 1）</button>
<button onclick="send('failure')">失败，结束本条（奖励 0）</button>
<button onclick="send('discard')">丢弃本条（不进训练，自动复位）</button>
<button onclick="send('step_failure')">F：当前步失败（奖励 −1，机械臂照常执行）</button>
<button onclick="send('cancel_return')">取消结束后复位</button>
<p>H、P、F 也可以点击按钮执行。暂停在环境处理下一步时生效。需要立即停止运动时使用 Pico 握持键或原停止运动按钮。
结束后会尝试返回已保存的默认位姿，须释放 Pico 握持键；返回中可按暂停、H 或取消复位。
等待下一条“等待开始”，摆好场景再开始。这里的按钮不控制 LoopInfra 录制。</p>
<div id="subtasks" hidden><h2>Subtask</h2><p id="subtask_now"></p><div id="subtask_buttons"></div>
<p id="subtask_hint"></p></div>
<section id="shadow_monitor" hidden>
<h2>后台 Monitor <small id="shadow_version"></small></h2><p id="shadow_status"></p>
<button id="shadow_y" onclick="send('shadow_on')">Y 开启</button>
<button id="shadow_n" onclick="send('shadow_off')">N 关闭</button>
<fieldset id="monitor_control" hidden>
<legend>本条控制模式</legend>
<label for="monitor_version">Monitor 版本</label>
<select id="monitor_version" onchange="configureMonitorVersion()">
<option value="1116v1">1116v1</option>
<option value="Sub-16ft_H-16v1">Sub-16ft_H-16v1</option>
</select>
<label for="monitor_mode">控制</label>
<select id="monitor_mode" onchange="configureMonitor()">
<option value="human">人类全部负责</option>
<option value="semi">Monitor 半自动</option>
<option value="full">Monitor 全自动</option>
</select>
<label for="monitor_rule">切阶段规则</label>
<select id="monitor_rule" onchange="configureMonitor()">
<option value="two">双点</option><option value="single">单点</option>
</select>
<p id="monitor_action" role="status"></p>
<div id="monitor_correction" hidden>
<p>人工纠正阶段</p>
<div id="monitor_correction_buttons"></div>
<p id="monitor_correction_status" role="status"></p>
</div>
<button id="monitor_resume" type="button" hidden onclick="send('resume')">继续 Policy</button>
<button id="monitor_takeover" type="button" hidden onclick="send('human_start')">人工 H 接管</button>
<button id="monitor_handback" type="button" hidden onclick="send('human_stop')">交回 Policy（P）</button>
<button id="monitor_manual" type="button" hidden onclick="send('monitor_manual')">本条转人工模式</button>
</fieldset>
<fieldset id="shadow_thresholds" hidden>
<legend>阶段完成阈值</legend>
<table><thead><tr><th>规则</th><th>S1</th><th>S2</th><th>S3</th><th>S4</th></tr></thead>
<tbody id="shadow_threshold_rows"></tbody></table>
<p>等待开始时可保存，下一条生效。</p>
<button id="shadow_threshold_save" type="button" onclick="saveShadowThresholds(false)">保存阈值</button>
<button id="shadow_threshold_reset" type="button" onclick="saveShadowThresholds(true)">恢复当前版本初始值</button>
<span id="shadow_threshold_message" role="status"></span>
</fieldset>
</section>
<pre id="state"></pre><p id="error"></p>
<details><summary>连接与回执信息（排查问题）</summary><pre id="debug"></pre></details><script>
const names={waiting:'等待开始',running:'运行中',paused:'已请求暂停',ended:'本条结束',returning:'本条结束，正在返回默认位姿'};
const modes={waiting:'等待开始',resetting:'正在返回初始位姿',reset_failed:'复位失败',fault:'动作未完成，本条已结束',policy:'策略控制',human:'Pico 人工控制',human_latched:'人工接管模式（按握持键才能运动）',paused:'已暂停',hold:'等待新观测或新动作'};
let monitorSnapshot=null;
let subtaskCount=0, shadowWaiting=false, shadowToggle=false, monitorUpdating=false, thresholdDirty=false, thresholdSaving=false;
const thresholdFields={single:[],two:[]};
function showThresholds(m){
const group=document.getElementById('shadow_thresholds');
group.hidden=!m.thresholds;
if(!m.thresholds)return;
const tbody=document.getElementById('shadow_threshold_rows');
for(const [key,label] of [['single','单点'],['two','双点']]){
 if(!thresholdFields[key].length){
  const row=tbody.insertRow(), heading=document.createElement('th');
  heading.scope='row';heading.textContent=label;row.append(heading);
  for(let i=0;i<4;i++){
   const input=document.createElement('input');
   input.type='number';input.min='0';input.max='4';input.step='0.001';input.required=true;
   input.setAttribute('aria-label',label+' S'+(i+1));
   input.oninput=()=>{thresholdDirty=true;};
   row.insertCell().append(input);thresholdFields[key].push(input);
  }
 }
 thresholdFields[key].forEach((input,i)=>{
  input.disabled=!shadowWaiting||thresholdSaving;
  if(!thresholdDirty&&document.activeElement!==input)input.value=m.thresholds[key][i];
 });
}
for(const id of ['shadow_threshold_save','shadow_threshold_reset'])
 document.getElementById(id).disabled=!shadowWaiting||thresholdSaving;
}
async function saveShadowThresholds(reset){
if(!shadowWaiting||thresholdSaving)return;
const info=document.getElementById('shadow_threshold_message');
const thresholds={};
if(!reset){
 for(const [name,inputs] of Object.entries(thresholdFields)){
  if(inputs.some(input=>!input.reportValidity()))return;
  thresholds[name]=inputs.map(input=>input.valueAsNumber);
 }
}
thresholdSaving=true;
try{
 const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(reset?{command:'shadow_thresholds_reset'}:{command:'shadow_thresholds_save',thresholds})});
 const state=await response.json();
 if(!response.ok)throw new Error(state.error||'阈值保存失败');
 thresholdDirty=false;
 info.textContent='已保存，下一条生效';
}catch(e){info.textContent=e.message;}
finally{thresholdSaving=false;await refresh();}
}
async function correctMonitorStage(index){
if(!monitorSnapshot)return;
const current=monitorSnapshot;
await send('monitor_correct',index,{episode:current.episode,revision:current.revision,
 subtask_revision:current.subtask_revision});
}
async function configureMonitorVersion(){
if(monitorUpdating)return;
monitorUpdating=true;
await send('shadow_version',undefined,{monitor_version:document.getElementById('monitor_version').value});
thresholdDirty=false;
document.getElementById('shadow_threshold_message').textContent='';
monitorUpdating=false;
await refresh();
}
async function configureMonitor(){
if(monitorUpdating)return;
monitorUpdating=true;
await send('monitor_mode',undefined,{monitor_mode:document.getElementById('monitor_mode').value,
 monitor_rule:document.getElementById('monitor_rule').value});
monitorUpdating=false;
await refresh();
}
function showMonitorControl(s,m){
const c=m.control, box=document.getElementById('monitor_control');
box.hidden=!c;
if(!c)return;
const mode=document.getElementById('monitor_mode'), rule=document.getElementById('monitor_rule');
const version=document.getElementById('monitor_version');
if(!monitorUpdating)version.value=m.version;
version.disabled=!shadowWaiting||monitorUpdating||thresholdSaving;
for(const option of version.options)
 option.disabled=option.value!=='1116v1'&&!m.versions?.[option.value]?.ready;
if(!monitorUpdating){mode.value=c.mode;rule.value=c.rule;}
mode.disabled=!shadowWaiting||monitorUpdating;
rule.disabled=!shadowWaiting||monitorUpdating||c.mode==='human';
monitorSnapshot=s;
const correcting=s.phase==='paused'&&!!s.monitor_correction;
const pausedByMonitor=s.phase==='paused'&&!!c.alarm;
const correction=document.getElementById('monitor_correction');
correction.hidden=c.mode==='human';
const buttons=document.getElementById('monitor_correction_buttons');
if(!buttons.children.length){
 for(let i=1;i<=s.subtask_count;i++){
  const button=document.createElement('button');
  button.textContent='纠正为 subtask '+i;
  button.onclick=()=>correctMonitorStage(i);
  buttons.append(button);
 }
}
for(let i=0;i<buttons.children.length;i++){
 const button=buttons.children[i];
 button.disabled=!['running','paused'].includes(s.phase);
 button.setAttribute('aria-pressed',String(i+1===s.subtask));
}
document.getElementById('monitor_correction_status').textContent=s.monitor_correction?
 '已人工纠正为 subtask '+s.monitor_correction.stage+
 (s.human_latched?'，可人工操作后交回 Policy':correcting?'，请继续 Policy 或转人工':''):'';

document.getElementById('monitor_action').textContent=pausedByMonitor?'H（Monitor 暂停 Policy）':
 'subtask '+s.subtask;
document.getElementById('monitor_resume').hidden=!(pausedByMonitor||correcting||(s.phase==='paused'&&s.human_latched))||c.fault;
document.getElementById('monitor_takeover').hidden=!['running','paused'].includes(s.phase)||s.human_latched;
document.getElementById('monitor_handback').hidden=!['running','paused'].includes(s.phase)||!s.human_latched||(c.fault&&c.mode!=='human');
document.getElementById('monitor_manual').hidden=!c.fault||c.mode==='human'||(s.phase!=='paused'&&!s.human_latched);
}
function showShadow(s){
const box=document.getElementById('shadow_monitor'), m=s.monitor_shadow;
box.hidden=!m;
shadowWaiting=!!m&&s.phase==='waiting';
if(!m){shadowToggle=false;return;}
document.getElementById('shadow_version').textContent=m.version||'';
const models={loading:'加载中',ready:'已加载（GPU1）',error:'加载失败，可继续人工采集'};
document.getElementById('shadow_status').textContent=
  '模型状态：'+(models[m.state]||m.state)+'　'+(s.phase==='waiting'?'下一条':'本条')+'：'+(m.selected?'启用':'关闭')+
  (m.error?' · '+m.error:'')+(m.settings_error?' · '+m.settings_error:'')+(m.variant_error?' · '+m.variant_error:'')+(m.control?.fault?' · Monitor 离线':'');
shadowToggle=shadowWaiting&&(!m.control||m.control.mode==='human');
for(const id of ['shadow_y','shadow_n'])document.getElementById(id).disabled=!shadowToggle;
showMonitorControl(s,m);
showThresholds(m);
}
function showSubtasks(s){const n=s.subtask_count||0;document.getElementById('subtasks').hidden=!n;if(!n)return;
document.getElementById('subtask_now').textContent='当前第 '+s.subtask+' / '+n+' 个：'+s.subtask_prompt+
 (s.monitor_shadow?.control?.s4_candidate?' · S4 完成候选，等待人工结束':'');
if(n!==subtaskCount){subtaskCount=n;const box=document.getElementById('subtask_buttons');box.textContent='';
s.subtask_prompts.forEach((p,i)=>{const b=document.createElement('button');b.textContent=(i+1)+'：'+p;
b.onclick=()=>send('subtask',i+1);box.appendChild(b);});}
document.getElementById('subtask_hint').textContent=s.run_mode==='training'
?'训练时按下一个编号（或 N 键）表示上一个 subtask 已完成（奖励 1）；每个 subtask 每条只奖励一次。按更小的编号回退到那个 subtask（不给奖励，要罚分再按 F）；回退后可直接按回本条已到过的编号（不给奖励）；最后一个完成后按“成功”。'
:'数字键 1–'+Math.min(n,9)+' 直接切换到对应 subtask，N 键切到下一个。';}
async function refresh(){try{const s=await (await fetch('/api/status')).json();
const phase=(s.phase==='paused'&&s.control_mode==='paused'?'已暂停':names[s.phase])+(s.outcome==='discard'?'（已丢弃，不进训练）':'');
document.getElementById('state').textContent=phase+' | 第 '+s.episode+' 条 | 已采集 '+(s.step||0)+' 步'+(s.step_failures?' | 本条已标记失败 '+s.step_failures+' 步':'')+'\\n'+(modes[s.control_mode]||'等待训练环境');
showSubtasks(s);showShadow(s);
document.getElementById('debug').textContent=JSON.stringify(s,null,2);
document.getElementById('error').textContent=s.reset_error||(s.episode_error ? '本条因动作未完成而结束：'+s.episode_error+'。在采集器恢复 Position/EEF 后点“开始本条”。' : '')||s.limit_error||s.last_return_error||(s.last_receipt_reason ? '动作未执行：'+s.last_receipt_reason : '');
}catch(e){document.getElementById('state').textContent='训练环境未连接';}}
async function send(command,index,extra={}){try{const r=await fetch('/api/control',{
method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({command,index,...extra})});
const s=await r.json();document.getElementById('error').textContent=s.error||'';await refresh();
}catch(e){document.getElementById('error').textContent=String(e);}}
document.addEventListener('keydown',e=>{if(e.repeat||e.target.closest?.('#shadow_thresholds, #monitor_control select'))return;const key=e.key.toLowerCase();
if(key==='h')send('human_start');else if(key==='p')send('human_stop');
else if(key==='f'&&document.querySelector('button[onclick*=step_failure]'))send('step_failure');
else if(shadowToggle&&key==='y')send('shadow_on');
else if(shadowToggle&&key==='n')send('shadow_off');
else if(subtaskCount&&key==='n')send('next_subtask');
else if(subtaskCount&&/^[1-9]$/.test(key)&&Number(key)<=subtaskCount)send('subtask',Number(key));});
setInterval(refresh,500);refresh();</script></html>"""
# Training only: the deployment panels (Stage1, Stage2 inference) drop them.
_STEP_FAILURE_BUTTON = (
    "<button onclick=\"send('step_failure')\">F：当前步失败（奖励 −1，机械臂照常执行）</button>\n"
)
_DISCARD_BUTTON = "<button onclick=\"send('discard')\">丢弃本条（不进训练，自动复位）</button>\n"


_XENSE_FOOTER = """<footer>
<p id="xense-connected" role="status">Xense 在线：读取中</p>
</footer>
<script>
async function refreshXenseCount() {
  const label = document.getElementById('xense-connected');
  try {
    const response = await fetch('/api/xense-count');
    if (!response.ok) throw new Error('Xense status unavailable');
    const {connected} = await response.json();
    label.textContent = Number.isInteger(connected)
      ? 'Xense 在线：' + connected + ' / 2'
      : 'Xense 在线：状态暂不可用';
  } catch (_) {
    label.textContent = 'Xense 在线：状态暂不可用';
  } finally {
    setTimeout(refreshXenseCount, 2000);
  }
}
refreshXenseCount();
</script>"""


def _xense_connected_count() -> int | None:
    """Read the physical connection count from the local collector status."""
    url = os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780").rstrip("/")
    token = os.environ.get("LOOPINFRA_TOKEN")
    headers = {"Authorization": "Bearer " + token} if token else {}
    request = urllib.request.Request(url + "/api/collection", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            data = json.load(response)
        xense = data.get("xense")
        if not isinstance(xense, dict):
            return None
        if xense.get("enabled") is False:
            return 0
        sensors = xense.get("sensors")
        if not isinstance(sensors, dict):
            return None
        roles = ("left_gripper_1", "right_gripper_2")
        rows = [sensors.get(role) for role in roles]
        if any(not isinstance(row, dict) or type(row.get("connected")) is not bool for row in rows):
            return None
        return sum(row["connected"] for row in rows)
    except (OSError, ValueError, AttributeError):
        return None


class TianjiHILControl:
    """Thread-safe episode state exposed by the env's loopback HTTP panel.

    With subtask prompts, the panel also owns the current subtask (1-based).
    Training follows A1: only the next subtask is accepted, and each advance
    is one completed subtask (a pending milestone reward); Success is only
    accepted in the last subtask. Stage1 and Stage2 inference select freely.
    Training also accepts "step failure": the step being executed earns -1,
    and phase, revisions and the H latch stay as they are. "Discard" ends the
    episode like Failure (with the same return to the saved pose), and its
    data never reaches training (env.train.discard_episodes).
    """

    def __init__(self, mode="training", subtask_prompts=None, *,
                 stage_switch_replan=False, event_path=None):
        if mode not in {"training", "dagger", "stage1", "inference"}:
            raise ValueError("HIL mode must be training, dagger, stage1 or inference")
        self.mode = mode
        self._stage_switch_replan = stage_switch_replan
        self._event_path = event_path
        self._condition = threading.Condition()
        self._phase = "waiting"
        self._human_latched = False
        self._outcome = None
        self._revision = 0
        self._collection_revision = 0
        self._episode = 0
        self._closed = False
        self._return_cancelled = False
        self._runtime = {}
        # Identify the owning desktop launch; an older panel must not pass readiness.
        self._launch_id = (
            os.environ.get("TIANJI_STAGE2_LAUNCH_ID") if mode == "training" else None
        )
        self._server = None
        self._thread = None
        self._prompts = tuple(subtask_prompts or ())
        self._subtask = 1 if self._prompts else 0
        self._subtask_revision = 0
        self._milestones = 0
        self._subtask_reached = self._subtask  # highest subtask this episode: a milestone counts once
        self._step_failure = False
        self._step_failures = 0
        self._shadow_enabled = mode == "training" and len(self._prompts) == 4 and shadow.selected()
        self._monitor_version = shadow_settings.BASE_VERSION
        self._monitor_profiles = None
        self._monitor_settings_error = None
        if mode == "training" and shadow.ROOT and os.environ.get("TIANJI_SHADOW_VERSION") == "1116v1":
            self._monitor_profiles, self._monitor_settings_error = shadow_settings.read()
        self._monitor_control = (monitor_control.AutoControl(shadow.ROOT)
                                 if monitor_control.enabled() and self._monitor_profiles is not None else None)
        if self._monitor_control is not None:
            self._monitor_version = shadow_settings.FT_VERSION
            self._monitor_profiles, self._monitor_settings_error = shadow_settings.read(self._monitor_version)
        self._shadow_seq = 0
        self._review_start_ns = None

    def _shadow_event(self, name, **extra):
        if self._event_path is not None:
            try:
                row = dict(name=name, monotonic_ns=time.monotonic_ns(), episode=self._episode,
                           stage=self._subtask, stage_revision=self._subtask_revision,
                           control_revision=self._revision, **extra)
                with self._event_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                logging.getLogger(__name__).exception("Could not write HIL event")
        if self._shadow_enabled or (self.mode == "training" and shadow.REVIEW_ROOT is not None):
            if self._shadow_enabled:
                self._shadow_seq += 1
            if name == "start" and self._monitor_profiles is not None:
                extra["monitor_thresholds"] = shadow_settings.values(self._monitor_profiles)
                if self._monitor_control is not None:
                    extra["monitor_version"] = self._monitor_version
            if self._monitor_control is not None:
                extra.update(monitor_mode=self._monitor_control.active,
                             monitor_rule=self._monitor_control.rule, control_revision=self._revision)
            shadow.event(name, self, journal=self._shadow_enabled, seq=self._shadow_seq,
                         prompt=self._prompts[self._subtask - 1] if self._subtask else None, **extra)

    def _auto_event(self, name, decision, **extra):
        self._shadow_event(name, source="monitor", decision=decision, **extra)

    def _pause_for_monitor(self, decision, alarm):
        self._phase = "paused"
        self._revision += 1
        self._collection_revision += 1
        auto = self._monitor_control
        auto.reset()
        auto.alarm = alarm
        auto.assisted = True
        auto.notice = "Monitor 已暂停 Policy，等待人工操作"
        self._auto_event("monitor_pause", decision, control="paused", alarm=alarm)
        self._condition.notify_all()

    def monitor_owns_boundaries(self):
        with self._condition:
            return self._monitor_control is not None and self._monitor_control.active != "human"

    def apply_monitor_boundary(self):
        with self._condition:
            return self._monitor_control is not None and self._monitor_control.boundary(self)

    def snapshot(self):
        with self._condition:
            return dict(
                self._runtime,
                **({"stage2_launch_id": self._launch_id} if self._launch_id else {}),
                run_mode=self.mode,
                human_latched=self._human_latched,
                phase=self._phase,
                outcome=self._outcome,
                revision=self._revision,
                collection_revision=self._collection_revision,
                episode=self._episode,
                return_cancelled=self._return_cancelled,
                subtask=self._subtask,
                subtask_count=len(self._prompts),
                subtask_prompt=self._prompts[self._subtask - 1] if self._prompts else None,
                subtask_prompts=list(self._prompts),
                subtask_revision=self._subtask_revision,
                subtask_selectable_max=min(len(self._prompts), max(self._subtask+1,self._subtask_reached)),
                step_failures=self._step_failures,
            )

    def select_subtask(self, index, *, source='human', decision=None):
        """Bind the selected prompt to a new control revision in car deployment."""
        with self._condition:
            count = len(self._prompts)
            if not count:
                raise ValueError("No subtask prompts were configured at startup")
            if type(index) is not int or not 1 <= index <= count:
                raise ValueError(f"Subtask must be 1..{count}")
            old_revision, old_stage = self._subtask_revision, self._subtask
            old_high = self._subtask_reached
            if self._stage_switch_replan:
                if self._phase not in {"running", "paused"}:
                    raise ValueError("Subtasks advance only during a running episode")
                if index > max(self._subtask + 1, old_high):
                    raise ValueError("Advance one new subtask at a time")
                self._subtask_reached = max(index, old_high)
            if self.mode == "training":
                if self._phase not in {"running", "paused"}:
                    raise ValueError("Subtasks advance only during a running episode")
                # 11-07: after going back, any subtask already reached this episode can be
                # pressed directly (no reward); a new one still needs the one before it.
                reach = min(count, max(self._subtask + 1, self._subtask_reached))
                if index > reach or index == self._subtask:
                    raise ValueError(
                        f"Training advances one subtask at a time: press {self._subtask + 1}"
                        + (f" (or a reached one up to {reach})" if reach > self._subtask + 1 else "")
                        if self._subtask < count
                        else "Last subtask: finish the episode with Success"
                    )
                # Pressing k reports subtask k-1 done (A1 milestone), once per episode. 11-06: a lower
                # number goes back to that subtask (no reward); re-advancing to a reached one gives none.
                if index > self._subtask_reached:
                    self._milestones += 1
                    self._subtask_reached = index
            elif self._phase in {"ended", "returning"}:
                raise ValueError(f"Cannot select a subtask while {self._phase}")
            if index != self._subtask:
                self._runtime.pop("monitor_correction", None)
                self._subtask = index
                self._subtask_revision += 1
                if self._stage_switch_replan:
                    self._revision += 1
                name = "first_advance" if index > old_high else "backtrack" if index < old_stage else "revisit"
                if self._monitor_control is not None:
                    auto = self._monitor_control
                    auto.reset(execution=True)
                    if auto.active != "human" and source == "human":
                        self._revision += 1
                        auto.notice = f"已由人工纠正到 S{index}"
                    self._shadow_event(name, from_stage=old_stage, from_revision=old_revision,
                                       source=source, decision=decision)
                else:
                    self._shadow_event(name, from_stage=old_stage, from_revision=old_revision)
            self._condition.notify_all()
            return self.snapshot()

    def correct_monitor_stage(self, index, expected):
        with self._condition:
            auto = self._monitor_control
            if auto is None or auto.active == "human" or self._phase not in {"running", "paused"}:
                raise ValueError("人工纠正只用于本条正在运行的 Monitor 自动模式")
            state = self.snapshot()
            if (not isinstance(expected, dict) or any(type(expected.get(key)) is not int or
                    expected[key] != state[key] for key in ("episode", "revision", "subtask_revision"))):
                raise ValueError("阶段状态已变化，请刷新后重新选择")
            if type(index) is not int or not 1 <= index <= len(self._prompts):
                raise ValueError(f"请选择 subtask 1–{len(self._prompts)}")
            previous = self._subtask
            if self._phase == "running" and not self._human_latched:
                self.command("pause")
            # An explicit correction is a stage assignment, with no completion reward.
            # Re-entering a reached stage must not generate duplicate milestones.
            self._subtask = index
            self._subtask_reached = max(self._subtask_reached, index)
            self._subtask_revision += 1
            self._revision += 1
            self._collection_revision += 1
            auto.reset(execution=True)
            auto.notice = f"已人工纠正为 subtask {index}"
            self._runtime["monitor_correction"] = dict(stage=index, from_stage=previous,
                                                       ns=time.monotonic_ns())
            self._shadow_event("manual_stage_correction", source="human", from_stage=previous,
                               forced_stage=True, correction_reward=0,
                               control="human" if self._human_latched else "paused")
            self._condition.notify_all()
            return self.panel_snapshot()

    def consume_milestones(self):
        """Completed subtasks since the last call (training rewards)."""
        with self._condition:
            count, self._milestones = self._milestones, 0
            return count

    def mark_step_failure(self):
        """Mark the step being executed as failed (training reward -1).

        Nothing else changes: no revision bump, so the executing chunk, the
        H latch and the human samples continue exactly as before. Repeated
        presses within one step mark it once; ``step_failures`` counts the
        marked steps of this episode, i.e. its -1 rewards.
        """
        with self._condition:
            if self.mode != "training":
                raise ValueError("Step failure is a training reward")
            if self._phase != "running":
                raise ValueError(f"Cannot mark a step failure while {self._phase}")
            if not self._step_failure:
                self._step_failures += 1
            self._step_failure = True
            return self.snapshot()

    def consume_step_failure(self):
        """Whether a step failure was marked since the last call."""
        with self._condition:
            marked, self._step_failure = self._step_failure, False
            return marked

    def command(self, command, index=None, *, thresholds=None, monitor_mode=None, monitor_rule=None,
                monitor_version=None, expected=None):
        auto = self._monitor_control
        if command == "monitor_correct":
            return self.correct_monitor_stage(index, expected)
        if command == "shadow_version":
            with self._condition:
                if auto is None or self._phase != "waiting" or monitor_version not in shadow_settings.VERSIONS:
                    raise ValueError("只能在等待开始时选择有效的 Monitor 版本")
                available = (shadow.status() or {}).get("versions", {})
                if monitor_version != shadow_settings.BASE_VERSION and not available.get(monitor_version, {}).get("ready"):
                    raise ValueError("所选 Monitor 版本尚未加载")
                self._monitor_profiles, self._monitor_settings_error = shadow_settings.read(monitor_version)
                self._monitor_version = monitor_version
                return self.panel_snapshot()
        if command in {"monitor_mode", "monitor_manual"}:
            with self._condition:
                if auto is None:
                    raise ValueError("此入口未启用 Monitor 控制")
                if command == "monitor_mode":
                    auto.configure(self, shadow, monitor_mode, monitor_rule)
                else:
                    if (self._phase != "paused" and not (self._phase == "running" and self._human_latched)) or auto.active == "human":
                        raise ValueError("请先暂停本条")
                    auto.active = "human"
                    auto.reset(execution=True)
                    auto.notice = "本条已转人工，请继续 Policy 或接管"
                    self._revision += 1
                    self._collection_revision += 1
                    self._auto_event("monitor_manual", None, control="paused" if self._phase == "paused" else "human", requested_by="human")
                return self.panel_snapshot()
        if command in {"shadow_thresholds_save", "shadow_thresholds_reset"}:
            with self._condition:
                if self._monitor_profiles is None or self._phase != "waiting":
                    raise ValueError("只能在等待开始时调整当前版本的阈值")
                profiles = (shadow_settings.defaults(self._monitor_version) if command == "shadow_thresholds_reset"
                            else shadow_settings.overrides(thresholds))
                try:
                    shadow_settings.write(profiles, self._monitor_version)
                except OSError as exc:
                    raise ValueError(f"阈值保存失败：{exc}") from exc
                self._monitor_profiles = profiles
                self._monitor_settings_error = None
                return self.panel_snapshot()
        if command in {"shadow_on", "shadow_off"}:
            with self._condition:
                if not shadow.ROOT or self.mode != "training" or self._phase != "waiting":
                    raise ValueError("后台 Monitor 只能在等待开始时设置")
                if len(self._prompts) != 4:
                    raise ValueError("Monitor 需要四个已配置的阶段")
                if auto is not None and auto.mode != "human":
                    raise ValueError("自动模式需要后台 Monitor，请先选择人类全部负责")
                value = command == "shadow_on"
                try:
                    shadow.selection(value)
                except OSError as exc:
                    raise ValueError(f"Monitor 设置未生效：{exc}") from exc
                self._shadow_enabled = value
                return self.panel_snapshot()
        if command == "step_failure":
            return self.mark_step_failure()
        if command == "discard" and self.mode not in {"training", "dagger"}:
            raise ValueError("Discard drops recorded data; this panel records none")
        if command in {"subtask", "next_subtask"}:
            with self._condition:
                if command == "next_subtask":
                    index = self._subtask + 1
                return self.select_subtask(index)
        with self._condition:
            was_human = self._human_latched
            was_phase = self._phase
            if command == "resume" and self._phase == "paused" and self._human_latched:
                command = "human_stop"
            if (command == "human_stop" and auto is not None and self._phase == "paused"
                    and not self._human_latched):
                command = "resume"
            if (auto is not None and auto.fault and auto.ready(shadow)
                    and (command == "resume" or command == "human_stop")):
                auto.fault = False
                auto.reset()
                self._auto_event("monitor_recovered", None, control="human" if self._human_latched else "paused")
            if (auto is not None and auto.active != "human" and auto.fault and
                ((command == "resume" and self._phase == "paused") or command == "human_stop")):
                raise ValueError("Monitor 已离线，请先将本条转人工")
            if (
                command == "success"
                and (self.mode == "training" or self._stage_switch_replan)
                and self._prompts
                and self._subtask < len(self._prompts)
                and self._phase in {"running", "paused"}
            ):
                raise ValueError(
                    f"Success is accepted after the last subtask; now {self._subtask}/{len(self._prompts)}"
                )
            if command in {"cancel_return", "pause", "human_start"} and self._phase in {
                "ended",
                "returning",
            }:
                self._return_cancelled = True
                self._phase = "ended"
                if command == "human_start":
                    self._human_latched = True
            elif command == "start" and self._phase == "waiting":
                if auto is not None:
                    if self._shadow_enabled and self._monitor_version != shadow_settings.BASE_VERSION:
                        available = (shadow.status() or {}).get("versions", {})
                        if not available.get(self._monitor_version, {}).get("ready"):
                            raise ValueError("所选 Monitor 版本未就绪，请选择 1116v1 或等待加载")
                    auto.start(self, shadow)
                self._phase = "running"
            elif command == "pause" and self._phase == "running":
                self._phase = "paused"
            elif command == "resume" and self._phase == "paused":
                self._phase = "running"
            elif command == "human_start" and self._phase in {"running", "paused"}:
                # Also from a pause: a joint-limit pause is undone by hand.
                self._phase = "running"
                self._human_latched = True
            elif command == "human_stop" and self._phase in {"running", "paused"}:
                self._phase = "running"
                self._human_latched = False
            elif command in {"success", "failure", "discard"} and self._phase in {
                "running",
                "paused",
            }:
                self._phase = "ended"
                self._outcome = command
                self._return_cancelled = False
            else:
                raise ValueError(f"Cannot {command!r} while {self._phase}")
            if command in {
                "start", "resume", "human_start", "human_stop", "success", "failure", "discard"
            }:
                self._runtime.pop("limit_error", None)
            self._revision += 1
            if command in {"start", "pause", "resume", "human_start", "human_stop"}:
                self._collection_revision += 1
            event_extra = {}
            if auto is not None and command != "start":
                auto.reset(execution=command == "human_stop")
                if command == "human_start" and not was_human:
                    event_extra["monitor_assisted"] = auto.assisted
                    auto.assisted = False
                    auto.alarm = None
                if command == "resume" and auto.alarm is not None:
                    auto.alarm = None
                    auto.assisted = False
                    auto.notice = "操作员已恢复 Policy"
                if command == "human_stop":
                    auto.assisted = False
                    auto.notice = "已交回 Policy，重新积累判断证据"
            if was_phase in {"waiting", "running", "paused"}:
                if command in {"start", "pause", "resume", "success", "failure", "discard"} or (
                    command in {"human_start", "human_stop"} and was_human != self._human_latched
                ):
                    control = {"start": "resetting", "pause": "paused", "resume": "hold",
                               "human_start": "human", "human_stop": "hold"}.get(command, "ended")
                    self._shadow_event(command, control=control, **event_extra)
            self._condition.notify_all()
            return self.snapshot()

    def begin_episode(self):
        with self._condition:
            # A failed return or the step that ended the last episode stays
            # visible until the next episode actually starts.
            kept = {
                key: self._runtime[key]
                for key in ("last_return_error", "episode_error")
                if self._runtime.get(key)
            }
            self._episode += 1
            if self._monitor_control is not None:
                self._monitor_control.new_episode()
            self._phase = "waiting"
            self._human_latched = False
            self._outcome = None
            self._return_cancelled = False
            self._revision += 1
            self._collection_revision += 1
            if self._prompts and self._subtask != 1:
                self._subtask = 1
                self._subtask_revision += 1
            self._milestones = 0
            self._subtask_reached = self._subtask
            self._step_failure = False
            self._step_failures = 0
            self._runtime = {"step": 0, "control_mode": "waiting", **kept}
            self._condition.notify_all()

    def retry_start(self, error):
        """A failed Start reset waits for the next Start of the same episode."""
        with self._condition:
            self._phase = "waiting"
            self._human_latched = False
            self._revision += 1
            self._collection_revision += 1
            self._runtime.update(control_mode="reset_failed", reset_error=error)
            self._shadow_event("reset_failed", control="waiting", reason=error)
            self._condition.notify_all()

    def pause_for_limit(self, message):
        """Pause policy execution whose targets keep being unreachable.

        Like the operator's pause; the message says why. H (Pico) turns the
        wrist back, Resume retries, Success/Failure ends the episode.
        """
        with self._condition:
            if self._phase != "running" or self._human_latched:
                return False
            self._phase = "paused"
            self._revision += 1
            self._collection_revision += 1
            if self._monitor_control is not None:
                self._monitor_control.reset()
            self._runtime.update(limit_error=message)
            self._shadow_event("policy_limit", control="paused", reason=message)
            self._condition.notify_all()
            return True

    def wait_until_active(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._phase in {"running", "ended"}
            )
            if self._closed:
                raise RuntimeError("Tianji HIL control closed")
            return self.snapshot()

    def update(self, **values):
        with self._condition:
            mode = values.get("control_mode")
            if (self._shadow_enabled and self._phase == "running" and not self._human_latched
                and mode in {"policy", "hold"} and mode != self._runtime.get("control_mode")):
                self._shadow_event("policy_mode", control=mode)
            self._runtime.update(values)

    def panel_snapshot(self):
        result = self.snapshot()
        if self.mode == "training":
            metadata = shadow.status()
            if metadata is not None:
                if self._monitor_profiles is not None:
                    metadata["thresholds"] = shadow_settings.values(self._monitor_profiles)
                    if self._monitor_settings_error:
                        metadata["settings_error"] = self._monitor_settings_error
                if self._monitor_control is not None:
                    with self._condition:
                        auto = self._monitor_control
                        metadata["control"] = auto.public(self._phase, auto.ready(shadow))
                        metadata["version"] = self._monitor_version
                        missing = metadata.get("versions", {}).get(shadow_settings.FT_VERSION, {})
                        if missing.get("error"):
                            metadata["variant_error"] = "Sub-16ft_H-16v1 加载失败"
                result["monitor_shadow"] = metadata
        return result

    def end_episode(self):
        with self._condition:
            if self._phase != "ended" and self._outcome is None:
                self._shadow_event("fault", control="ended", reason=self._runtime.get("episode_error"))
            self._phase = "ended"
            self._condition.notify_all()

    def begin_return(self):
        """Claim one explicit episode End; never accept a new Start mid-return."""
        with self._condition:
            if (
                self._phase != "ended"
                or self._outcome is None
                or self._return_cancelled
            ):
                return None
            self._phase = "returning"
            self._human_latched = False
            self._runtime.update(control_mode="resetting", last_return_error=None)
            self._condition.notify_all()
            return self.snapshot()

    def finish_return(self, error=None):
        with self._condition:
            self._phase = "ended"
            self._runtime.update(
                control_mode="reset_failed" if error else "waiting",
                last_return_error=error,
            )
            self._condition.notify_all()

    def page(self) -> str:
        """The panel HTML for this mode; deployment modes carry no rewards."""
        page = _PAGE if self.mode == "stage1" else _PAGE.replace(
            "</html>", _XENSE_FOOTER + "</html>"
        )
        if self.mode == "training":
            return page
        if self.mode == "dagger":
            return (
                page.replace("HIL 训练", "HIL · DAgger 采集")
                .replace(_STEP_FAILURE_BUTTON, "")
                .replace("（奖励 1）", "")
                .replace("（奖励 0）", "")
            )
        if self._stage_switch_replan:
            banner = os.environ.get("TIANJI_CAR_BANNER", "").strip() or "Stage2-30HZ-1118v1 · 小车 HIL"
            page = page.replace("HIL 训练", banner)
        return (
            page.replace(
                "HIL 训练",
                "HIL · Stage1 推理" if self.mode == "stage1" else "HIL · Stage2 推理",
            )
            .replace(_STEP_FAILURE_BUTTON, "")
            .replace(_DISCARD_BUTTON, "")
            .replace("（奖励 1）", "")
            .replace("（奖励 0）", "")
            .replace("训练环境", "推理环境")
            .replace("已采集", "已运行")
        )

    def start_server(self, port: int):
        """Bind locally; the panel only writes this environment's episode state."""
        control = self
        page = self.page()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, body, content_type="application/json"):
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/":
                    self.respond(200, page, "text/html; charset=utf-8")
                elif self.path == "/api/xense-count" and control.mode != "stage1":
                    self.respond(200, json.dumps({"connected": _xense_connected_count()}))
                elif self.path == "/api/status":
                    self.respond(200, json.dumps(control.panel_snapshot()))
                else:
                    self.respond(404, "{}")

            def do_POST(self):
                if self.path != "/api/control":
                    self.respond(404, "{}")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if (
                        not 0 < length <= 1024
                        or self.headers.get("Content-Type") != "application/json"
                    ):
                        raise ValueError("Expected a small application/json command")
                    body = json.loads(self.rfile.read(length))
                    result = control.command(body["command"], body.get("index"), thresholds=body.get("thresholds"),
                                             monitor_mode=body.get("monitor_mode"), monitor_rule=body.get("monitor_rule"),
                                             monitor_version=body.get("monitor_version"), expected=body)
                except (ValueError, TypeError, KeyError) as exc:
                    self.respond(400, json.dumps({"error": str(exc)}))
                else:
                    self.respond(200, json.dumps(result))

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="tianji-hil", daemon=True
        )
        self._thread.start()
        if self._monitor_control is not None:
            self._monitor_control.listen(self, shadow)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._monitor_control is not None:
            self._monitor_control.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2)
