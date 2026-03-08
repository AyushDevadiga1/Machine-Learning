import { useState, useEffect, useRef } from "react";

/* ── palette ── */
const C = {
  bg:       "#0d0d0d",
  surface:  "#161616",
  card:     "#1e1e1e",
  border:   "#2a2a2a",
  borderHi: "#3a3a3a",
  accent:   "#f0a500",
  accentDim:"#f0a50022",
  green:    "#34d399",
  greenDim: "#34d39920",
  red:      "#f87171",
  redDim:   "#f8717120",
  blue:     "#60a5fa",
  purple:   "#a78bfa",
  text:     "#e8e8e8",
  muted:    "#666",
  faint:    "#333",
};

/* ── data ── */
const DATA = [
  { id:1, outlook:"Sunny",    wind:"Strong", play:"No"  },
  { id:2, outlook:"Sunny",    wind:"Weak",   play:"No"  },
  { id:3, outlook:"Overcast", wind:"Strong", play:"Yes" },
  { id:4, outlook:"Overcast", wind:"Weak",   play:"Yes" },
  { id:5, outlook:"Rain",     wind:"Strong", play:"No"  },
  { id:6, outlook:"Rain",     wind:"Weak",   play:"Yes" },
];

function gini(rows) {
  const n = rows.length; if (!n) return 0;
  const p = rows.filter(r => r.play === "Yes").length / n;
  return +(1 - p*p - (1-p)*(1-p)).toFixed(4);
}
const f = v => typeof v === "number" ? v.toFixed(4) : v;

const STEPS = [
  { label:"Dataset",          icon:"01", title:"The Training Data",          sub:"6 samples · 2 features · 1 target" },
  { label:"Root Gini",        icon:"02", title:"Root Gini Impurity",         sub:"Measuring disorder before any split" },
  { label:"Split: Outlook",   icon:"03", title:"Candidate Split — Outlook",  sub:"Sunny / Overcast / Rain" },
  { label:"Split: Wind",      icon:"04", title:"Candidate Split — Wind",     sub:"Strong / Weak" },
  { label:"Best Split",       icon:"05", title:"Choosing the Best Split",    sub:"Outlook wins with Gain = 0.3333" },
  { label:"Rain Branch",      icon:"06", title:"Recursion — Rain Branch",    sub:"Still mixed → split again on Wind" },
  { label:"Final Tree",       icon:"07", title:"Complete Decision Tree",     sub:"All leaves pure · depth 2 · 100% accuracy" },
];

/* ─────────────── tiny helpers ─────────────── */
function Tag({ children, color = C.accent }) {
  return (
    <span style={{
      display:"inline-block", padding:"1px 8px", borderRadius:3,
      background: color + "18", border:`1px solid ${color}44`,
      color, fontSize:11, fontFamily:"'IBM Plex Mono',monospace", letterSpacing:.5,
    }}>{children}</span>
  );
}

function Divider() {
  return <div style={{ height:1, background:C.border, margin:"12px 0" }}/>;
}

/* ─────────────── animated counter ─────────────── */
function CountUp({ to, duration = 800 }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = null;
    const step = ts => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      setVal(+(to * p).toFixed(4));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [to]);
  return <span>{val.toFixed(4)}</span>;
}

/* ─────────────── progress bar ─────────────── */
function GiniBar({ label, value, max = 0.5, color, delay = 0 }) {
  const pct = (value / max) * 100;
  return (
    <div style={{ marginBottom:10, animation:`fadeUp .4s ease ${delay}s both` }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
        <span style={{ fontSize:12, color:C.muted, fontFamily:"'IBM Plex Mono',monospace" }}>{label}</span>
        <span style={{ fontSize:12, color, fontFamily:"'IBM Plex Mono',monospace", fontWeight:600 }}>{f(value)}</span>
      </div>
      <div style={{ height:6, background:C.faint, borderRadius:3, overflow:"hidden" }}>
        <div style={{
          height:"100%", borderRadius:3,
          width: pct + "%", background: color,
          transition:"width 1s cubic-bezier(.4,0,.2,1)",
          boxShadow:`0 0 8px ${color}88`,
        }}/>
      </div>
    </div>
  );
}

/* ─────────────── calculation line ─────────────── */
function CL({ items, delay=0, highlight=false }) {
  return (
    <div style={{
      display:"flex", gap:6, flexWrap:"wrap", alignItems:"center",
      padding: highlight ? "6px 10px" : "0",
      background: highlight ? C.accentDim : "transparent",
      borderLeft: highlight ? `3px solid ${C.accent}` : "3px solid transparent",
      borderRadius:2,
      marginBottom: 4,
      animation:`fadeUp .35s ease ${delay}s both`,
      fontFamily:"'IBM Plex Mono',monospace",
      fontSize:12.5,
      lineHeight:1.9,
    }}>
      {items.map((item, i) => (
        <span key={i} style={{ color: item.color || C.muted }}>{item.text}</span>
      ))}
    </div>
  );
}

/* ─────────────── tree SVG ─────────────── */
function TreeSVG({ step }) {
  const show = n => step >= n;

  /* node */
  const Node = ({ x, y, text, sub, type="decision", delay=0 }) => {
    const isYes = type === "yes";
    const isNo  = type === "no";
    const isLeaf = isYes || isNo;
    const bg     = isYes ? C.greenDim : isNo ? C.redDim : "#ffffff0a";
    const border = isYes ? C.green    : isNo ? C.red    : C.accent;
    const txt    = isYes ? C.green    : isNo ? C.red    : C.text;
    return (
      <g style={{ animation:`fadeUp .45s ease ${delay}s both` }}>
        <rect x={x-48} y={y-18} width={96} height={36}
          rx={isLeaf ? 18 : 6}
          fill={bg} stroke={border} strokeWidth={1.5}
        />
        <text x={x} y={sub ? y-4 : y+1}
          textAnchor="middle" dominantBaseline="middle"
          fill={txt} fontSize={11.5}
          fontFamily="'IBM Plex Mono',monospace" fontWeight="600">
          {text}
        </text>
        {sub && <text x={x} y={y+10}
          textAnchor="middle" dominantBaseline="middle"
          fill={C.muted} fontSize={8.5}
          fontFamily="'IBM Plex Mono',monospace">
          {sub}
        </text>}
      </g>
    );
  };

  /* edge */
  const Edge = ({ x1,y1,x2,y2,label,lcolor=C.muted,delay=0 }) => {
    const len = Math.sqrt((x2-x1)**2+(y2-y1)**2);
    const mx=(x1+x2)/2, my=(y1+y2)/2;
    return (
      <g>
        <line x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={C.faint} strokeWidth={1.5} strokeDasharray="0"
          style={{
            strokeDasharray:len, strokeDashoffset:len,
            animation:`dash .6s ease ${delay}s forwards`,
          }}/>
        <text x={mx+5} y={my-4}
          fill={lcolor} fontSize={9.5}
          fontFamily="'IBM Plex Mono',monospace" fontWeight="600"
          style={{ animation:`fadeUp .3s ease ${delay+.4}s both` }}>
          {label}
        </text>
      </g>
    );
  };

  return (
    <svg width="100%" viewBox="0 0 380 220" style={{ overflow:"visible" }}>
      {show(4) && <>
        <Node x={190} y={34} text="Outlook ?" delay={0}/>
        <Edge x1={155} y1={50} x2={72}  y2={108} label="Sunny"    lcolor={C.accent} delay={.15}/>
        <Edge x1={190} y1={52} x2={190} y2={108} label="Overcast" lcolor={C.purple} delay={.3}/>
        <Edge x1={225} y1={50} x2={308} y2={108} label="Rain"     lcolor={C.blue}   delay={.45}/>
        <Node x={72}  y={124} text="✕  No"  sub="0Y · 2N" type="no"  delay={.5}/>
        <Node x={190} y={124} text="✓  Yes" sub="2Y · 0N" type="yes" delay={.65}/>
        {show(5)
          ? <Node x={308} y={124} text="Wind ?"       delay={.1}/>
          : <Node x={308} y={124} text="mixed" sub="1Y · 1N" delay={.8}/>
        }
      </>}
      {show(5) && <>
        <Edge x1={280} y1={140} x2={244} y2={192} label="Strong" lcolor={C.red}   delay={.2}/>
        <Edge x1={336} y1={140} x2={360} y2={192} label="Weak"   lcolor={C.green} delay={.4}/>
        <Node x={244} y={208} text="✕  No"  sub="0Y · 1N" type="no"  delay={.55}/>
        <Node x={360} y={208} text="✓  Yes" sub="1Y · 0N" type="yes" delay={.7}/>
      </>}
    </svg>
  );
}

/* ─────────────── calc content per step ─────────────── */
function CalcContent({ step }) {
  const yes=3, no=3, rG=gini(DATA);
  const sunny=DATA.filter(r=>r.outlook==="Sunny");
  const over =DATA.filter(r=>r.outlook==="Overcast");
  const rain =DATA.filter(r=>r.outlook==="Rain");
  const gS=gini(sunny),gO=gini(over),gR=gini(rain);
  const wO=+(2/6*gS+2/6*gO+2/6*gR).toFixed(4);
  const gainO=+(rG-wO).toFixed(4);
  const str=DATA.filter(r=>r.wind==="Strong"),wk=DATA.filter(r=>r.wind==="Weak");
  const gSt=gini(str),gWk=gini(wk);
  const wW=+(3/6*gSt+3/6*gWk).toFixed(4);
  const gainW=+(rG-wW).toFixed(4);

  const H = ({ children }) => (
    <div style={{ fontSize:11, color:C.muted, fontFamily:"'IBM Plex Mono',monospace",
      letterSpacing:2, textTransform:"uppercase", marginBottom:10 }}>{children}</div>
  );

  const panels = [
    /* 0 — overview */
    <div key={0}>
      <H>Algorithm: CART · Gini Criterion</H>
      {[
        "Compute Gini impurity at root (all data)",
        "For each feature, try it as a split",
        "Compute weighted Gini across branches",
        "Gain  =  Gini_parent − Gini_weighted",
        "Pick the feature with the highest Gain",
        "Recurse on each branch until leaves are pure",
      ].map((t,i) => (
        <div key={i} style={{
          display:"flex", gap:12, alignItems:"flex-start",
          marginBottom:8, animation:`fadeUp .4s ease ${i*.08}s both`,
        }}>
          <span style={{ color:C.accent, fontFamily:"'IBM Plex Mono',monospace",
            fontSize:11, minWidth:18, paddingTop:2 }}>{String(i+1).padStart(2,"0")}</span>
          <span style={{ color:C.text, fontSize:13, lineHeight:1.5 }}>{t}</span>
        </div>
      ))}
      <Divider/>
      <div style={{ display:"flex", gap:8 }}>
        <Tag>Gini = 1 − Σ pᵢ²</Tag>
        <Tag>0 = pure</Tag>
        <Tag>0.5 = max impure</Tag>
      </div>
    </div>,

    /* 1 — root gini */
    <div key={1}>
      <H>Root Node — All 6 Samples</H>
      <CL delay={0} items={[{text:"Samples:", color:C.muted},{text:`${DATA.length}`,color:C.text},{text:"  Yes:",color:C.muted},{text:`${yes}`,color:C.green},{text:"  No:",color:C.muted},{text:`${no}`,color:C.red}]}/>
      <Divider/>
      <CL delay={.1} items={[{text:"Gini",color:C.accent},{text:"=",color:C.muted},{text:"1 − P(Yes)² − P(No)²",color:C.text}]}/>
      <CL delay={.2} items={[{text:"     =",color:C.muted},{text:`1 − (${yes}/6)² − (${no}/6)²`,color:C.text}]}/>
      <CL delay={.3} items={[{text:"     =",color:C.muted},{text:`1 − ${(yes/6*yes/6).toFixed(4)} − ${(no/6*no/6).toFixed(4)}`,color:C.text}]}/>
      <CL delay={.5} highlight items={[{text:"     =",color:C.muted},{text:f(rG),color:C.accent},{text:"  ← root impurity",color:C.muted}]}/>
      <Divider/>
      <GiniBar label="Root Gini" value={rG} color={C.accent} delay={.6}/>
      <div style={{ fontSize:12, color:C.muted, marginTop:8, animation:"fadeUp .4s ease .7s both" }}>
        0.5 means perfectly mixed — half Yes, half No.
      </div>
    </div>,

    /* 2 — outlook split */
    <div key={2}>
      <H>Split by Outlook (3 branches)</H>
      {[
        ["Sunny",    C.accent,  sunny, gS],
        ["Overcast", C.purple,  over,  gO],
        ["Rain",     C.blue,    rain,  gR],
      ].map(([name,clr,rows,g],i) => (
        <div key={name} style={{ marginBottom:10, animation:`fadeUp .4s ease ${i*.15}s both` }}>
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
            <Tag color={clr}>{name} · {rows.length} rows</Tag>
            <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:12, color:g===0?C.green:C.muted }}>
              Gini = {f(g)}{g===0?" ✓ pure":""}
            </span>
          </div>
          <GiniBar label="" value={g} color={clr} delay={i*.15+.1}/>
        </div>
      ))}
      <Divider/>
      <CL delay={.5} items={[{text:"Weighted Gini",color:C.muted},{text:"=",color:C.muted},{text:`(2/6)×${f(gS)} + (2/6)×${f(gO)} + (2/6)×${f(gR)}`,color:C.text}]}/>
      <CL delay={.65} items={[{text:"              =",color:C.muted},{text:f(wO),color:C.text}]}/>
      <CL delay={.8} highlight items={[{text:"Gain(Outlook)",color:C.accent},{text:"=",color:C.muted},{text:`${f(rG)} − ${f(wO)}`,color:C.text},{text:"=",color:C.muted},{text:f(gainO),color:C.green}]}/>
    </div>,

    /* 3 — wind split */
    <div key={3}>
      <H>Split by Wind (2 branches)</H>
      {[
        ["Strong", C.red,   str, gSt],
        ["Weak",   C.green, wk,  gWk],
      ].map(([name,clr,rows,g],i) => (
        <div key={name} style={{ marginBottom:10, animation:`fadeUp .4s ease ${i*.15}s both` }}>
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
            <Tag color={clr}>{name} · {rows.length} rows</Tag>
            <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:12, color:C.muted }}>Gini = {f(g)}</span>
          </div>
          <GiniBar label="" value={g} color={clr} delay={i*.15+.1}/>
        </div>
      ))}
      <Divider/>
      <CL delay={.35} items={[{text:"Weighted Gini",color:C.muted},{text:"=",color:C.muted},{text:`(3/6)×${f(gSt)} + (3/6)×${f(gWk)}`,color:C.text}]}/>
      <CL delay={.5}  items={[{text:"              =",color:C.muted},{text:f(wW),color:C.text}]}/>
      <CL delay={.65} highlight items={[{text:"Gain(Wind)",color:C.blue},{text:"=",color:C.muted},{text:`${f(rG)} − ${f(wW)}`,color:C.text},{text:"=",color:C.muted},{text:f(gainW),color:C.blue}]}/>
    </div>,

    /* 4 — comparison */
    <div key={4}>
      <H>Feature Gain Comparison</H>
      <GiniBar label="Gain(Outlook)" value={gainO} max={0.4} color={C.green} delay={0}/>
      <GiniBar label="Gain(Wind)"    value={gainW} max={0.4} color={C.blue}  delay={.15}/>
      <Divider/>
      <div style={{ animation:"fadeUp .4s ease .3s both" }}>
        <div style={{ display:"flex", gap:12, marginBottom:8 }}>
          <Tag color={C.green}>Outlook → {f(gainO)}</Tag>
          <Tag color={C.blue}>Wind → {f(gainW)}</Tag>
        </div>
        <div style={{ fontSize:13, color:C.text, lineHeight:1.7 }}>
          Outlook has <span style={{color:C.accent,fontWeight:600}}>6× higher gain</span> than Wind.
          It becomes the root split node.
        </div>
      </div>
      <Divider/>
      <div style={{ animation:"fadeUp .4s ease .5s both" }}>
        <div style={{ fontSize:11, color:C.muted, marginBottom:8, fontFamily:"'IBM Plex Mono',monospace", letterSpacing:1, textTransform:"uppercase" }}>Resulting branches</div>
        {[
          ["Sunny",    "→ No  (0Y, 2N)",  "pure",  C.red],
          ["Overcast", "→ Yes (2Y, 0N)",  "pure",  C.green],
          ["Rain",     "→ ??? (1Y, 1N)",  "mixed", C.accent],
        ].map(([b,r,s,c],i)=>(
          <div key={b} style={{ display:"flex", gap:10, alignItems:"center",
            marginBottom:5, animation:`fadeUp .3s ease ${.55+i*.1}s both`}}>
            <Tag color={c}>{b}</Tag>
            <span style={{ fontSize:12, color:C.text, fontFamily:"'IBM Plex Mono',monospace" }}>{r}</span>
            <span style={{ fontSize:10, color:s==="pure"?C.green:C.accent }}>{s==="pure"?"● pure":"● recurse"}</span>
          </div>
        ))}
      </div>
    </div>,

    /* 5 — rain branch */
    <div key={5}>
      <H>Recursion on Rain branch</H>
      <div style={{ marginBottom:10, animation:"fadeUp .4s ease both" }}>
        <Tag color={C.blue}>Rain rows only</Tag>
        <div style={{ marginTop:8, fontFamily:"'IBM Plex Mono',monospace", fontSize:12, lineHeight:2 }}>
          <span style={{color:C.muted}}>ID5:</span> Rain + Strong <span style={{color:C.red}}>→ No</span><br/>
          <span style={{color:C.muted}}>ID6:</span> Rain + Weak &nbsp;&nbsp;<span style={{color:C.green}}>→ Yes</span>
        </div>
      </div>
      <Divider/>
      <CL delay={.2} items={[{text:"Split on Wind:",color:C.muted}]}/>
      <CL delay={.35} items={[{text:"  Strong",color:C.red},{text:"→ [No]",color:C.text},{text:"  Gini =",color:C.muted},{text:"0.0000 ✓ pure",color:C.green}]}/>
      <CL delay={.5}  items={[{text:"  Weak  ",color:C.green},{text:"→ [Yes]",color:C.text},{text:"Gini =",color:C.muted},{text:"0.0000 ✓ pure",color:C.green}]}/>
      <Divider/>
      <CL delay={.65} highlight items={[{text:"Both leaves are pure.",color:C.text},{text:"No further recursion needed.",color:C.muted}]}/>
    </div>,

    /* 6 — final */
    <div key={6}>
      <H>Decision Rules (human-readable)</H>
      {[
        ["IF Outlook = Sunny",                          "→  Don't play",  C.red],
        ["IF Outlook = Overcast",                       "→  Play!",        C.green],
        ["IF Outlook = Rain  AND  Wind = Strong",       "→  Don't play",  C.red],
        ["IF Outlook = Rain  AND  Wind = Weak",         "→  Play!",        C.green],
      ].map(([cond,res,c],i)=>(
        <div key={i} style={{
          marginBottom:8, padding:"8px 10px",
          background: c===C.green?C.greenDim:C.redDim,
          borderLeft:`3px solid ${c}`,
          borderRadius:3,
          animation:`fadeUp .4s ease ${i*.12}s both`,
        }}>
          <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:11.5, color:C.muted, marginBottom:2 }}>{cond}</div>
          <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:12, color:c, fontWeight:600 }}>{res}</div>
        </div>
      ))}
      <Divider/>
      <div style={{ display:"flex", gap:8, flexWrap:"wrap", animation:"fadeUp .4s ease .5s both" }}>
        <Tag color={C.accent}>Depth: 2</Tag>
        <Tag color={C.accent}>Nodes: 7</Tag>
        <Tag color={C.green}>Pure leaves: 4/4</Tag>
        <Tag color={C.green}>Accuracy: 100%</Tag>
      </div>
    </div>,
  ];

  return panels[step] || null;
}

/* ─────────────── main ─────────────── */
export default function App() {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    if (auto) {
      timer.current = setInterval(() => {
        setStep(s => { if (s >= STEPS.length-1) { setAuto(false); return s; } return s+1; });
      }, 4500);
    }
    return () => clearInterval(timer.current);
  }, [auto]);

  const hlRows  = step===1 ? DATA.map(r=>r.id) : step===5 ? [5,6] : null;
  const dimRows = step===5 ? DATA.filter(r=>r.outlook!=="Rain").map(r=>r.id) : null;

  const outlookColor = o => o==="Sunny"?C.accent:o==="Overcast"?C.purple:C.blue;

  return (
    <div style={{ minHeight:"100vh", background:C.bg, color:C.text, fontFamily:"'DM Sans',sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&family=DM+Serif+Display&display=swap');
        @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
        @keyframes dash   { to{stroke-dashoffset:0} }
        @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:.4} }
        * { box-sizing:border-box; margin:0; padding:0; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:${C.bg}; }
        ::-webkit-scrollbar-thumb { background:${C.faint}; border-radius:2px; }
        button { font-family: inherit; }
      `}</style>

      {/* ── top bar ── */}
      <div style={{
        background:C.surface, borderBottom:`1px solid ${C.border}`,
        padding:"0 28px", height:52,
        display:"flex", alignItems:"center", justifyContent:"space-between",
        position:"sticky", top:0, zIndex:100,
      }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <span style={{
            fontFamily:"'DM Serif Display',serif", fontSize:17, color:C.accent, letterSpacing:.5
          }}>Decision Tree</span>
          <span style={{ color:C.faint }}>·</span>
          <span style={{ fontSize:12, color:C.muted, fontFamily:"'IBM Plex Mono',monospace" }}>
            CART · Gini Impurity
          </span>
        </div>
        {/* step pills */}
        <div style={{ display:"flex", gap:4 }}>
          {STEPS.map((s,i) => (
            <button key={i} onClick={() => { setAuto(false); setStep(i); }}
              style={{
                padding:"4px 10px", borderRadius:4, fontSize:11, cursor:"pointer",
                fontFamily:"'IBM Plex Mono',monospace", fontWeight:600, letterSpacing:.5,
                background: i===step ? C.accent : i<step ? C.accentDim : "transparent",
                color:      i===step ? "#000"   : i<step ? C.accent    : C.muted,
                border:     `1px solid ${i===step?C.accent:i<step?C.accent+"44":C.faint}`,
                transition: "all .2s",
              }}>{s.icon}</button>
          ))}
        </div>
      </div>

      {/* ── step header ── */}
      <div style={{
        background:C.surface, borderBottom:`1px solid ${C.border}`,
        padding:"16px 28px",
      }}>
        <div style={{ display:"flex", alignItems:"baseline", gap:16 }} key={step}>
          <div style={{
            fontFamily:"'DM Serif Display',serif", fontSize:22, color:C.text,
            animation:"fadeUp .35s ease",
          }}>{STEPS[step].title}</div>
          <div style={{ fontSize:13, color:C.muted, animation:"fadeUp .35s ease .05s both" }}>
            {STEPS[step].sub}
          </div>
        </div>
      </div>

      {/* ── main 3-col ── */}
      <div style={{ display:"grid", gridTemplateColumns:"200px 1fr 1fr", height:"calc(100vh - 130px)" }}>

        {/* col 1: dataset */}
        <div style={{
          borderRight:`1px solid ${C.border}`,
          padding:"20px 16px",
          overflowY:"auto",
          background:C.surface,
        }}>
          <div style={{ fontSize:10, color:C.muted, letterSpacing:2, textTransform:"uppercase",
            marginBottom:12, fontFamily:"'IBM Plex Mono',monospace" }}>Training Data</div>

          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12,
            fontFamily:"'IBM Plex Mono',monospace" }}>
            <thead>
              <tr style={{ borderBottom:`1px solid ${C.border}` }}>
                {["#","Outlook","Wind","Play"].map(h=>(
                  <th key={h} style={{ padding:"4px 6px", color:C.muted,
                    fontWeight:600, textAlign:"left", fontSize:10, letterSpacing:1 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {DATA.map(r => {
                const hl = hlRows?.includes(r.id);
                const dm = dimRows?.includes(r.id);
                return (
                  <tr key={r.id} style={{
                    opacity: dm ? 0.18 : 1,
                    background: hl ? C.accentDim : "transparent",
                    borderLeft: hl ? `2px solid ${C.accent}` : "2px solid transparent",
                    transition:"all .3s",
                  }}>
                    <td style={{ padding:"5px 6px", color:C.faint }}>{r.id}</td>
                    <td style={{ padding:"5px 6px", color:outlookColor(r.outlook) }}>{r.outlook}</td>
                    <td style={{ padding:"5px 6px", color:C.muted }}>{r.wind}</td>
                    <td style={{ padding:"5px 6px", color:r.play==="Yes"?C.green:C.red, fontWeight:600 }}>{r.play}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div style={{ marginTop:20 }}>
            <div style={{ fontSize:10, color:C.muted, letterSpacing:2, textTransform:"uppercase",
              marginBottom:10, fontFamily:"'IBM Plex Mono',monospace" }}>Formulas</div>
            <div style={{ background:C.card, border:`1px solid ${C.border}`,
              borderRadius:6, padding:"12px", fontFamily:"'IBM Plex Mono',monospace", fontSize:11.5 }}>
              <div style={{ color:C.accent, marginBottom:6 }}>Gini Impurity</div>
              <div style={{ color:C.muted, lineHeight:1.9 }}>
                G = 1 − Σ pᵢ²<br/>
                <span style={{fontSize:10,color:C.faint}}>range: [0, 0.5]</span>
              </div>
              <div style={{ height:1, background:C.border, margin:"10px 0"}}/>
              <div style={{ color:C.accent, marginBottom:6 }}>Gini Gain</div>
              <div style={{ color:C.muted, lineHeight:1.9 }}>
                ΔG = G_parent − G_weighted<br/>
                <span style={{fontSize:10,color:C.faint}}>pick highest ΔG</span>
              </div>
            </div>
          </div>
        </div>

        {/* col 2: calculations */}
        <div style={{
          borderRight:`1px solid ${C.border}`,
          padding:"20px",
          overflowY:"auto",
          background:C.bg,
        }}>
          <div style={{ fontSize:10, color:C.muted, letterSpacing:2, textTransform:"uppercase",
            marginBottom:14, fontFamily:"'IBM Plex Mono',monospace" }}>Step {step+1} Calculations</div>
          <CalcContent step={step}/>
        </div>

        {/* col 3: tree */}
        <div style={{ padding:"20px", overflowY:"auto", background:C.bg }}>
          <div style={{ fontSize:10, color:C.muted, letterSpacing:2, textTransform:"uppercase",
            marginBottom:14, fontFamily:"'IBM Plex Mono',monospace" }}>Tree Structure</div>

          <div style={{
            background:C.card, border:`1px solid ${C.border}`,
            borderRadius:8, padding:"16px", minHeight:240,
          }}>
            {step < 4
              ? <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
                  justifyContent:"center", height:200, gap:8 }}>
                  <div style={{ fontSize:13, color:C.faint, fontFamily:"'IBM Plex Mono',monospace" }}>
                    tree builds at step 05
                  </div>
                  <div style={{ display:"flex", gap:6 }}>
                    {[0,1,2,3].map(i=>(
                      <div key={i} style={{
                        width:8,height:8,borderRadius:"50%",
                        background: step>i?C.accent:C.faint,
                        transition:"background .3s",
                      }}/>
                    ))}
                  </div>
                </div>
              : <TreeSVG step={step}/>
            }
          </div>

          {step >= 4 && (
            <div style={{ marginTop:14, display:"flex", flexDirection:"column", gap:6,
              animation:"fadeUp .4s ease" }}>
              {[["Decision node",C.accent,4],["Yes leaf",C.green,20],["No leaf",C.red,20]].map(([l,c,r])=>(
                <div key={l} style={{ display:"flex", alignItems:"center", gap:8,
                  fontSize:11, fontFamily:"'IBM Plex Mono',monospace", color:C.muted }}>
                  <div style={{ width:12,height:12,borderRadius:r,
                    background:`${c}18`, border:`1.5px solid ${c}` }}/>
                  {l}
                </div>
              ))}
            </div>
          )}

          {step === 6 && (
            <div style={{
              marginTop:14, background:C.greenDim,
              border:`1px solid ${C.green}44`, borderRadius:6, padding:"12px 14px",
              animation:"fadeUp .5s ease",
            }}>
              <div style={{ fontSize:13, fontWeight:600, color:C.green, marginBottom:6 }}>
                Tree complete ✓
              </div>
              <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                <Tag color={C.green}>depth 2</Tag>
                <Tag color={C.green}>7 nodes</Tag>
                <Tag color={C.green}>4 pure leaves</Tag>
                <Tag color={C.green}>100% train acc</Tag>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── controls ── */}
      <div style={{
        position:"fixed", bottom:0, left:0, right:0, zIndex:100,
        background:C.surface, borderTop:`1px solid ${C.border}`,
        padding:"10px 28px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
      }}>
        <div style={{ fontSize:11, color:C.muted, fontFamily:"'IBM Plex Mono',monospace" }}>
          {STEPS[step].label}
        </div>

        <div style={{ display:"flex", gap:8, alignItems:"center" }}>
          {[
            ["↺", () => { setAuto(false); setStep(0); }],
            ["← Prev", () => { setAuto(false); setStep(s=>Math.max(0,s-1)); }],
          ].map(([l,fn])=>(
            <button key={l} onClick={fn} style={{
              padding:"6px 14px", border:`1px solid ${C.border}`,
              borderRadius:5, background:"transparent",
              color:C.muted, cursor:"pointer", fontSize:13,
              transition:"all .2s",
            }}>{l}</button>
          ))}

          <button onClick={()=>setAuto(a=>!a)} style={{
            padding:"7px 24px", border:"none", borderRadius:5, cursor:"pointer",
            background: auto ? C.accentDim : C.accent,
            color: auto ? C.accent : "#000",
            fontWeight:600, fontSize:13,
            transition:"all .2s",
          }}>{auto ? "⏸  Pause" : "▶  Play"}</button>

          <button onClick={()=>{ setAuto(false); setStep(s=>Math.min(STEPS.length-1,s+1)); }} style={{
            padding:"6px 14px", border:`1px solid ${C.border}`,
            borderRadius:5, background:"transparent",
            color:C.muted, cursor:"pointer", fontSize:13,
            transition:"all .2s",
          }}>Next →</button>
        </div>

        <div style={{ fontSize:11, color:C.faint, fontFamily:"'IBM Plex Mono',monospace" }}>
          {step+1} / {STEPS.length}
        </div>
      </div>
    </div>
  );
}