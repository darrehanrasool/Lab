let cur = 0;

    /* accent mapping per stage (colour theory: cool core + one warm, one alert) */
    const STAGE_AC = ["crimson", "blue", "indigo", "gold", "teal", "blue", "indigo", "teal", "gold", "blue", "teal"];
    const ACV = { blue: "var(--blue)", teal: "var(--teal)", indigo: "var(--indigo)", gold: "var(--gold)", crimson: "var(--crimson)" };
    const acOf = s => s.k === "title" ? "blue" : STAGE_AC[s.st];

    /* ---------- builders ---------- */
    function bullets(arr) {
      return `<ul class="bl">` + arr.map(p => {
        const m = p.match(/^\s+[\u2022•]\s*(.*)$/);
        return m ? `<li class="sub">${m[1]}</li>` : `<li>${p}</li>`;
      }).join("") + `</ul>`;
    }
    function figure(s) {
      if (!s.fig || !FIG_ASSETS[s.fig]) return "";
      const t = (s.figTitle || "").replace(/'/g, "\\'");
      return `<div class="fig">
    <div class="fig-h"><span class="fig-b">📊 Scientific figure</span>
      <button class="fig-btn" onclick="openViewer('${s.fig}','${t}')">⛶ Full screen</button></div>
    <div class="fig-w" onclick="openViewer('${s.fig}','${t}')"><img src="${FIG_ASSETS[s.fig]}" alt="${s.figTitle || 'Figure'}"></div>
    <div class="fig-m"><b>${s.figTitle || ""}</b><p>${s.figCap || ""}</p></div></div>`;
    }
    function maths(s) {
      return (s.math || []).map(m => `<div class="fm"><span class="fm-h">${m.h}</span>
    <div class="fm-m">${m.html || m.m}</div><span class="fm-d">${m.d || ""}</span></div>`).join("");
    }
    function table(s) {
      if (!s.table) return "";
      return `<table><thead><tr>${s.table[0].map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${s.table.slice(1).map(w => `<tr>${w.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }
    function split() {
      return `<div class="sp-bar"><div class="sp-train">📚 TRAINING 70% · 309 patients</div><div class="sp-val">🔍 VAL 20% · 88</div><div class="sp-test">🔒 TEST 10% · 45</div></div>
  <div class="sp-cards">
   <div class="sp-c" style="--k:var(--blue)"><div class="r"><span class="t">TRAIN (70%)</span><span class="c">309</span></div><h4>The Model's Textbook</h4><p>Learns all 11 parameter weights (&beta;<sub>0</sub> to &beta;<sub>10</sub>) exclusively here. 5-Fold CV: R² = 0.4587 ± 0.0538.</p></div>
   <div class="sp-c" style="--k:var(--teal)"><div class="r"><span class="t">VAL (20%)</span><span class="c">88</span></div><h4>The Practice Mock Exam</h4><p>Used to verify convergence, check that errors are normal, and confirm zero overfitting.</p></div>
   <div class="sp-c" style="--k:var(--crimson)"><div class="r"><span class="t">🔒 TEST (10%)</span><span class="c">45</span></div><h4>The Board Exam</h4><p>Strictly locked in a vault! Never seen during training. Evaluated strictly once (Test R² = 0.4882).</p></div>
  </div>
  <div class="sp-ok"><b>✓ Zero Distribution Drift:</b> Progression is identical across sets (Train: 151.2, Val: 155.1, Test: 152.8).</div>`;
    }
    const tag = s => `<div class="tg"><i></i>${s.sp} · Stage ${s.st + 1}: ${D.stages[s.st]}</div>`;
    const NODE_C = ["blue", "teal", "indigo", "gold", "teal", "blue", "indigo", "teal", "gold", "blue", "teal"];

    function cover(s) {
      const tiles = s.plain.map(p => { const m = p.match(/^<b>(.*?):?<\/b>\s*(.*)$/); return m ? `<div class="cv-i"><b>${m[1].replace(/:$/, "")}</b><span>${m[2]}</span></div>` : "" }).join("");
      return `<div class="cd"><div class="cover">
    <div class="cv-l">
      <div class="cv-eye"><i></i>Department of Artificial Intelligence · IUST Kashmir</div>
      <h1 class="cv-t">Diabetes Progression Prediction via <em>Multiple Linear Regression</em></h1>
      <div class="cv-g">${tiles}</div>
    </div>
    <div class="cv-r">
      <div class="cv-lab">Under the guidance of</div>
      <div class="pm"><div class="av" style="background:var(--indigo)">MJ</div><div><div class="nm">Dr. M. Abdul Jawad</div><div class="rl">Faculty · Course Instructor</div></div></div>
      <div class="cv-sep"></div>
      <div class="cv-lab">Presented by</div>
      <div class="pm"><div class="av">DR</div><div><div class="nm">Dar Rehan Rasool</div><span class="rg">IUST0120023376</span></div></div>
      <div class="pm"><div class="av" style="background:var(--teal)">SP</div><div><div class="nm">Shafiya Bashir Pampori</div><span class="rg" style="color:var(--teal);background:color-mix(in srgb,var(--teal) 14%,transparent)">IUST0126023407</span></div></div>
    </div></div></div>`;
    }
    function thanks(s) {
      return `<div class="cd">
    <div class="hd">${tag(s)}<h2>${s.t}</h2></div>
    <div class="gr"><div class="col fit">${bullets(s.plain.slice(0, 4))}</div>
    <div class="th-r"><h1>Thank You!</h1>
      <p>We welcome your questions, evaluation feedback, and live demonstration of our clinical decision support web application.</p>
      <div class="kp">
        <div><small>Holdout Test R²</small><strong style="color:var(--teal)">0.4882</strong><em>Stanford Empirical Limit</em></div>
        <div><small>Triage Sensitivity</small><strong style="color:var(--blue)">76.92%</strong><em>Cost-Sensitive Imbalance</em></div>
        <div><small>Inference Latency</small><strong style="color:var(--indigo)">&lt; 1.2 ms</strong><em>Closed-Form Normal Eq.</em></div></div>
      <div class="live">🌐 Live Doctor Dashboard Active: <code>http://127.0.0.1:5003</code></div>
      <div class="th-by"><b>Dar Rehan Rasool</b> (IUST0120023376) · <b>Shafiya Bashir Pampori</b> (IUST0126023407)<br>Faculty: <b>Dr. M. Abdul Jawad</b></div>
    </div></div></div>`;
    }
    function slideHTML(s, i) {
      if (s.k === "title") return cover(s);
      if (s.k === "thanks") return thanks(s);
      let l, r;
      if (s.k === "pipe") {
        l = `<div class="col fit">${bullets(s.plain)}</div>`;
        r = `<div class="col fit"><div class="col-h">The 11-stage clinical assembly line</div><div class="pp">${D.pipe.map((p, j) => {
          const m = p[0].match(/^(\d+)\.\s*(.*)$/);
          return `<div class="nd" style="--nc:${ACV[NODE_C[j]]}"><span class="num">${m ? m[1] : j + 1}</span><div><b>${m ? m[2] : p[0]}</b><span>${p[1]}</span></div></div>`
        }).join("")}</div></div>`;
      } else if (s.k === "split") {
        l = `<div class="col fit">${bullets(s.plain)}</div>`;
        r = `<div class="col fit">${split()}</div>`;
      } else {
        if (s.table) { /* bullets + table on the left, figure + formulas on the right */
          l = `<div class="col fit">${bullets(s.plain)}${table(s)}</div>`;
          r = `<div class="col fit">${figure(s)}${maths(s)}</div>`;
        } else {
          l = `<div class="col fit">${bullets(s.plain)}</div>`;
          r = `<div class="col fit">${figure(s)}${maths(s)}</div>`;
        }
      }
      return `<div class="cd"><div class="hd">${tag(s)}<h2>${s.t}</h2></div><div class="gr">${l}${r}</div></div>`;
    }

    $("steps").innerHTML = D.stages.map((s, i) => `<span class="st" data-i="${i}" style="--sc:${ACV[STAGE_AC[i]]}"><span class="n">${i + 1}</span>${s}</span>`).join("");
    $("stage").innerHTML = S.map((s, i) => `<div class="sl a-${acOf(s)}" style="--sc:${ACV[acOf(s)]}">${slideHTML(s, i)}</div>`).join("");
    const sl = [...document.querySelectorAll(".sl")], sp = [...document.querySelectorAll(".st")];

    /* ---------- auto-fit: shrink type until nothing overflows ---------- */
    function fitAll() {
      sl.forEach(el => {
        let fs = 28; el.style.setProperty("--fs", fs + "px");
        const cols = [...el.querySelectorAll(".fit")];
        const over = () => cols.some(c => c.scrollHeight > c.clientHeight + 2);
        while (over() && fs > 17) { fs -= 1; el.style.setProperty("--fs", fs + "px"); }
      });
    }

    /* ---------- navigation ---------- */
    function go(n) {
      cur = Math.max(0, Math.min(S.length - 1, n));
      sl.forEach((e, i) => e.className = e.className.replace(/\b(on|pv)\b/g, "").trim() + (i === cur ? " on" : i < cur ? " pv" : ""));
      sp.forEach((e, i) => { e.classList.toggle("on", i === S[cur].st); e.classList.toggle("done", i < S[cur].st) });
      $("ct").textContent = (cur + 1) + " / " + S.length;
      $("prog").firstElementChild.style.width = ((cur + 1) / S.length * 100) + "%";
      document.documentElement.style.setProperty("--sc", ACV[acOf(S[cur])]);
      $("deck").style.setProperty("--sc", ACV[acOf(S[cur])]);
      $("pv").disabled = cur === 0; $("nx").disabled = cur === S.length - 1;
      $("notes-t").textContent = S[cur].say || "";
      history.replaceState(null, "", "#" + (cur + 1));
    }
    sp.forEach(e => e.onclick = () => go(S.findIndex(s => s.st === +e.dataset.i)));
    $("nx").onclick = () => go(cur + 1); $("pv").onclick = () => go(cur - 1);
    $("th").onclick = toggleTheme; $("nt").onclick = () => $("notes").classList.toggle("open");
    $("fs").onclick = togglePresFS;
    function toggleTheme() { const r = document.documentElement; r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark" }
    function togglePresFS() {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen && document.documentElement.requestFullscreen().catch(() => { });
    }

    /* ---------- figure viewer ---------- */
    function openViewer(key, title) {
      if (!FIG_ASSETS[key]) return;
      $("vw-i").src = FIG_ASSETS[key]; $("vw-t").textContent = title || "Figure"; $("vw-c").textContent = "Click the image to toggle actual size · Esc to close";
      $("vw-b").classList.remove("full"); $("vw-z").textContent = "🔍 Actual size";
      $("viewer").classList.add("open");
    }
    function closeViewer() {
      if (document.fullscreenElement === $("viewer")) document.exitFullscreen();
      $("viewer").classList.remove("open");
    }
    function viewerZoom() { const f = $("vw-b").classList.toggle("full"); $("vw-z").textContent = f ? "🔍 Fit to screen" : "🔍 Actual size" }
    $("vw-x").onclick = closeViewer; $("vw-z").onclick = viewerZoom; $("vw-i").onclick = viewerZoom;
    $("vw-f").onclick = () => {
      if (document.fullscreenElement === $("viewer")) document.exitFullscreen();
      else $("viewer").requestFullscreen ? $("viewer").requestFullscreen().catch(() => { }) : 0;
    };
    /* keep old function names working */
    window.openLightbox = openViewer; window.closeLightbox = closeViewer;

    addEventListener("keydown", e => {
      if ($("viewer").classList.contains("open")) {
        if (e.key === "Escape" && !document.fullscreenElement) closeViewer();
        return;
      }
      if (["ArrowRight", " ", "PageDown"].includes(e.key)) { e.preventDefault(); go(cur + 1) }
      else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); go(cur - 1) }
      else if (e.key === "Home") go(0); else if (e.key === "End") go(S.length - 1);
      else if (e.key === "f" || e.key === "F") togglePresFS();
      else if (e.key === "t" || e.key === "T") toggleTheme();
      else if (e.key === "n" || e.key === "N") $("notes").classList.toggle("open");
    });
    let tx = null;
    addEventListener("touchstart", e => tx = e.touches[0].clientX, { passive: true });
    addEventListener("touchend", e => { if (tx === null) return; const d = e.changedTouches[0].clientX - tx; if (Math.abs(d) > 60) go(cur + (d < 0 ? 1 : -1)); tx = null });

    /* ---------- scale canvas to screen ---------- */
    function scale() {
      const k = Math.min(innerWidth / 1920, innerHeight / 1080);
      $("deck").style.transform = `translate(-50%,-50%) scale(${k})`;
    }
    addEventListener("resize", scale); scale();

    /* ---------- init ---------- */
    const start = parseInt(location.hash.slice(1)) || 1;
    go(start - 1);
    (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => { fitAll() });
    setTimeout(fitAll, 400);
  </script>
</body>

</html>