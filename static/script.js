(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("riskForm"), submitBtn = $("submitBtn"), errorNote = $("errorNote"), verdict = $("verdict");
  const CIRC = 540.35; // 2*PI*86, matches CSS

  const MONTHS = ["sept", "aug", "jul", "jun", "may", "apr"];
  const PRESETS = {
    low: { limit_bal: 200000, sex: "Female", education: "University", marriage: "Married", age: 38,
      pay: [-1, -1, 0, 0, -1, -1], bill: [12000, 9000, 15000, 11000, 8000, 7000], paid: [12000, 9000, 15000, 11000, 8000, 7000] },
    high: { limit_bal: 30000, sex: "Male", education: "High School", marriage: "Single", age: 26,
      pay: [3, 3, 2, 2, 2, 2], bill: [29500, 28800, 29900, 27500, 26000, 25000], paid: [0, 1000, 0, 1500, 0, 0] },
  };

  function fill(p) {
    ["limit_bal", "sex", "education", "marriage", "age"].forEach((k) => ($(k).value = p[k]));
    MONTHS.forEach((m, i) => {
      $(`pay_status_${m}`).value = p.pay[i];
      $(`bill_amt_${m}`).value = p.bill[i];
      $(`pay_amt_${m}`).value = p.paid[i];
    });
  }
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => fill(PRESETS[b.dataset.preset])));

  fetch("/health").then((r) => r.json()).then((d) => {
    $("apiDot").classList.add("ok");
    $("apiStatusText").textContent = "service ready";
    const a = d.metrics["at_0.50"].accuracy * 100, r = d.metrics.at_tuned.recall * 100;
    $("modelNote").textContent =
      `Logistic regression on 30,000 cardholders · test accuracy ${a.toFixed(1)}% at 0.50, ` +
      `ROC-AUC ${d.metrics.roc_auc.toFixed(2)}, recall ${r.toFixed(0)}% at the tuned threshold. Model estimates, not a lending decision.`;
  }).catch(() => {
    $("apiDot").classList.add("down");
    $("apiStatusText").textContent = "service unreachable";
  });

  function animateNumber(el, to, ms) {
    const t0 = performance.now();
    (function tick(now) {
      const t = Math.min(1, (now - t0) / ms);
      el.textContent = (to * (1 - Math.pow(1 - t, 3))).toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
    })(t0);
  }

  function render(d) {
    const pct = d.default_probability * 100, thr = d.threshold * 100, high = d.default_prediction === 1;
    verdict.hidden = false;
    verdict.scrollIntoView({ behavior: "smooth", block: "nearest" });
    $("gaugeFill").style.stroke = high ? "var(--risk-red)" : "var(--brass)";
    requestAnimationFrame(() => ($("gaugeFill").style.strokeDashoffset = CIRC * (1 - pct / 100)));
    $("gaugeThreshold").style.transform = `rotate(${thr * 3.6}deg)`;
    animateNumber($("probNumber"), pct, 1000);
    const badge = $("stampBadge");
    badge.classList.remove("stamp--in", "risk-high");
    void badge.offsetWidth;
    if (high) badge.classList.add("risk-high");
    $("stampText").textContent = high ? "HIGH RISK" : "LOW RISK";
    requestAnimationFrame(() => badge.classList.add("stamp--in"));
    $("factProb").textContent = `${pct.toFixed(1)}%`;
    $("factThreshold").textContent = `${thr.toFixed(1)}%`;
    $("factResult").textContent = d.Result;

    const max = Math.max(...d.factors.map((f) => Math.abs(f.impact)), 0.01);
    $("factorList").innerHTML = d.factors.map((f) => {
      const w = (Math.abs(f.impact) / max) * 50, up = f.impact > 0;
      return `<li class="${up ? "up" : "down"}"><span>${f.feature}</span>
        <span class="bar"><i style="width:${w}%"></i></span>
        <span class="tag">${up ? "raises" : "lowers"} risk</span></li>`;
    }).join("");
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorNote.hidden = true;
    submitBtn.disabled = true;
    submitBtn.classList.add("loading");
    submitBtn.querySelector(".btn-label").textContent = "Reviewing file…";
    const payload = {};
    new FormData(form).forEach((v, k) => (payload[k] = v));
    try {
      const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      render(data);
    } catch (err) {
      errorNote.textContent = err.message;
      errorNote.hidden = false;
    } finally {
      submitBtn.disabled = false;
      submitBtn.classList.remove("loading");
      submitBtn.querySelector(".btn-label").textContent = "Assess risk";
    }
  });
})();
