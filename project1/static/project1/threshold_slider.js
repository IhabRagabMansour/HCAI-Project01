// Probability threshold slider for binary classification.
// Recomputes confusion matrix + 4 metrics client-side from a sample of
// test-set probabilities embedded by Django's json_script.
(function () {
    const dataEl = document.getElementById("threshold-data");
    const slider = document.getElementById("threshold-slider");
    if (!dataEl || !slider) return;

    let data;
    try { data = JSON.parse(dataEl.textContent); }
    catch (e) { return; }
    if (!data || !data.samples || !data.samples.length) return;

    const valueEl  = document.getElementById("threshold-value");
    const tnEl = document.getElementById("th-tn");
    const fpEl = document.getElementById("th-fp");
    const fnEl = document.getElementById("th-fn");
    const tpEl = document.getElementById("th-tp");
    const accEl = document.getElementById("th-accuracy");
    const f1El  = document.getElementById("th-f1");
    const precEl = document.getElementById("th-precision");
    const recEl  = document.getElementById("th-recall");

    function counts(threshold) {
        let tn = 0, fp = 0, fn = 0, tp = 0;
        for (const s of data.samples) {
            const pred = s.proba >= threshold ? 1 : 0;
            if (s.y_true === 1 && pred === 1) tp++;
            else if (s.y_true === 1 && pred === 0) fn++;
            else if (s.y_true === 0 && pred === 1) fp++;
            else tn++;
        }
        return { tn, fp, fn, tp };
    }

    function metrics(c) {
        const total = c.tn + c.fp + c.fn + c.tp;
        const accuracy  = total ? (c.tp + c.tn) / total : 0;
        const precision = (c.tp + c.fp) ? c.tp / (c.tp + c.fp) : 0;
        const recall    = (c.tp + c.fn) ? c.tp / (c.tp + c.fn) : 0;
        const f1        = (precision + recall)
            ? 2 * precision * recall / (precision + recall)
            : 0;
        return { accuracy, precision, recall, f1 };
    }

    function update() {
        const t = parseFloat(slider.value);
        valueEl.textContent = t.toFixed(2);
        const c = counts(t);
        const m = metrics(c);
        tnEl.textContent = c.tn;
        fpEl.textContent = c.fp;
        fnEl.textContent = c.fn;
        tpEl.textContent = c.tp;
        accEl.textContent  = m.accuracy.toFixed(4);
        f1El.textContent   = m.f1.toFixed(4);
        precEl.textContent = m.precision.toFixed(4);
        recEl.textContent  = m.recall.toFixed(4);
    }

    slider.addEventListener("input", update);
    update();
})();
