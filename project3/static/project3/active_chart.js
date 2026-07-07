// Active-learning performance curve: team accuracy vs number of expert queries,
// for uncertainty sampling vs random, with reference lines for the classifier-only
// baseline and the full-supervision ceiling. Uses the waitForChart guard.
(function () {
    const dataEl = document.getElementById("active-data");
    const canvas = document.getElementById("active-chart");
    if (!dataEl || !canvas) return;

    let payload;
    try { payload = JSON.parse(dataEl.textContent); }
    catch (e) { return; }
    if (!payload || !payload.checkpoints) return;

    waitForChart(render);

    function waitForChart(cb) {
        if (typeof window.Chart !== "undefined") { cb(); return; }
        let n = 0;
        const t = setInterval(() => {
            if (typeof window.Chart !== "undefined") { clearInterval(t); cb(); }
            else if (++n >= 100) { clearInterval(t); }
        }, 50);
    }

    function render() {
        const xs = payload.checkpoints;
        const flat = (v) => xs.map(() => v);

        new window.Chart(canvas, {
            type: "line",
            data: {
                labels: xs,
                datasets: [
                    {
                        label: "Uncertainty sampling",
                        data: payload.uncertainty,
                        borderColor: "#27ae60",
                        backgroundColor: "#27ae60",
                        pointRadius: 3,
                        borderWidth: 2,
                        tension: 0.2,
                    },
                    {
                        label: "Random",
                        data: payload.random,
                        borderColor: "#d62728",
                        backgroundColor: "#d62728",
                        pointRadius: 3,
                        borderWidth: 2,
                        tension: 0.2,
                    },
                    {
                        label: "Full supervision (all pool labels)",
                        data: flat(payload.full_supervision),
                        borderColor: "#999",
                        borderDash: [6, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                    },
                    {
                        label: "Classifier only (no deferral)",
                        data: flat(payload.classifier_only),
                        borderColor: "#1f77b4",
                        borderDash: [3, 3],
                        borderWidth: 1.5,
                        pointRadius: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                interaction: { mode: "nearest", intersect: false },
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}` },
                    },
                },
                scales: {
                    x: { title: { display: true, text: "Number of expert queries" } },
                    y: { title: { display: true, text: "Human-AI team test accuracy" } },
                },
            },
        });
    }
})();
