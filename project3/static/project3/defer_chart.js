// Accuracy comparison bar chart for the learning-to-defer section.
// Highlights the expert-advantage team bar. Uses the waitForChart guard.
(function () {
    const dataEl = document.getElementById("summary-data");
    const canvas = document.getElementById("summary-chart");
    if (!dataEl || !canvas) return;

    let payload;
    try { payload = JSON.parse(dataEl.textContent); }
    catch (e) { return; }
    if (!payload || !payload.labels) return;

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
        // Color the "Advantage team" bar distinctly; oracle in grey.
        const colors = payload.labels.map((lbl) => {
            if (lbl === "Advantage team") return "#27ae60";
            if (lbl === "Oracle") return "#999";
            return "#275CB2";
        });

        new window.Chart(canvas, {
            type: "bar",
            data: {
                labels: payload.labels,
                datasets: [{
                    label: "Accuracy",
                    data: payload.values,
                    backgroundColor: colors,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: (ctx) => ctx.parsed.y.toFixed(4) } },
                },
                scales: {
                    y: { beginAtZero: true, max: 1, title: { display: true, text: "Test accuracy" } },
                },
            },
        });
    }
})();
