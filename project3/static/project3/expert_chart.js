// Grouped bar chart: expert vs. classifier per-class accuracy.
// Uses the waitForChart guard so it never runs before Chart.js has loaded.
(function () {
    const dataEl = document.getElementById("compare-data");
    const canvas = document.getElementById("compare-chart");
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
        new window.Chart(canvas, {
            type: "bar",
            data: {
                labels: payload.labels,
                datasets: [
                    {
                        label: "Classifier",
                        data: payload.classifier,
                        backgroundColor: "#1f77b4",
                    },
                    {
                        label: "Expert",
                        data: payload.expert,
                        backgroundColor: "#ff7f0e",
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}` },
                    },
                },
                scales: {
                    y: { beginAtZero: true, max: 1, title: { display: true, text: "Accuracy" } },
                    x: { title: { display: true, text: "Class" } },
                },
            },
        });
    }
})();
