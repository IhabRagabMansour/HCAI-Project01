// Per-class accuracy bar chart for the baseline classifier.
// Uses the waitForChart guard so it never runs before Chart.js has loaded.
(function () {
    const dataEl = document.getElementById("perclass-data");
    const canvas = document.getElementById("perclass-chart");
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
                datasets: [{
                    label: "Accuracy (recall)",
                    data: payload.accuracy,
                    backgroundColor: "#275CB2",
                    borderColor: "#1e4a9a",
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: { label: (ctx) => ctx.parsed.y.toFixed(4) },
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
