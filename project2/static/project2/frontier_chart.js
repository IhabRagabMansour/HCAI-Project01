// Accuracy vs. complexity scatter. Each point is one trained grid model;
// the lambda-selected model is highlighted. Uses the waitForChart guard so it
// never runs before Chart.js has loaded (tracking-prevention / slow-load safe).
(function () {
    const dataEl = document.getElementById("frontier-data");
    const canvas = document.getElementById("frontier-chart");
    if (!dataEl || !canvas) return;

    let payload;
    try { payload = JSON.parse(dataEl.textContent); }
    catch (e) { return; }
    if (!payload || !payload.points || !payload.points.length) return;

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
        const others = payload.points.filter(p => !p.selected);
        const selected = payload.points.filter(p => p.selected);

        new window.Chart(canvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Models",
                        data: others.map(p => ({ x: p.x, y: p.y, label: p.label })),
                        backgroundColor: "rgba(39, 92, 178, 0.55)",
                        pointRadius: 6,
                        pointHoverRadius: 8,
                    },
                    {
                        label: "Selected (λ)",
                        data: selected.map(p => ({ x: p.x, y: p.y, label: p.label })),
                        backgroundColor: "#d62728",
                        pointRadius: 9,
                        pointHoverRadius: 11,
                        pointStyle: "rectRot",
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const p = ctx.raw;
                                return `${payload.complexity_label.toLowerCase()}=${p.x}, acc=${p.y.toFixed(4)} (${p.label})`;
                            },
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: payload.complexity_label + " (Ω)" }, beginAtZero: true },
                    y: { title: { display: true, text: "Test accuracy" } },
                },
            },
        });
    }
})();
