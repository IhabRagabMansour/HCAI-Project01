(function () {
    const dataEl = document.getElementById("compare-payload");
    if (!dataEl) return;

    let payload;
    try {
        payload = JSON.parse(dataEl.textContent);
    } catch (e) {
        return;
    }

    const canvas = document.getElementById("compare-chart");
    const select = document.getElementById("compare-metric");
    if (!canvas || !select) return;

    let chart = null;
    let current = payload.selected;

    function render(metricKey) {
        const m = payload.metrics[metricKey];
        if (!m) return;

        if (chart) {
            chart.destroy();
            chart = null;
        }

        chart = new Chart(canvas, {
            type: "bar",
            data: {
                labels: payload.labels,
                datasets: [
                    {
                        label: "Train",
                        data: m.train,
                        backgroundColor: "rgba(31, 119, 180, 0.45)",
                        borderColor: "#1f77b4",
                        borderWidth: 1,
                    },
                    {
                        label: "Test",
                        data: m.test,
                        backgroundColor: "rgba(31, 119, 180, 0.95)",
                        borderColor: "#1f77b4",
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y === null ? "—" : ctx.parsed.y.toFixed(4)}`,
                        },
                    },
                },
                scales: {
                    y: {
                        title: { display: true, text: m.display },
                        beginAtZero: true,
                    },
                    x: { title: { display: true, text: "Model" } },
                },
            },
        });
    }

    function updateUrl(metricKey) {
        const url = new URL(window.location.href);
        url.searchParams.set("metric", metricKey);
        history.replaceState(null, "", url.toString());
    }

    select.addEventListener("change", () => {
        current = select.value;
        render(current);
        updateUrl(current);
    });

    render(current);
})();
