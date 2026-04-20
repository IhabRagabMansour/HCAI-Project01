(function () {
    const dataUrl = window.__chartDataUrl;

    const typeSelect  = document.getElementById("chart-type");
    const xSelect     = document.getElementById("chart-x");
    const ySelect     = document.getElementById("chart-y");
    const modeSelect  = document.getElementById("chart-mode");
    const ctrlX       = document.getElementById("ctrl-x");
    const ctrlY       = document.getElementById("ctrl-y");
    const ctrlMode    = document.getElementById("ctrl-mode");
    const xLabel      = document.getElementById("chart-x-label");
    const errorEl     = document.getElementById("chart-error");
    const canvas      = document.getElementById("scatter-chart");

    let chart = null;

    // ── Control visibility ──────────────────────────────────────────────────
    function updateControls(type) {
        const showX    = type !== "heatmap";
        const showY    = type === "scatter";
        const showMode = type === "scatter" || type === "boxplot";

        ctrlX.style.display    = showX    ? "" : "none";
        ctrlY.style.display    = showY    ? "" : "none";
        ctrlMode.style.display = showMode ? "" : "none";

        xLabel.textContent = (type === "histogram" || type === "boxplot") ? "Column" : "X axis";
    }

    // ── URL helpers ─────────────────────────────────────────────────────────
    function buildFetchUrl(type, x, y, mode) {
        return `${dataUrl}?${new URLSearchParams({ type, x, y, mode })}`;
    }

    function updatePageUrl(type, x, y, mode) {
        const url = new URL(window.location.href);
        url.searchParams.set("type", type);
        url.searchParams.set("x", x);
        url.searchParams.set("y", y);
        url.searchParams.set("mode", mode);
        url.hash = "chart-section";
        history.replaceState(null, "", url.toString());
    }

    // ── Fetch + render ──────────────────────────────────────────────────────
    async function refresh() {
        const type = typeSelect.value;
        const x    = xSelect.value;
        const y    = ySelect.value;
        const mode = modeSelect.value;

        errorEl.style.display = "none";
        try {
            const resp = await fetch(buildFetchUrl(type, x, y, mode));
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.error || resp.statusText);
            }
            const data = await resp.json();
            render(data, type, x, y);
            updatePageUrl(type, x, y, mode);
        } catch (e) {
            errorEl.textContent = "Chart error: " + e.message;
            errorEl.style.display = "block";
        }
    }

    // ── Color helpers ───────────────────────────────────────────────────────
    function corrColor(v) {
        // blue (positive) ↔ white (zero) ↔ red (negative)
        if (v >= 0) {
            const t = Math.round(255 * (1 - v));
            return `rgb(${t},${t},255)`;
        }
        const t = Math.round(255 * (1 + v));
        return `rgb(255,${t},${t})`;
    }

    // ── Renderers ───────────────────────────────────────────────────────────
    function render(data, type, x, y) {
        if (chart) { chart.destroy(); chart = null; }

        if (type === "scatter") {
            chart = new Chart(canvas, {
                type: "scatter",
                data: { datasets: data.datasets },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: "top" },
                        tooltip: { callbacks: { label: ctx => `(${ctx.parsed.x}, ${ctx.parsed.y})` } },
                    },
                    scales: {
                        x: { title: { display: true, text: x } },
                        y: { title: { display: true, text: y } },
                    },
                },
            });

        } else if (type === "histogram") {
            chart = new Chart(canvas, {
                type: "bar",
                data: {
                    labels: data.labels,
                    datasets: data.datasets.map(ds => ({ ...ds, barPercentage: 1.0, categoryPercentage: 1.0 })),
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { title: { display: true, text: x } },
                        y: { title: { display: true, text: "Count" }, beginAtZero: true },
                    },
                },
            });

        } else if (type === "boxplot") {
            chart = new Chart(canvas, {
                type: "boxplot",
                data: { labels: data.labels, datasets: data.datasets },
                options: {
                    responsive: true,
                    plugins: { legend: { position: "top" } },
                    scales: {
                        y: { title: { display: true, text: x } },
                    },
                },
            });

        } else if (type === "heatmap") {
            const numCols = data.cols.length;
            chart = new Chart(canvas, {
                type: "matrix",
                data: {
                    datasets: [{
                        data: data.data,
                        backgroundColor(ctx) {
                            return corrColor(ctx.dataset.data[ctx.dataIndex].v);
                        },
                        borderColor: "white",
                        borderWidth: 2,
                        width:  ({ chart: c }) => ((c.chartArea || {}).width  || 400) / numCols - 2,
                        height: ({ chart: c }) => ((c.chartArea || {}).height || 400) / numCols - 2,
                    }],
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                title: () => "",
                                label: ctx => `${ctx.raw.y} × ${ctx.raw.x}: ${ctx.raw.v}`,
                            },
                        },
                    },
                    scales: {
                        x: { type: "category", labels: data.cols, ticks: { maxRotation: 45 } },
                        y: { type: "category", labels: [...data.cols].reverse() },
                    },
                },
            });
        }
    }

    // ── Event listeners ─────────────────────────────────────────────────────
    typeSelect.addEventListener("change", () => { updateControls(typeSelect.value); refresh(); });
    xSelect.addEventListener("change", refresh);
    ySelect.addEventListener("change", refresh);
    modeSelect.addEventListener("change", refresh);

    // ── Init ────────────────────────────────────────────────────────────────
    typeSelect.value  = window.__chartInitType;
    xSelect.value     = window.__chartInitX;
    ySelect.value     = window.__chartInitY;
    modeSelect.value  = window.__chartInitMode;

    updateControls(window.__chartInitType);
    refresh();
})();
