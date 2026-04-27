(function () {
    const dataEl = document.getElementById("model-evaluation");
    if (!dataEl) return;

    let data;
    try {
        data = JSON.parse(dataEl.textContent);
    } catch (e) {
        return;
    }
    if (!data) return;

    if (data.problem_type === "classification") {
        renderConfusionMatrix(data);
    } else if (data.problem_type === "regression") {
        renderPredictedVsActual(data);
        renderResiduals(data);
    }

    if (data.feature_importance && data.feature_importance.length > 0) {
        renderFeatureImportance(data.feature_importance);
    }

    // ── Confusion matrix heatmap ────────────────────────────────────────
    function renderConfusionMatrix(d) {
        const canvas = document.getElementById("cm-chart");
        if (!canvas) return;

        const labels = d.labels;
        const cm = d.confusion_matrix;
        let max = 0;
        for (const row of cm) for (const v of row) if (v > max) max = v;

        const points = [];
        for (let i = 0; i < cm.length; i++) {
            for (let j = 0; j < cm[i].length; j++) {
                points.push({ x: labels[j], y: labels[i], v: cm[i][j] });
            }
        }

        new Chart(canvas, {
            type: "matrix",
            data: {
                datasets: [{
                    data: points,
                    backgroundColor(ctx) {
                        const v = ctx.dataset.data[ctx.dataIndex].v;
                        const alpha = max > 0 ? 0.10 + 0.85 * (v / max) : 0.10;
                        return `rgba(39, 92, 178, ${alpha})`;
                    },
                    borderColor: "white",
                    borderWidth: 2,
                    width:  ({ chart: c }) => ((c.chartArea || {}).width  || 400) / labels.length - 2,
                    height: ({ chart: c }) => ((c.chartArea || {}).height || 400) / labels.length - 2,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: () => "",
                            label: (ctx) => `True: ${ctx.raw.y} · Predicted: ${ctx.raw.x} → ${ctx.raw.v}`,
                        },
                    },
                },
                scales: {
                    x: { type: "category", labels, title: { display: true, text: "Predicted" } },
                    y: { type: "category", labels: [...labels].reverse(), title: { display: true, text: "True" } },
                },
            },
        });
    }

    // ── Predicted vs Actual scatter (regression) ────────────────────────
    function renderPredictedVsActual(d) {
        const canvas = document.getElementById("pva-chart");
        if (!canvas) return;

        const points = d.predictions_sample.map(p => ({ x: p.y_true, y: p.y_pred }));
        const all = d.predictions_sample.flatMap(p => [p.y_true, p.y_pred]);
        const min = Math.min(...all);
        const max = Math.max(...all);

        new Chart(canvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Predictions",
                        data: points,
                        backgroundColor: "rgba(31, 119, 180, 0.6)",
                        pointRadius: 3.5,
                    },
                    {
                        type: "line",
                        label: "Perfect (y = x)",
                        data: [{ x: min, y: min }, { x: max, y: max }],
                        borderColor: "#999",
                        borderDash: [6, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        showLine: true,
                        fill: false,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Actual ${ctx.parsed.x.toFixed(3)} → Predicted ${ctx.parsed.y.toFixed(3)}`,
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: "Actual" } },
                    y: { title: { display: true, text: "Predicted" } },
                },
            },
        });
    }

    // ── Residuals scatter (regression) ──────────────────────────────────
    function renderResiduals(d) {
        const canvas = document.getElementById("res-chart");
        if (!canvas) return;

        const points = d.residuals_sample.map(p => ({ x: p.y_pred, y: p.residual }));
        const xs = d.residuals_sample.map(p => p.y_pred);
        const min = Math.min(...xs);
        const max = Math.max(...xs);

        new Chart(canvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Residuals",
                        data: points,
                        backgroundColor: "rgba(214, 39, 40, 0.6)",
                        pointRadius: 3.5,
                    },
                    {
                        type: "line",
                        label: "Zero residual",
                        data: [{ x: min, y: 0 }, { x: max, y: 0 }],
                        borderColor: "#999",
                        borderDash: [6, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        showLine: true,
                        fill: false,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Predicted ${ctx.parsed.x.toFixed(3)} → Residual ${ctx.parsed.y.toFixed(3)}`,
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: "Predicted" } },
                    y: { title: { display: true, text: "Residual (Actual − Predicted)" } },
                },
            },
        });
    }

    // ── Feature importance bar chart ────────────────────────────────────
    function renderFeatureImportance(items) {
        const canvas = document.getElementById("fi-chart");
        if (!canvas) return;

        const top = items.slice(0, 20);
        new Chart(canvas, {
            type: "bar",
            data: {
                labels: top.map(i => i.name),
                datasets: [{
                    label: "Importance",
                    data: top.map(i => i.value),
                    backgroundColor: "#275CB2",
                    borderColor: "#1e4a9a",
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                indexAxis: "y",
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: "Importance" }, beginAtZero: true },
                },
            },
        });
    }
})();
