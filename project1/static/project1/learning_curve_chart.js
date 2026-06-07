// Learning curve: two lines (train + validation) with semi-transparent
// std-deviation bands. Reads JSON embedded by Django's json_script.
(function () {
    const dataEl = document.getElementById("learning-curve-data");
    const canvas = document.getElementById("lc-chart");
    if (!dataEl || !canvas) return;

    let lc;
    try { lc = JSON.parse(dataEl.textContent); }
    catch (e) { return; }
    if (!lc || !lc.train_sizes || !lc.train_sizes.length) return;

    const sizes = lc.train_sizes;

    // Build std-band arrays for each line: lower bound (invisible) + upper
    // bound (filled to previous = lower).
    const trainLower = lc.train_mean.map((m, i) => m - lc.train_std[i]);
    const trainUpper = lc.train_mean.map((m, i) => m + lc.train_std[i]);
    const valLower   = lc.val_mean.map((m, i) => m - lc.val_std[i]);
    const valUpper   = lc.val_mean.map((m, i) => m + lc.val_std[i]);

    new Chart(canvas, {
        type: "line",
        data: {
            labels: sizes,
            datasets: [
                // Train std band — lower, invisible
                {
                    label: "_train_lower",
                    data: trainLower,
                    borderColor: "transparent",
                    backgroundColor: "transparent",
                    pointRadius: 0,
                    fill: false,
                    tension: 0.2,
                },
                // Train std band — upper, filled toward previous (lower)
                {
                    label: "_train_upper",
                    data: trainUpper,
                    borderColor: "transparent",
                    backgroundColor: "rgba(31, 119, 180, 0.20)",
                    pointRadius: 0,
                    fill: "-1",
                    tension: 0.2,
                },
                // Train mean line
                {
                    label: "Train",
                    data: lc.train_mean,
                    borderColor: "#1f77b4",
                    backgroundColor: "#1f77b4",
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2,
                },
                // Val std band — lower, invisible
                {
                    label: "_val_lower",
                    data: valLower,
                    borderColor: "transparent",
                    backgroundColor: "transparent",
                    pointRadius: 0,
                    fill: false,
                    tension: 0.2,
                },
                // Val std band — upper, filled toward previous (lower)
                {
                    label: "_val_upper",
                    data: valUpper,
                    borderColor: "transparent",
                    backgroundColor: "rgba(255, 127, 14, 0.20)",
                    pointRadius: 0,
                    fill: "-1",
                    tension: 0.2,
                },
                // Validation mean line
                {
                    label: "Validation",
                    data: lc.val_mean,
                    borderColor: "#ff7f0e",
                    backgroundColor: "#ff7f0e",
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        // Hide the synthetic band datasets (labels prefixed with "_")
                        filter: (item) => !item.text.startsWith("_"),
                    },
                },
                tooltip: {
                    filter: (item) => !item.dataset.label.startsWith("_"),
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: { title: { display: true, text: "Training set size" } },
                y: { title: { display: true, text: lc.metric_label || "Score" } },
            },
        },
    });
})();
