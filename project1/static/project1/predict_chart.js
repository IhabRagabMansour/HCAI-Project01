(function () {
    const dataEl = document.getElementById("proba-data");
    const canvas = document.getElementById("proba-chart");
    if (!dataEl || !canvas) return;

    let items;
    try { items = JSON.parse(dataEl.textContent); }
    catch (e) { return; }
    if (!items || items.length === 0) return;

    // Sort descending so the top class is at the top of a horizontal bar
    const sorted = [...items].sort((a, b) => b.prob - a.prob);
    const maxProb = Math.max(...sorted.map(i => i.prob));

    new Chart(canvas, {
        type: "bar",
        data: {
            labels: sorted.map(i => i.label),
            datasets: [{
                label: "Probability",
                data: sorted.map(i => i.prob),
                backgroundColor: sorted.map(i =>
                    i.prob === maxProb ? "#275CB2" : "rgba(39, 92, 178, 0.45)"
                ),
            }],
        },
        options: {
            responsive: true,
            indexAxis: "y",
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(4)} (${(ctx.parsed.x * 100).toFixed(1)}%)`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Probability" },
                    min: 0, max: 1,
                    ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` },
                },
            },
        },
    });
})();
