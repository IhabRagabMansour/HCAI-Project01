// PDP & ALE feature-effect plots. Fetches both JSON endpoints for the
// currently selected model + feature and renders two line charts, each with
// three species-probability curves. Re-fetches on control change (no full
// page reload) and keeps the URL in sync. Uses the waitForChart guard.
(function () {
    const root = document.getElementById("feature-effects");
    if (!root) return;

    const featureSel = document.getElementById("fe_feature");
    const gridInput = document.getElementById("fe_n_grid");
    const binsInput = document.getElementById("fe_n_bins");
    const errorEl = document.getElementById("fe-error");
    const pdpCanvas = document.getElementById("pdp-chart");
    const aleCanvas = document.getElementById("ale-chart");

    const model = root.dataset.model;
    const lambda = root.dataset.lambda;
    const pdpUrl = root.dataset.pdpUrl;
    const aleUrl = root.dataset.aleUrl;

    // Consistent species colors across both plots.
    const COLORS = {
        Adelie: "#1f77b4",
        Chinstrap: "#ff7f0e",
        Gentoo: "#2ca02c",
    };

    let pdpChart = null;
    let aleChart = null;

    waitForChart(refresh);

    function waitForChart(cb) {
        if (typeof window.Chart !== "undefined") { cb(); return; }
        let n = 0;
        const t = setInterval(() => {
            if (typeof window.Chart !== "undefined") { clearInterval(t); cb(); }
            else if (++n >= 100) { clearInterval(t); }
        }, 50);
    }

    function commonParams() {
        return new URLSearchParams({
            model: model,
            lambda: lambda,
            feature: featureSel.value,
        });
    }

    async function refresh() {
        errorEl.style.display = "none";
        const feature = featureSel.value;
        const nGrid = gridInput.value;
        const nBins = binsInput.value;

        try {
            const pdpParams = commonParams(); pdpParams.set("n_grid", nGrid);
            const aleParams = commonParams(); aleParams.set("n_bins", nBins);

            const [pdpResp, aleResp] = await Promise.all([
                fetch(`${pdpUrl}?${pdpParams}`),
                fetch(`${aleUrl}?${aleParams}`),
            ]);
            if (!pdpResp.ok || !aleResp.ok) throw new Error("Failed to load feature effects.");

            const pdp = await pdpResp.json();
            const ale = await aleResp.json();

            renderPdp(pdp, feature);
            renderAle(ale, feature);
            syncUrl(feature, nGrid, nBins);
        } catch (e) {
            errorEl.textContent = e.message || "Error loading feature effects.";
            errorEl.style.display = "block";
        }
    }

    function datasetsFor(payload, xs) {
        // payload.curves is [n_x][n_classes]; transpose to one series per class.
        return payload.classes.map((cls, ci) => ({
            label: cls,
            data: xs.map((x, xi) => ({ x: x, y: payload.curves[xi][ci] })),
            borderColor: COLORS[cls] || "#666",
            backgroundColor: COLORS[cls] || "#666",
            pointRadius: 2,
            borderWidth: 2,
            tension: 0.2,
            fill: false,
        }));
    }

    function renderPdp(pdp, feature) {
        if (pdpChart) pdpChart.destroy();
        pdpChart = new window.Chart(pdpCanvas, {
            type: "line",
            data: { datasets: datasetsFor(pdp, pdp.grid) },
            options: lineOptions(feature, "Average predicted probability"),
        });
    }

    function renderAle(ale, feature) {
        if (aleChart) aleChart.destroy();
        aleChart = new window.Chart(aleCanvas, {
            type: "line",
            data: { datasets: datasetsFor(ale, ale.centers) },
            options: lineOptions(feature, "Accumulated local effect (centered)"),
        });
    }

    function lineOptions(feature, yTitle) {
        return {
            responsive: true,
            interaction: { mode: "nearest", intersect: false },
            plugins: {
                legend: { position: "top" },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: { type: "linear", title: { display: true, text: feature } },
                y: { title: { display: true, text: yTitle } },
            },
        };
    }

    function syncUrl(feature, nGrid, nBins) {
        const url = new URL(window.location.href);
        url.searchParams.set("fe_feature", feature);
        url.searchParams.set("fe_n_grid", nGrid);
        url.searchParams.set("fe_n_bins", nBins);
        history.replaceState(null, "", url.toString());
    }

    featureSel.addEventListener("change", refresh);
    gridInput.addEventListener("change", refresh);
    binsInput.addEventListener("change", refresh);
})();
