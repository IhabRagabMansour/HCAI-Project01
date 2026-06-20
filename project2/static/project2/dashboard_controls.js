// Dashboard controls: live lambda readout + auto-submit on change.
// State lives in the URL query string so every panel stays consistent.
(function () {
    const form = document.getElementById("p2-controls");
    if (!form) return;

    const slider = document.getElementById("lambda");
    const valueEl = document.getElementById("lambda-value");
    const modelSelect = document.getElementById("model");

    // Live numeric readout while dragging (no submit on every tick).
    if (slider && valueEl) {
        slider.addEventListener("input", () => {
            valueEl.textContent = parseFloat(slider.value).toFixed(3);
        });
        // Submit when the user releases the slider.
        slider.addEventListener("change", () => form.submit());
    }

    // Submit immediately when the model class changes.
    if (modelSelect) {
        modelSelect.addEventListener("change", () => form.submit());
    }

    // Counterfactual controls (associated with this form via the form= attribute)
    // also auto-submit so every panel stays consistent with the selected model.
    ["cf_row", "cf_target", "cf_k"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("change", () => form.submit());
    });
})();
