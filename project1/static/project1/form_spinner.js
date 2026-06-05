// Generic submit-time spinner for any form carrying a data-loading-label attr.
// Disables submit buttons, swaps their text to the configured label, and
// injects a CSS spinner glyph so the user knows the request is in flight.
(function () {
    document.querySelectorAll("form[data-loading-label]").forEach((form) => {
        form.addEventListener("submit", () => {
            const labelText = form.dataset.loadingLabel || "Working…";
            const buttons = form.querySelectorAll('button[type="submit"]');
            buttons.forEach((btn) => {
                if (btn.disabled) return;
                btn.disabled = true;
                btn.dataset.originalText = btn.innerHTML;
                btn.innerHTML = `<span class="p1-spinner"></span>${labelText}`;
            });
        });
    });
})();
