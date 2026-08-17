// Records how long the participant took on a trial.
//
// The clock starts when the trial is rendered and the elapsed time is written
// into a hidden field just before the form is submitted. The server treats this
// as advisory and validates the range, since a client value can never be trusted.
(function () {
    const field = document.getElementById("response_time_ms");
    if (!field) return;

    const form = field.form;
    if (!form) return;

    const start = performance.now();

    form.addEventListener("submit", () => {
        field.value = String(Math.round(performance.now() - start));
    });
})();
