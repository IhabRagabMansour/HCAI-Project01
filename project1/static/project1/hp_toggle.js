(function () {
    const select = document.getElementById("id_algorithm");
    const sections = document.querySelectorAll(".p1-hp-section");
    const modeRadios = document.querySelectorAll('input[name="training_mode"]');
    const manualSection = document.querySelector(".p1-manual-hp-section");
    const randomSearchSection = document.querySelector(".p1-random-search-section");
    if (!select || !sections.length) return;

    function update() {
        const v = select.value;
        sections.forEach((s) => {
            s.style.display = s.dataset.algorithm === v ? "" : "none";
        });

        if (manualSection || randomSearchSection) {
            const selectedMode = document.querySelector('input[name="training_mode"]:checked')?.value || "manual";
            if (manualSection) {
                manualSection.style.display = selectedMode === "manual" ? "" : "none";
            }
            if (randomSearchSection) {
                randomSearchSection.style.display = selectedMode === "random_search" ? "" : "none";
            }
        }
    }

    select.addEventListener("change", update);
    modeRadios.forEach((radio) => radio.addEventListener("change", update));
    update();
})();
