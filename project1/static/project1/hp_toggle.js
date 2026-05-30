(function () {
    const select = document.getElementById("id_algorithm");
    const sections = document.querySelectorAll(".p1-hp-section");
    if (!select || !sections.length) return;

    function update() {
        const v = select.value;
        sections.forEach((s) => {
            s.style.display = s.dataset.algorithm === v ? "" : "none";
        });
    }

    select.addEventListener("change", update);
    update();
})();
