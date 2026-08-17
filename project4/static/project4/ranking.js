// Reordering for the ten-film ranking task.
//
// Two ways to move a film, because drag-and-drop alone is not accessible:
//
//   * drag the row with the mouse or a touch-capable pointer, or
//   * press the up/down buttons, which are real focusable buttons and so work
//     with the keyboard and with a screen reader.
//
// This script is progressive enhancement. The buttons are submit buttons, so
// with JavaScript disabled each press posts the move and the server performs
// exactly the same swap; here we intercept the click and do it in the page
// instead. The order is mirrored into a hidden field on every change, and the
// server still re-checks that it is a permutation of what it sent.
(function () {
    const list = document.getElementById("p4-rank-list");
    const field = document.getElementById("ranked_movie_ids");
    const status = document.getElementById("p4-rank-status");
    if (!list || !field) return;

    const items = () => Array.from(list.querySelectorAll(".p4-rank-item"));

    // Buttons inside a draggable row can otherwise swallow the click.
    list.querySelectorAll(".p4-rank-move").forEach((button) => {
        button.draggable = false;
    });

    function sync() {
        const rows = items();
        rows.forEach((row, index) => {
            const number = row.querySelector(".p4-rank-number");
            if (number) number.textContent = String(index + 1);

            // Nothing above the first row, nothing below the last.
            row.querySelectorAll(".p4-rank-move").forEach((button) => {
                const up = button.dataset.direction === "up";
                button.disabled = up ? index === 0 : index === rows.length - 1;
            });
        });
        field.value = rows.map((row) => row.dataset.movieId).join(",");
    }

    function announce(row) {
        if (!status) return;
        const rows = items();
        const title = row.querySelector(".p4-movie-title");
        status.textContent =
            (title ? title.textContent.trim() : "Film") +
            " moved to position " + (rows.indexOf(row) + 1) + " of " + rows.length + ".";
    }

    // ── Move buttons ────────────────────────────────────────────────────────

    list.addEventListener("click", (event) => {
        const button = event.target.closest(".p4-rank-move");
        if (!button) return;

        event.preventDefault();          // do not submit — reorder in place
        const row = button.closest(".p4-rank-item");
        if (!row) return;

        if (button.dataset.direction === "up") {
            if (row.previousElementSibling) {
                list.insertBefore(row, row.previousElementSibling);
            }
        } else if (row.nextElementSibling) {
            list.insertBefore(row.nextElementSibling, row);
        }

        sync();
        announce(row);
        // Keep focus on the button that moved, so repeated presses keep working.
        if (!button.disabled) button.focus();
        else {
            const sibling = row.querySelector(".p4-rank-move:not([disabled])");
            if (sibling) sibling.focus();
        }
    });

    // ── Drag and drop ───────────────────────────────────────────────────────

    let dragging = null;

    // The row the pointer is currently above, or null to append at the end.
    function rowAfter(y) {
        return items()
            .filter((row) => row !== dragging)
            .find((row) => {
                const box = row.getBoundingClientRect();
                return y < box.top + box.height / 2;
            }) || null;
    }

    list.addEventListener("dragstart", (event) => {
        const row = event.target.closest(".p4-rank-item");
        if (!row) return;
        dragging = row;
        row.classList.add("p4-rank-item--dragging");
        if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/plain", row.dataset.movieId || "");
        }
    });

    list.addEventListener("dragover", (event) => {
        if (!dragging) return;
        event.preventDefault();          // required to allow a drop
        if (event.dataTransfer) event.dataTransfer.dropEffect = "move";

        const target = rowAfter(event.clientY);
        if (target !== dragging.nextElementSibling) {
            list.insertBefore(dragging, target);
        }
    });

    list.addEventListener("drop", (event) => event.preventDefault());

    list.addEventListener("dragend", () => {
        if (!dragging) return;
        dragging.classList.remove("p4-rank-item--dragging");
        sync();
        announce(dragging);
        dragging = null;
    });

    sync();
})();
