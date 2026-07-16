"""Precompute every Project 3 artifact in one step.

Usage:
    python manage.py p3_build            # build whatever is missing
    python manage.py p3_build --force    # delete cached artifacts and rebuild

Builds, in dependency order: the AG News cache, the baseline classifier and its
test evaluation, the simulated expert predictions and evaluation, the
learning-to-defer results, the active-learning curves, the human-demo query
pool, and finally renders the PDF report once as a sanity check. After this
command finishes, every Project 3 page loads instantly from cache.
"""

from __future__ import annotations

import shutil
import time

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Precompute all Project 3 artifacts (dataset, models, results, report)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force", action="store_true",
            help="Delete computed artifacts (models/results) first and rebuild.",
        )
        parser.add_argument(
            "--force-data", action="store_true",
            help="Also delete the AG News dataset cache (forces re-download).",
        )

    def _step(self, label, fn):
        t0 = time.perf_counter()
        result = fn()
        self.stdout.write(f"  [ok] {label}  ({time.perf_counter() - t0:.1f}s)")
        return result

    def handle(self, *args, **options):
        from project3.services import active, baseline, data, defer, expert

        if options["force"]:
            shutil.rmtree(baseline.ARTIFACT_DIR, ignore_errors=True)
            self.stdout.write("Cleared computed artifacts.")
        if options["force_data"]:
            shutil.rmtree(data.CACHE_DIR, ignore_errors=True)
            self.stdout.write("Cleared dataset cache.")

        self.stdout.write("Building Project 3 artifacts...")
        self._step("AG News dataset", data.get_agnews)
        self._step("Baseline classifier (fit)", baseline.get_baseline)
        self._step("Baseline evaluation", baseline.get_baseline_eval)
        self._step("Expert train predictions", expert.get_expert_train_predictions)
        self._step("Expert test predictions", expert.get_expert_test_predictions)
        self._step("Expert evaluation", expert.get_expert_eval)
        self._step("Learning-to-defer results", defer.get_deferral)
        self._step("Active-learning curves", active.get_active)
        self._step("Human-demo query pool", active.get_demo_query_pool)

        # Render the PDF once as an end-to-end sanity check.
        from project3.services.report import build_report_pdf
        pdf = self._step("PDF report", build_report_pdf)
        if not pdf.startswith(b"%PDF-"):
            self.stderr.write("WARNING: report did not render as a valid PDF")

        self.stdout.write(self.style.SUCCESS("All Project 3 artifacts ready."))
