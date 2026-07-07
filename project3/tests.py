import numpy as np
from django.test import TestCase

from .services.data import (
    get_agnews, class_distribution, CLASS_NAMES, N_CLASSES,
)


# ── Stage P3-1: AG News data service ───────────────────────────────────────

class AGNewsDataTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = get_agnews()

    def test_four_classes(self):
        self.assertEqual(self.data.n_classes, 4)
        self.assertEqual(self.data.class_names, ["World", "Sports", "Business", "Sci/Tech"])

    def test_split_sizes(self):
        self.assertEqual(self.data.n_train, 120000)
        self.assertEqual(self.data.n_test, 7600)

    def test_texts_are_strings(self):
        self.assertIsInstance(self.data.X_train[0], str)
        self.assertIsInstance(self.data.X_test[0], str)
        self.assertGreater(len(self.data.X_train[0]), 0)

    def test_labels_in_valid_range(self):
        self.assertTrue(set(np.unique(self.data.y_train)).issubset({0, 1, 2, 3}))
        self.assertTrue(set(np.unique(self.data.y_test)).issubset({0, 1, 2, 3}))

    def test_train_and_test_lengths_match_labels(self):
        self.assertEqual(len(self.data.X_train), len(self.data.y_train))
        self.assertEqual(len(self.data.X_test), len(self.data.y_test))

    def test_cached_returns_same_object(self):
        # lru_cache should hand back the identical instance
        self.assertIs(get_agnews(), self.data)


class ClassDistributionTest(TestCase):
    def test_counts_each_class(self):
        y = np.array([0, 0, 1, 2, 3, 3, 3])
        dist = class_distribution(y)
        self.assertEqual(dist["World"], 2)
        self.assertEqual(dist["Sports"], 1)
        self.assertEqual(dist["Business"], 1)
        self.assertEqual(dist["Sci/Tech"], 3)

    def test_all_classes_present_even_if_zero(self):
        y = np.array([0, 0, 0])
        dist = class_distribution(y)
        self.assertEqual(set(dist.keys()), set(CLASS_NAMES))
        self.assertEqual(dist["Sports"], 0)


# ── View / integration ──────────────────────────────────────────────────────

class IndexViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project3/")
        self.assertEqual(response.status_code, 200)

    def test_shows_dataset_info(self):
        response = self.client.get("/project3/")
        self.assertContains(response, "AG News")
        self.assertContains(response, "120000")
        self.assertContains(response, "7600")

    def test_lists_all_classes(self):
        response = self.client.get("/project3/")
        for cls in CLASS_NAMES:
            self.assertContains(response, cls)

    def test_explains_learning_to_defer(self):
        response = self.client.get("/project3/")
        self.assertContains(response, "Learning to Defer")
        self.assertContains(response, "defer")


class HomePageProject3LinkTest(TestCase):
    def test_home_lists_project3(self):
        response = self.client.get("/home/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Project 3")


# ── Stage P3-2: baseline classifier (Task 1) ───────────────────────────────

class BaselineServiceTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from .services.baseline import get_baseline_eval
        cls.ev = get_baseline_eval()

    def test_accuracy_is_strong(self):
        self.assertGreater(self.ev["accuracy"], 0.85)
        self.assertLessEqual(self.ev["accuracy"], 1.0)

    def test_confusion_matrix_is_4x4(self):
        cm = self.ev["confusion_matrix"]
        self.assertEqual(len(cm), 4)
        for row in cm:
            self.assertEqual(len(row), 4)

    def test_confusion_matrix_totals_match_test_size(self):
        total = sum(sum(row) for row in self.ev["confusion_matrix"])
        self.assertEqual(total, self.ev["n_test"])

    def test_per_class_accuracy(self):
        pca = self.ev["per_class_accuracy"]
        self.assertEqual(len(pca), 4)
        for a in pca:
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)

    def test_pipeline_predicts_proba(self):
        import numpy as np
        from .services.baseline import get_baseline
        pipe = get_baseline()
        proba = pipe.predict_proba([
            "The team won the championship final last night in overtime.",
            "Shares fell sharply as the central bank raised interest rates.",
        ])
        self.assertEqual(proba.shape, (2, 4))
        np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_pipeline_predicts_sports_for_sports_text(self):
        from .services.baseline import get_baseline
        from .services.data import CLASS_NAMES
        pipe = get_baseline()
        pred = pipe.predict(["The football team scored a goal to win the match."])[0]
        self.assertEqual(CLASS_NAMES[int(pred)], "Sports")


class BaselineViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project3/baseline/")
        self.assertEqual(response.status_code, 200)

    def test_shows_model_and_accuracy(self):
        response = self.client.get("/project3/baseline/")
        self.assertContains(response, "TF-IDF")
        self.assertContains(response, "Test accuracy")

    def test_shows_confusion_matrix(self):
        response = self.client.get("/project3/baseline/")
        self.assertContains(response, "Confusion Matrix")
        for cls in ["World", "Sports", "Business", "Sci/Tech"]:
            self.assertContains(response, cls)

    def test_has_perclass_chart(self):
        response = self.client.get("/project3/baseline/")
        self.assertContains(response, 'id="perclass-chart"')
        self.assertContains(response, "baseline_chart.js")

    def test_nav_links_to_baseline(self):
        response = self.client.get("/project3/")
        self.assertContains(response, "/project3/baseline/")


# ── Stage P3-3: simulated expert (Task 2) ──────────────────────────────────

class ExpertSimulationTest(TestCase):
    def test_region_specific_competence(self):
        # In-competence classes should be predicted correctly far more often.
        import numpy as np
        from .services.expert import simulate_expert, COMPETENCE_IDS, P_HIGH, P_LOW
        y = np.array([2] * 2000 + [0] * 2000)  # 2000 Business (comp), 2000 World (non-comp)
        pred = simulate_expert(y, seed=0)
        comp_acc = (pred[:2000] == 2).mean()
        noncomp_acc = (pred[2000:] == 0).mean()
        self.assertGreater(comp_acc, 0.85)      # ~P_HIGH
        self.assertLess(noncomp_acc, 0.6)       # ~P_LOW
        self.assertGreater(comp_acc, noncomp_acc)

    def test_not_a_perfect_oracle(self):
        import numpy as np
        from .services.expert import simulate_expert
        y = np.array([2] * 1000)  # even in competence, not 100%
        pred = simulate_expert(y, seed=1)
        self.assertLess((pred == 2).mean(), 1.0)

    def test_predictions_are_valid_labels(self):
        import numpy as np
        from .services.expert import simulate_expert
        pred = simulate_expert(np.array([0, 1, 2, 3] * 100), seed=2)
        self.assertTrue(set(np.unique(pred)).issubset({0, 1, 2, 3}))

    def test_reproducible_with_seed(self):
        import numpy as np
        from .services.expert import simulate_expert
        y = np.array([0, 1, 2, 3] * 250)
        np.testing.assert_array_equal(
            simulate_expert(y, seed=5), simulate_expert(y, seed=5)
        )

    def test_wrong_predictions_differ_from_true(self):
        import numpy as np
        from .services.expert import simulate_expert
        y = np.array([1] * 1000)
        pred = simulate_expert(y, seed=3)
        wrong = pred[pred != 1]
        self.assertTrue(set(np.unique(wrong)).issubset({0, 2, 3}))


class ExpertEvalTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from .services.expert import get_expert_eval
        cls.ev = get_expert_eval()

    def test_imperfect_overall(self):
        self.assertLess(self.ev["overall_accuracy"], 0.9)
        self.assertGreater(self.ev["overall_accuracy"], 0.5)

    def test_expert_beats_classifier_in_competence(self):
        # The expert should be better on exactly its competence topics.
        for c in self.ev["comparison"]:
            if c["in_competence"]:
                self.assertTrue(c["expert_better"], msg=f"{c['cls']} should favor expert")

    def test_classifier_beats_expert_outside_competence(self):
        for c in self.ev["comparison"]:
            if not c["in_competence"]:
                self.assertFalse(c["expert_better"], msg=f"{c['cls']} should favor classifier")

    def test_competence_topics_are_business_and_scitech(self):
        self.assertEqual(set(self.ev["competence_topics"]), {"Business", "Sci/Tech"})


class ExpertViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project3/expert/")
        self.assertEqual(response.status_code, 200)

    def test_shows_description_and_accuracy(self):
        response = self.client.get("/project3/expert/")
        self.assertContains(response, "Topic-specialist")
        self.assertContains(response, "Overall test accuracy")

    def test_has_comparison_chart(self):
        response = self.client.get("/project3/expert/")
        self.assertContains(response, 'id="compare-chart"')
        self.assertContains(response, "expert_chart.js")

    def test_shows_examples(self):
        response = self.client.get("/project3/expert/")
        self.assertContains(response, "Where Deferring Helps")
        self.assertContains(response, "Where Deferring Hurts")

    def test_nav_links_to_expert(self):
        response = self.client.get("/project3/")
        self.assertContains(response, "/project3/expert/")


# ── Stage P3-4: learning to defer (Task 3) ─────────────────────────────────

class DeferralMetricsTest(TestCase):
    def test_team_uses_expert_when_deferred(self):
        import numpy as np
        from .services.defer import deferral_metrics
        y = np.array([0, 1, 2, 3])
        clf = np.array([0, 1, 0, 0])   # right on 0,1; wrong on 2,3
        exp = np.array([3, 3, 2, 3])   # right on 2,3; wrong on 0,1
        defer = np.array([False, False, True, True])
        m = deferral_metrics(y, clf, exp, defer)
        # team = [clf0, clf1, exp2, exp3] = [0,1,2,3] -> all correct
        self.assertEqual(m["team_accuracy"], 1.0)
        self.assertEqual(m["deferral_rate"], 0.5)

    def test_oracle_is_upper_bound(self):
        import numpy as np
        from .services.defer import deferral_metrics
        y = np.array([0, 1, 2, 3])
        clf = np.array([0, 1, 0, 0])
        exp = np.array([3, 3, 2, 3])
        m = deferral_metrics(y, clf, exp, np.array([False, False, True, True]))
        self.assertGreaterEqual(m["oracle_accuracy"], m["team_accuracy"])

    def test_useful_and_harmful_counts(self):
        import numpy as np
        from .services.defer import deferral_metrics
        y = np.array([0, 1, 2, 3])
        clf = np.array([9, 1, 2, 0])   # wrong,right,right,wrong
        exp = np.array([0, 9, 9, 3])   # right,wrong,wrong,right
        defer = np.array([True, True, False, True])
        m = deferral_metrics(y, clf, exp, defer)
        # deferred idx 0 (exp right, clf wrong -> useful), idx1 (exp wrong, clf right -> harmful), idx3 (both? clf wrong exp right -> useful)
        self.assertAlmostEqual(m["useful_deferral_frac"], 2 / 3)
        self.assertAlmostEqual(m["harmful_deferral_frac"], 1 / 3)


class DeferralResultTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from .services.defer import get_deferral
        cls.r = get_deferral()

    def test_advantage_beats_classifier_alone(self):
        adv = self.r["advantage"]
        self.assertGreater(adv["team_accuracy"], adv["classifier_accuracy"])

    def test_advantage_beats_expert_alone(self):
        adv = self.r["advantage"]
        self.assertGreater(adv["team_accuracy"], adv["expert_accuracy"])

    def test_advantage_beats_confidence_baseline(self):
        self.assertGreater(
            self.r["advantage"]["team_accuracy"],
            self.r["confidence"]["team_accuracy"],
        )

    def test_team_below_oracle(self):
        adv = self.r["advantage"]
        self.assertLess(adv["team_accuracy"], adv["oracle_accuracy"])

    def test_defers_something_but_not_everything(self):
        rate = self.r["advantage"]["deferral_rate"]
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_has_example_articles(self):
        self.assertGreater(len(self.r["deferred_examples"]), 0)
        self.assertGreater(len(self.r["kept_examples"]), 0)


class DeferViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project3/defer/")
        self.assertEqual(response.status_code, 200)

    def test_shows_strategies_and_metrics(self):
        response = self.client.get("/project3/defer/")
        self.assertContains(response, "Expert-advantage")
        self.assertContains(response, "Team accuracy")
        self.assertContains(response, "Deferral rate")

    def test_has_summary_chart(self):
        response = self.client.get("/project3/defer/")
        self.assertContains(response, 'id="summary-chart"')
        self.assertContains(response, "defer_chart.js")

    def test_shows_deferred_and_kept(self):
        response = self.client.get("/project3/defer/")
        self.assertContains(response, "Deferred Articles")
        self.assertContains(response, "Kept Articles")

    def test_nav_links_to_defer(self):
        response = self.client.get("/project3/")
        self.assertContains(response, "/project3/defer/")
