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
