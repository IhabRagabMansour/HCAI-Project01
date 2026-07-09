from django.db import models


class HumanLabel(models.Model):
    """A label provided by a real user acting as the expert (Task 5 demo).

    Stores which pool article was shown, the true label, and the user's choice,
    so we can display progress and the user's accuracy. This mirrors how real
    human queries would feed the active-learning pipeline.
    """

    article_index = models.PositiveIntegerField()   # index into the demo query pool
    true_label = models.PositiveIntegerField()
    chosen_label = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    @property
    def is_correct(self) -> bool:
        return self.chosen_label == self.true_label
