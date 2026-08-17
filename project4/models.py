"""Study data models.

Only pseudonymous data is stored: a random participant code, the counterbalancing
group, interaction logs, and questionnaire answers. No names, emails, or other
direct identifiers are collected anywhere in the study (see the ethics section of
the protocol).
"""

from __future__ import annotations

import secrets

from django.db import models

from .services.study import (
    CONDITION_ORDERS, FIRST_STEP, PAIRWISE, RANKING, STEP_SEQUENCE,
)

STEP_CHOICES = [(s, s) for s in STEP_SEQUENCE]
ORDER_CHOICES = [(o, o) for o in CONDITION_ORDERS]
CONDITION_CHOICES = [(PAIRWISE, "Pairwise"), (RANKING, "Ranking")]

# Which part of the study a trial belongs to.
BLOCK_PRACTICE = "practice"
BLOCK_MAIN = "main"
BLOCK_HELDOUT = "heldout"
BLOCK_CHOICES = [
    (BLOCK_PRACTICE, "Practice"),
    (BLOCK_MAIN, "Main"),
    (BLOCK_HELDOUT, "Held-out"),
]


def generate_participant_code() -> str:
    """Short, unguessable, non-identifying participant code."""
    return secrets.token_hex(6)


class StudySession(models.Model):
    """One participant's run through the study."""

    participant_code = models.CharField(
        max_length=32, unique=True, default=generate_participant_code,
    )
    condition_order = models.CharField(max_length=2, choices=ORDER_CHOICES)
    seed = models.PositiveIntegerField()          # regenerates the trial plan
    current_step = models.CharField(
        max_length=32, choices=STEP_CHOICES, default=FIRST_STEP,
    )
    consented = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"{self.participant_code} ({self.condition_order})"

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None


class PairwiseTrial(models.Model):
    """One pairwise choice (used for practice, the main block, and held-out)."""

    session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="pairwise_trials",
    )
    block = models.CharField(max_length=16, choices=BLOCK_CHOICES, default=BLOCK_MAIN)
    task_index = models.PositiveIntegerField()
    left_movie_id = models.PositiveIntegerField()
    right_movie_id = models.PositiveIntegerField()
    chosen_movie_id = models.PositiveIntegerField()
    response_time_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["session_id", "block", "task_index"]
        unique_together = [("session", "block", "task_index")]

    @property
    def rejected_movie_id(self) -> int:
        return (
            self.right_movie_id
            if self.chosen_movie_id == self.left_movie_id
            else self.left_movie_id
        )


class RankingTrial(models.Model):
    """One ten-movie ranking, stored as shown ids plus the submitted order."""

    session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="ranking_trials",
    )
    block = models.CharField(max_length=16, choices=BLOCK_CHOICES, default=BLOCK_MAIN)
    task_index = models.PositiveIntegerField()
    movie_ids = models.JSONField()            # movies as displayed
    ranked_movie_ids = models.JSONField()     # best first
    response_time_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["session_id", "block", "task_index"]
        unique_together = [("session", "block", "task_index")]


class QuestionnaireResponse(models.Model):
    """Background, per-condition, and final questionnaires."""

    KIND_BACKGROUND = "background"
    KIND_CONDITION = "condition"
    KIND_FINAL = "final"
    KIND_CHOICES = [
        (KIND_BACKGROUND, "Background"),
        (KIND_CONDITION, "Condition"),
        (KIND_FINAL, "Final"),
    ]

    session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="questionnaires",
    )
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    condition = models.CharField(
        max_length=16, choices=CONDITION_CHOICES, null=True, blank=True,
    )
    answers = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["session_id", "created_at"]
