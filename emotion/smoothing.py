"""Temporal smoothing (kept separate from DeepFace so it is easy to test and explain)."""

from collections import Counter, deque


class TemporalSmoother:
    """
    Keeps the last N predictions and shows the MOST COMMON one, so a single
    odd frame cannot flip the displayed result.

        Happy, Happy, Neutral, Happy, Happy  ->  displayed: Happy (4 of 5)

    It also averages the model's probabilities over the same window, so the
    confidence shown on screen matches the expression that is displayed.
    """

    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)  # items: (label, scores_dict)

    def add(self, emotion_label, scores=None):
        self.history.append((emotion_label, scores or {}))

    def get_smoothed(self):
        if not self.history:
            return "N/A"
        labels = [label for label, _ in self.history]
        return Counter(labels).most_common(1)[0][0]

    def get_consistency_ratio(self):
        """Share (0-1) of recent predictions that agree with the smoothed result."""
        if not self.history:
            return 0.0
        smoothed = self.get_smoothed()
        return sum(1 for label, _ in self.history if label == smoothed) / len(self.history)

    def get_average_scores(self):
        """Average probability (0-100) of each expression over the window."""
        totals, count = {}, 0
        for _, scores in self.history:
            if not scores:
                continue
            count += 1
            for name, value in scores.items():
                totals[name] = totals.get(name, 0.0) + value
        return {name: round(total / count, 1) for name, total in totals.items()} if count else {}

    def get_confidence(self):
        """Average model probability of the DISPLAYED expression (0-100)."""
        return self.get_average_scores().get(self.get_smoothed(), 0.0)

    def reset(self):
        self.history.clear()
