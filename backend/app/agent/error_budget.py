from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ErrorBudget:
    global_left: int
    per_step: int
    _step_counts: dict[str, int] = field(default_factory=dict)

    def consume(self, step_key: str) -> bool:
        used = self._step_counts.get(step_key, 0)
        if self.global_left <= 0 or used >= self.per_step:
            return False
        self._step_counts[step_key] = used + 1
        self.global_left -= 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.global_left <= 0