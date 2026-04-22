"""Internal pipeline dataclasses used to normalize orchestration inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Normalized config container for internal pipeline orchestration."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "ResolvedRunConfig":
        return cls(raw=dict(config or {}))

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class SingleSceneRequest:
    """Single-scene orchestration payload."""

    hs_file: str
    hyp_type: str
    output_dir: str
    config: ResolvedRunConfig


@dataclass(frozen=True)
class BatchSceneRequest:
    """Batch orchestration payload."""

    input_dir: str
    output_dir: str
    config: ResolvedRunConfig
