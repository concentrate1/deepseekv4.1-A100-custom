"""Strict, dependency-free parsing of the reloadable DSpark runtime settings."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


DEFAULT_DSPARK_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "dspark_runtime.json"
)


class DSparkConfigError(ValueError):
    """The DSpark runtime config does not match the supported schema."""


@dataclass(frozen=True)
class DSparkConfig:
    draft_temperature: float = 0.0

    def __post_init__(self):
        value = self.draft_temperature
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DSparkConfigError("draft_temperature must be a finite, non-negative number")
        try:
            normalized = float(value)
        except OverflowError as exc:
            raise DSparkConfigError(
                "draft_temperature must be a finite, non-negative number"
            ) from exc
        if not math.isfinite(normalized) or normalized < 0:
            raise DSparkConfigError("draft_temperature must be a finite, non-negative number")
        object.__setattr__(self, "draft_temperature", normalized)


def _reject_constant(value: str):
    raise DSparkConfigError(f"non-JSON numeric constant is not allowed: {value}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DSparkConfigError(f"duplicate config key: {key}")
        result[key] = value
    return result


def load_dspark_config(path: str | Path = DEFAULT_DSPARK_CONFIG_PATH) -> DSparkConfig:
    """Read and validate the complete version-one DSpark config.

    Missing files and invalid content raise instead of silently using defaults, so
    a caller can preserve its already loaded component when a reload fails.
    """
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"),
                          parse_constant=_reject_constant,
                          object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise DSparkConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"draft_temperature"}:
        raise DSparkConfigError(
            "DSpark config must be an object containing only draft_temperature"
        )
    return DSparkConfig(draft_temperature=data["draft_temperature"])
