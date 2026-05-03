"""Project configuration models and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

try:  # pragma: no cover - depends on interpreter version
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


@dataclass(slots=True)
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"
    logs_dir: str = "data/logs"
    reports_dir: str = "reports/output"


@dataclass(slots=True)
class DataConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2026-04-30"
    benchmark_symbol: str = "^GSPC"
    vix_symbol: str = "^VIX"
    risk_free_series: str = "DGS3MO"
    refresh: bool = False


@dataclass(slots=True)
class QualityConfig:
    max_forward_fill_days: int = 5
    min_price_history_days: int = 252
    stale_price_days: int = 2
    ipo_buffer_days: int = 63
    stale_fundamental_days: int = 183
    max_missing_feature_ratio: float = 0.20


@dataclass(slots=True)
class UniverseConfig:
    source: str = "wikipedia_sp500"
    membership_min_reliability: float = 0.70


@dataclass(slots=True)
class FeatureConfig:
    normalization: str = "zscore"
    random_noise_feature_count: int = 3


@dataclass(slots=True)
class RecommendationConfig:
    buy_fraction: float = 0.10
    min_buy_count: int = 1
    caution_max_buys: int = 5
    closed_vix: float = 35.0
    caution_vix: float = 25.0


@dataclass(slots=True)
class ValidationSplitConfig:
    train_end: str
    val_start: str
    val_end: str


@dataclass(slots=True)
class ValidationConfig:
    embargo_days: int = 42
    holdout_start: str = "2025-01-01"
    holdout_end: str = "2026-04-30"
    splits: list[ValidationSplitConfig] = field(default_factory=list)


@dataclass(slots=True)
class BacktestConfig:
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "daily"


@dataclass(slots=True)
class BaselineModelConfig:
    positive_class_quantile: float = 0.80
    C: float = 1.0
    max_iter: int = 500


@dataclass(slots=True)
class RankerModelConfig:
    backend: str = "lightgbm"
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    feature_fraction: float = 0.70
    bagging_fraction: float = 0.70
    bagging_freq: int = 5
    min_child_samples: int = 50
    max_depth: int = 6
    early_stopping_rounds: int = 50


@dataclass(slots=True)
class ModelConfig:
    baseline: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    ranker: RankerModelConfig = field(default_factory=RankerModelConfig)


@dataclass(slots=True)
class ProjectConfig:
    root_dir: Path
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    horizons: dict[str, int] = field(default_factory=lambda: {
        "1d": 1,
        "1w": 5,
        "2w": 10,
        "1m": 21,
        "2m": 42,
    })

    @property
    def raw_dir(self) -> Path:
        return _resolve_path(self.root_dir, self.paths.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return _resolve_path(self.root_dir, self.paths.processed_dir)

    @property
    def models_dir(self) -> Path:
        return _resolve_path(self.root_dir, self.paths.models_dir)

    @property
    def logs_dir(self) -> Path:
        return _resolve_path(self.root_dir, self.paths.logs_dir)

    @property
    def reports_dir(self) -> Path:
        return _resolve_path(self.root_dir, self.paths.reports_dir)

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.logs_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _build_validation_splits(items: list[dict[str, str]]) -> list[ValidationSplitConfig]:
    return [ValidationSplitConfig(**item) for item in items]


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if expanded.is_absolute():
        return expanded
    return root_dir / expanded


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path).resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    paths = PathsConfig(**raw.get("paths", {}))
    data = DataConfig(**raw.get("data", {}))
    quality = QualityConfig(**raw.get("quality", {}))
    universe = UniverseConfig(**raw.get("universe", {}))
    features = FeatureConfig(**raw.get("features", {}))
    recommendation = RecommendationConfig(**raw.get("recommendation", {}))
    validation_raw = raw.get("validation", {})
    validation = ValidationConfig(
        embargo_days=validation_raw.get("embargo_days", 42),
        holdout_start=validation_raw.get("holdout_start", "2025-01-01"),
        holdout_end=validation_raw.get("holdout_end", "2026-04-30"),
        splits=_build_validation_splits(validation_raw.get("splits", [])),
    )
    backtest = BacktestConfig(**raw.get("backtest", {}))
    model_raw = raw.get("model", {})
    model = ModelConfig(
        baseline=BaselineModelConfig(**model_raw.get("baseline", {})),
        ranker=RankerModelConfig(**model_raw.get("ranker", {})),
    )
    horizons = {str(key): int(value) for key, value in raw.get("horizons", {}).items()}

    cfg = ProjectConfig(
        root_dir=config_path.parent.parent,
        paths=paths,
        data=data,
        quality=quality,
        universe=universe,
        features=features,
        recommendation=recommendation,
        validation=validation,
        backtest=backtest,
        model=model,
        horizons=horizons or ProjectConfig(root_dir=config_path.parent.parent).horizons,
    )
    cfg.ensure_directories()
    return cfg
