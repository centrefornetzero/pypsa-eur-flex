# SPDX-FileCopyrightText: 2023-2026 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""Generate scenario overrides from an experiment orchestrator config.

Expected config structure:

experiment_orchestrator:
  planning_horizons: [2030, 2040, 2050]
  flex_options:
    flex00_baseline: {sector: {bev_dsm: false, bev_dsm_availability: 0.0, v2g: false}}
    ...
  policy_scenarios:
    baseline: {}
    no_h2: {sector: {hydrogen_fuel_cell: false, hydrogen_turbine: false}}
"""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge dictionaries without mutating inputs."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_runtime() -> tuple[dict, Path]:
    if "snakemake" in globals():
        smk: Any = globals()["snakemake"]
        cfg = smk.config
        output_path = Path(smk.output[0])
        return cfg, output_path

    # Local fallback for direct execution in development.
    cfg_path = Path(__file__).resolve().parent / "config.ian_2026.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    scenario_file = (
        cfg.get("run", {}).get("scenarios", {}).get("file")
        or "config/scenarios.experiments.yaml"
    )
    output_path = (Path(__file__).resolve().parents[1] / scenario_file).resolve()
    return cfg, output_path


def _build_scenarios(config: dict) -> OrderedDict[str, dict]:
    orchestrator = config.get("experiment_orchestrator", {})
    flex_options = orchestrator.get("flex_options", {})
    policy_scenarios = orchestrator.get("policy_scenarios", {})
    planning_horizons = orchestrator.get("planning_horizons")
    name_separator = orchestrator.get("name_separator", "__")

    if not flex_options:
        raise ValueError("Missing experiment_orchestrator.flex_options")
    if not policy_scenarios:
        raise ValueError("Missing experiment_orchestrator.policy_scenarios")
    if not planning_horizons:
        raise ValueError("Missing experiment_orchestrator.planning_horizons")

    scenarios = OrderedDict()
    for scenario_name, scenario_overrides in policy_scenarios.items():
        for flex_name, flex_overrides in flex_options.items():
            run_name = f"{scenario_name}{name_separator}{flex_name}"
            merged = _merge_dicts(scenario_overrides or {}, flex_overrides or {})
            merged = _merge_dicts(
                {"scenario": {"planning_horizons": planning_horizons}},
                merged,
            )
            scenarios[run_name] = merged

    return scenarios


def main() -> None:
    config, output_path = _load_runtime()
    scenarios = _build_scenarios(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(dict(scenarios), f, sort_keys=False)

    print(f"Generated {len(scenarios)} scenarios in {output_path}")


if __name__ == "__main__":
    main()
