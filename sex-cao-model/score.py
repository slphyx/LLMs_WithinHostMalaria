#!/usr/bin/env python3
"""Score the qualified all-marker participant-block B0/B2 comparison.

For each participant, all of that participant's Pfs25 and PfMGET rows are
jointly deleted by PSIS while the complete 18S and piperaquine course remains
conditioned upon.  No sex-marker row from the held-out participant is used for
calibration.  The scorer checks the fit configuration and strict MCMC gate
before permitting a performance comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from arviz_stats.base import array_stats
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
QUALIFIED = HERE / "results/qualified_run"
RUN_CONTRACT = QUALIFIED / "run_contract.json"
PARTICIPANTS = tuple(
    str(value) for value in (*range(101, 107), *range(201, 205), *range(301, 308))
)
ARMS = ("b0", "b2")
POPULATION_PARAMETERS = (
    "popmean_ipl", "popmean_iplMu", "popmean_iplSigma", "popmean_pmf",
    "popmean_kmax", "popmean_EC50", "popmean_gamma", "popmean_f",
    "popmean_deltaP", "popmean_m", "popmean_deltaG", "popmean_deltaG5",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.resolve().glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CmdStan CSVs in {directory}")
    return paths


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#") and line.strip():
                return line.rstrip("\n").split(",")
    raise RuntimeError(f"No CSV header in {path}")


def cmdstan_metadata(path: Path) -> dict[str, Any]:
    patterns: dict[str, tuple[re.Pattern[str], Any]] = {
        "sampling": (re.compile(r"^#\s+num_samples = (\d+)"), int),
        "warmup": (re.compile(r"^#\s+num_warmup = (\d+)"), int),
        "adapt_delta": (re.compile(r"^#\s+delta = ([-+0-9.eE]+)"), float),
        "max_treedepth": (re.compile(r"^#\s+max_depth = (\d+)"), int),
        "metric": (re.compile(r"^#\s+metric = ([A-Za-z0-9_]+)"), str),
        "chain_id": (re.compile(r"^# id = (\d+)"), int),
        "seed": (re.compile(r"^#\s+seed = (\d+)"), int),
        "sig_figs": (re.compile(r"^#\s+sig_figs = (\d+)"), int),
    }
    result: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            for name, (pattern, converter) in patterns.items():
                match = pattern.match(line)
                if match:
                    result[name] = converter(match.group(1))
    missing = set(patterns) - result.keys()
    if missing:
        raise RuntimeError(f"Missing CmdStan metadata in {path}: {sorted(missing)}")
    return result


def expected_configuration(contract: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arm_contract = contract["arms"][arm]
    return {
        "seed": arm_contract["seed"],
        "chains": contract["chains"],
        "parallel_chains": contract["parallel_chains"],
        "threads_per_chain": contract["threads_per_chain"],
        "warmup": arm_contract["warmup"],
        "sampling": arm_contract["sampling"],
        "adapt_delta": arm_contract["adapt_delta"],
        "metric": contract["metric"],
        "max_treedepth": contract["max_treedepth"],
        "sig_figs": contract["sig_figs"],
    }


def validate_csv_configuration(
    paths: list[Path], contract: Mapping[str, Any], arm: str
) -> list[dict[str, Any]]:
    expected = expected_configuration(contract, arm)
    if len(paths) != expected["chains"]:
        raise RuntimeError(f"{arm} requires exactly {expected['chains']} posterior chains")
    metadata = [cmdstan_metadata(path) for path in paths]
    for path, observed in zip(paths, metadata, strict=True):
        for name in (
            "seed", "warmup", "sampling", "metric", "max_treedepth", "sig_figs"
        ):
            if observed[name] != expected[name]:
                raise RuntimeError(
                    f"{arm} {name} mismatch in {path}: {observed[name]} != {expected[name]}"
                )
        if not math.isclose(
            float(observed["adapt_delta"]), float(expected["adapt_delta"]), abs_tol=1e-12
        ):
            raise RuntimeError(f"{arm} adapt_delta mismatch in {path}")
    chain_ids = sorted(int(item["chain_id"]) for item in metadata)
    # CmdStanPy 1.3 may launch four single-chain processes whose CSV metadata
    # each reports the default id=1; the distinct manifest-listed files identify
    # the chains in that case.
    if chain_ids not in (
        [1] * expected["chains"],
        list(range(1, expected["chains"] + 1)),
    ):
        raise RuntimeError(f"{arm} has an unexpected CmdStan chain-id pattern")
    return metadata


def validate_fit_manifest(
    directory: Path,
    paths: list[Path],
    contract: Mapping[str, Any],
    arm: str,
    *,
    allow_unmanifested: bool,
) -> dict[str, Any]:
    expected_arm = contract["arms"][arm]
    manifest_path = directory / "fit_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("schema") != "sex-cao-fit-manifest-v1":
            raise RuntimeError(f"Unexpected fit manifest schema in {manifest_path}")
        if manifest.get("arm") != arm:
            raise RuntimeError(f"Manifest arm mismatch in {manifest_path}")
        if manifest.get("allocation_active") != expected_arm["allocation_active"]:
            raise RuntimeError(f"Manifest allocation flag mismatch in {manifest_path}")
        expected = expected_configuration(contract, arm)
        if manifest.get("configuration") != expected:
            raise RuntimeError(f"{arm} fit manifest has the wrong configuration")
        manifest_kind = "recorded_fit"
    elif allow_unmanifested:
        manifest = {"csv_files": []}
        manifest_kind = "unmanifested_override"
    else:
        raise RuntimeError(
            f"No fit_manifest.json in {directory}; "
            "use --allow-unmanifested only for a non-claim diagnostic"
        )

    listed = manifest.get("csv_files", [])
    if listed:
        by_name = {Path(item["path"]).name: item for item in listed}
        if set(by_name) != {path.name for path in paths}:
            raise RuntimeError(f"{arm} posterior CSV set differs from its manifest")
        for path in paths:
            record = by_name[path.name]
            if int(record["size"]) != path.stat().st_size:
                raise RuntimeError(f"Posterior CSV size mismatch: {path}")

    metadata = validate_csv_configuration(paths, contract, arm)
    return {
        "manifest_kind": manifest_kind,
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "cmdstan_metadata": metadata,
    }


def log_likelihood_by_chain(directory: Path, count: int) -> np.ndarray:
    columns = [f"log_lik_sex.{index}" for index in range(1, count + 1)]
    chains = [
        pd.read_csv(path, comment="#", usecols=columns).to_numpy(dtype=float)
        for path in csv_paths(directory)
    ]
    if len({chain.shape for chain in chains}) != 1:
        raise RuntimeError(f"Unequal chain shapes in {directory}")
    return np.stack(chains, axis=0)


def psis_weights(group_log_likelihood: np.ndarray) -> dict[str, Any]:
    values = np.asarray(group_log_likelihood, dtype=float)
    if values.ndim != 2 or values.size < 20 or not np.all(np.isfinite(values)):
        raise ValueError("PSIS input must be finite chain-by-draw data")
    sample_count = values.size
    likelihood = np.exp(values - float(np.max(values)))
    relative_efficiency = float(array_stats.ess(
        likelihood, chain_axis=0, draw_axis=1, method="bulk", relative=True
    ))
    relative_efficiency = float(np.clip(relative_efficiency, 1 / sample_count, 1))
    raw_log_weights = -values.reshape(-1)
    raw_normalized = raw_log_weights - logsumexp(raw_log_weights)
    psis_error = None
    try:
        log_weights, pareto_k = array_stats.psislw(
            raw_log_weights, r_eff=relative_efficiency, axis=-1
        )
        log_weights = np.asarray(log_weights, dtype=float)
        if not np.isfinite(float(pareto_k)) or not np.all(np.isfinite(log_weights)):
            log_weights = raw_normalized
            pareto_k = math.inf
            psis_error = "nonfinite Pareto-k or smoothed log weights"
    except (ValueError, FloatingPointError) as error:
        log_weights = raw_normalized
        pareto_k = math.inf
        psis_error = f"{type(error).__name__}: {error}"
    good_k = float(min(1 - 1 / math.log10(sample_count), 0.7))
    return {
        "log_weights": log_weights,
        "pareto_k": float(pareto_k),
        "good_k_threshold": good_k,
        "pareto_warning": bool(not np.isfinite(pareto_k) or pareto_k > good_k),
        "psis_error": psis_error,
        "relative_mcmc_efficiency": relative_efficiency,
        "samples": sample_count,
    }


def weighted_lpd(log_weights: np.ndarray, log_likelihood: np.ndarray) -> float:
    return float(logsumexp(log_weights + np.asarray(log_likelihood).reshape(-1)))


def split_rows(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the all-marker participant-deletion contract (zero calibration rows)."""

    subject = np.asarray(data["sex_subject"], dtype=int)
    male = np.asarray(data["sex_is_male"], dtype=int)
    result: dict[str, dict[str, Any]] = {}
    for subject_index, participant in enumerate(PARTICIPANTS, start=1):
        scored = np.flatnonzero(subject == subject_index)
        female = scored[male[scored] == 0]
        male_rows = scored[male[scored] == 1]
        if not len(scored) or not len(female) or not len(male_rows):
            raise RuntimeError(f"Incomplete sex-marker block for participant {participant}")
        result[participant] = {
            "calibration_rows": [],
            "scored_rows": scored.tolist(),
            "female_scored_rows": female.tolist(),
            "male_scored_rows": male_rows.tolist(),
        }
    all_scored = [index for value in result.values() for index in value["scored_rows"]]
    if sorted(all_scored) != list(range(int(data["nSexObs"]))):
        raise RuntimeError("All-marker split must score every sex-marker row exactly once")
    return result


def score_model(
    directory: Path,
    data: Mapping[str, Any],
    split: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_chain = log_likelihood_by_chain(directory, int(data["nSexObs"]))
    flattened = by_chain.reshape(-1, by_chain.shape[-1])
    participants: dict[str, Any] = {}
    for participant in PARTICIPANTS:
        contract = split[participant]
        scored = np.asarray(contract["scored_rows"], dtype=int)
        group = np.sum(by_chain[:, :, scored], axis=2)
        psis = psis_weights(group)
        weights = np.asarray(psis.pop("log_weights"), dtype=float)
        streams: dict[str, Any] = {}
        for sex in ("female", "male"):
            indices = np.asarray(contract[f"{sex}_scored_rows"], dtype=int)
            row_lpd = np.asarray([
                weighted_lpd(weights, flattened[:, index]) for index in indices
            ])
            joint_lpd = weighted_lpd(weights, np.sum(flattened[:, indices], axis=1))
            streams[sex] = {
                "scored_rows": int(len(indices)),
                "pointwise_lpd": float(np.sum(row_lpd)),
                "pointwise_mean_nlpd": float(-np.mean(row_lpd)),
                "joint_trajectory_lpd": joint_lpd,
                "joint_trajectory_mean_nlpd": float(-joint_lpd / len(indices)),
            }
        joint_lpd = weighted_lpd(weights, group.reshape(-1))
        participants[participant] = {
            "calibration_row_count": 0,
            "scored_row_count": int(len(scored)),
            "psis": psis,
            "streams": streams,
            "sex_equal_pointwise_mean_nlpd": float(
                0.5 * (
                    streams["female"]["pointwise_mean_nlpd"]
                    + streams["male"]["pointwise_mean_nlpd"]
                )
            ),
            "joint_all_sex_trajectory_lpd": joint_lpd,
            "joint_all_sex_trajectory_mean_nlpd": float(-joint_lpd / len(scored)),
        }
    primary = np.asarray([
        participants[p]["sex_equal_pointwise_mean_nlpd"] for p in PARTICIPANTS
    ])
    female = np.asarray([
        participants[p]["streams"]["female"]["pointwise_mean_nlpd"]
        for p in PARTICIPANTS
    ])
    male = np.asarray([
        participants[p]["streams"]["male"]["pointwise_mean_nlpd"]
        for p in PARTICIPANTS
    ])
    return {
        "counts": {
            "chains": int(by_chain.shape[0]),
            "draws_per_chain": int(by_chain.shape[1]),
            "available_sex_rows": int(data["nSexObs"]),
            "calibration_sex_rows": 0,
            "scored_sex_rows": int(sum(v["scored_row_count"] for v in participants.values())),
        },
        "primary": {
            "participant_sex_equal_pointwise_mean_nlpd": float(np.mean(primary)),
            "female_participant_equal_pointwise_mean_nlpd": float(np.mean(female)),
            "male_participant_equal_pointwise_mean_nlpd": float(np.mean(male)),
            "maximum_participant_block_pareto_k": float(
                max(v["psis"]["pareto_k"] for v in participants.values())
            ),
            "all_participant_blocks_psis_valid": not any(
                v["psis"]["pareto_warning"] for v in participants.values()
            ),
        },
        "participants": participants,
    }


def selected_parameter_columns(names: list[str]) -> list[str]:
    exact = {
        *POPULATION_PARAMETERS,
        "sigma_total", "log_amplitude_female", "log_amplitude_male",
        "log_scale", "rho_male", "tau_rho",
    }
    prefixes = (
        "L.", "omega.", "etaStd.", "log_sigma_sex.", "alpha_detect.",
        "detection_slope.", "rho_male_subject.", "z_rho.",
    )
    return [name for name in names if name in exact or name.startswith(prefixes)]


def mcmc_diagnostics(directory: Path, max_treedepth: int = 13) -> dict[str, Any]:
    paths = csv_paths(directory)
    names = header(paths[0])
    parameters = selected_parameter_columns(names)
    diagnostic_columns = ["energy__", "divergent__", "treedepth__"]
    frames = [
        pd.read_csv(path, comment="#", usecols=[*parameters, *diagnostic_columns])
        for path in paths
    ]
    if len({len(frame) for frame in frames}) != 1:
        raise RuntimeError("MCMC chains have unequal retained draw counts")
    maximum_rhat = -math.inf
    minimum_bulk = math.inf
    minimum_tail = math.inf
    for parameter in parameters:
        values = np.stack([frame[parameter].to_numpy(dtype=float) for frame in frames])
        if np.ptp(values) == 0:
            continue
        maximum_rhat = max(maximum_rhat, float(array_stats.rhat(
            values, chain_axis=0, draw_axis=1, method="rank"
        )))
        minimum_bulk = min(minimum_bulk, float(array_stats.ess(
            values, chain_axis=0, draw_axis=1, method="bulk"
        )))
        minimum_tail = min(minimum_tail, float(array_stats.ess(
            values, chain_axis=0, draw_axis=1, method="tail", prob=(0.05, 0.95)
        )))
    divergences = int(sum(frame["divergent__"].sum() for frame in frames))
    depth_hits = int(sum((frame["treedepth__"] >= max_treedepth).sum() for frame in frames))
    ebfmi = []
    for frame in frames:
        energy = frame["energy__"].to_numpy(dtype=float)
        variance = float(np.var(energy))
        ebfmi.append(float(np.mean(np.diff(energy) ** 2) / variance))
    passed = bool(
        maximum_rhat <= 1.01
        and minimum_bulk >= 400
        and minimum_tail >= 400
        and divergences == 0
        and depth_hits == 0
        and min(ebfmi) >= 0.30
    )
    return {
        "passed": passed,
        "sampled_parameters": len(parameters),
        "maximum_rhat": maximum_rhat,
        "minimum_bulk_ess": minimum_bulk,
        "minimum_tail_ess": minimum_tail,
        "divergences": divergences,
        "max_treedepth_hits": depth_hits,
        "ebfmi_by_chain": ebfmi,
        "thresholds": {
            "maximum_rhat": 1.01,
            "minimum_bulk_ess": 400,
            "minimum_tail_ess": 400,
            "divergences": 0,
            "max_treedepth_hits": 0,
            "minimum_ebfmi": 0.30,
        },
    }


def bootstrap(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    resampled = np.mean(values[indices], axis=1)
    return {
        "mean_nlpd_gain": float(np.mean(values)),
        "participant_bootstrap_95": [
            float(np.quantile(resampled, 0.025)),
            float(np.quantile(resampled, 0.975)),
        ],
        "strict_participant_wins": int(np.count_nonzero(values > 0)),
        "strict_participant_losses": int(np.count_nonzero(values < 0)),
        "participant_gains": dict(zip(PARTICIPANTS, map(float, values), strict=True)),
    }


def compare(
    b0: Mapping[str, Any], b2: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    score_contract = contract["score"]
    rng = np.random.default_rng(int(score_contract["bootstrap_seed"]))
    indices = rng.integers(
        0,
        len(PARTICIPANTS),
        size=(int(score_contract["bootstrap_replicates"]), len(PARTICIPANTS)),
    )
    primary = np.asarray([
        b0["participants"][p]["sex_equal_pointwise_mean_nlpd"]
        - b2["participants"][p]["sex_equal_pointwise_mean_nlpd"]
        for p in PARTICIPANTS
    ])
    joint = np.asarray([
        b0["participants"][p]["joint_all_sex_trajectory_mean_nlpd"]
        - b2["participants"][p]["joint_all_sex_trajectory_mean_nlpd"]
        for p in PARTICIPANTS
    ])
    sex_stratified = {}
    for sex in ("female", "male"):
        values = np.asarray([
            b0["participants"][p]["streams"][sex]["pointwise_mean_nlpd"]
            - b2["participants"][p]["streams"][sex]["pointwise_mean_nlpd"]
            for p in PARTICIPANTS
        ])
        sex_stratified[sex] = bootstrap(values, indices)
    return {
        "definition": "B0 NLPD minus B2 NLPD; positive favours B2",
        "primary": bootstrap(primary, indices),
        "secondary_joint_trajectory": bootstrap(joint, indices),
        "sex_stratified": sex_stratified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path, default=HERE / "outputs/qualified_refit/b0")
    parser.add_argument("--b2", type=Path, default=HERE / "outputs/qualified_refit/b2")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/qualified_score.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unmanifested",
        action="store_true",
        help="permit a diagnostic score without a fit manifest (never permits a claim)",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite {args.output}; pass --overwrite")
    return args


def main() -> None:
    args = parse_args()
    contract = read_json(RUN_CONTRACT)
    data_by_arm = {
        arm: read_json(QUALIFIED / contract["arms"][arm]["data"]) for arm in ARMS
    }
    nonflag = []
    for arm in ARMS:
        current = dict(data_by_arm[arm])
        current.pop("allocation_active")
        nonflag.append(json.dumps(current, sort_keys=True, separators=(",", ":")))
    if nonflag[0] != nonflag[1]:
        raise RuntimeError("Bundled B0/B2 data differ by more than allocation_active")
    data = data_by_arm["b2"]
    split = split_rows(data)
    directories = {"b0": args.b0.resolve(), "b2": args.b2.resolve()}
    paths = {arm: csv_paths(directory) for arm, directory in directories.items()}
    manifests = {
        arm: validate_fit_manifest(
            directories[arm],
            paths[arm],
            contract,
            arm,
            allow_unmanifested=args.allow_unmanifested,
        )
        for arm in ARMS
    }
    models = {arm: score_model(directories[arm], data, split) for arm in ARMS}
    convergence = {
        arm: mcmc_diagnostics(directories[arm], int(contract["max_treedepth"]))
        for arm in ARMS
    }
    comparison = compare(models["b0"], models["b2"], contract)
    psis_valid = all(
        model["primary"]["all_participant_blocks_psis_valid"]
        for model in models.values()
    )
    mcmc_valid = all(value["passed"] for value in convergence.values())
    manifest_valid = all(
        value["manifest_kind"] != "unmanifested_override" for value in manifests.values()
    )
    valid = psis_valid and mcmc_valid and manifest_valid
    status = (
        "valid_qualified_all_marker_mcmc_psis_comparison"
        if valid
        else "diagnostic_only_fit_manifest_missing"
        if not manifest_valid
        else "diagnostic_only_mcmc_gate_failed"
        if not mcmc_valid
        else "diagnostic_only_high_pareto_k_requires_exact_refit"
    )
    result = {
        "schema": "sex-cao-qualified-all-marker-score-v1",
        "status": status,
        "performance_claim_permitted": valid,
        "contract": {
            "cohort": "public 17-volunteer Cao development cohort",
            "deletion": contract["score"]["endpoint"],
            "calibration_sex_rows": 0,
            "scored_sex_rows": contract["score"]["sex_rows"],
            "conditioning": "complete participant 18S and piperaquine course",
            "weighting": contract["score"]["primary_weighting"],
            "claim_ceiling": (
                "internal participant-block prediction conditional on complete 18S/PK; "
                "not external, prospective, or causal validation"
            ),
            "bootstrap_unit": contract["score"]["bootstrap_unit"],
            "bootstrap_replicates": contract["score"]["bootstrap_replicates"],
            "bootstrap_seed": contract["score"]["bootstrap_seed"],
        },
        "models": models,
        "comparison": comparison,
        "diagnostics": {
            "mcmc": convergence,
            "both_mcmc_strictly_valid": mcmc_valid,
            "both_participant_blocks_psis_valid": psis_valid,
            "fit_manifests": manifests,
        },
        "public_release_only": True,
        "protected_data_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "b0_primary_nlpd": models["b0"]["primary"]["participant_sex_equal_pointwise_mean_nlpd"],
        "b2_primary_nlpd": models["b2"]["primary"]["participant_sex_equal_pointwise_mean_nlpd"],
        "comparison": comparison,
        "convergence": convergence,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
