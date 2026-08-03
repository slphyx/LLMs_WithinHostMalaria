#!/usr/bin/env python3
"""Prepare the compact, qualified Cao-style sex-assay trajectory artefact.

This is a preparation step, not part of the lightweight figure rebuild. It
reads four-chain B0 and B2 qualified-run CmdStan CSVs, deletes every sex-marker
row for one participant as a single PSIS block while retaining that
participant's complete 18S/PK series, replays the exact Stan recurrence, and
writes a deterministic gzip-compressed JSON summary.  The rendering script
subsequently needs only this compact public-release artefact.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

import score


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs/cao_extension_sex_trajectory_summary.json.gz"
DEFAULT_ARM_DIRECTORIES = {
    "b0": HERE / "outputs/refit/b0/qualified",
    "b2": HERE / "outputs/refit/b2/qualified",
}
SEXES = (("female", 0, "Pfs25"), ("male", 1, "PfMGET"))
GRID_STEP_HOURS = 6
QUANTILES = (0.025, 0.5, 0.975)


def integrated_flow(rate: np.ndarray) -> np.ndarray:
    result = -np.expm1(-rate) / rate
    small = rate < 1e-8
    result[small] = 1 - 0.5 * rate[small] + rate[small] ** 2 / 6
    return result


def weighted_quantile(values: np.ndarray, weights: np.ndarray) -> list[float]:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return [float(np.interp(q, cumulative, ordered)) for q in QUANTILES]


def read_arm(directory: Path, n_sex_obs: int) -> tuple[pd.DataFrame, np.ndarray]:
    paths = score.csv_paths(directory)
    if len(paths) != 4:
        raise RuntimeError(f"Expected four CmdStan CSVs in {directory}")
    parameter_columns = [
        "log_scale",
        "log_sigma_sex.1", "log_sigma_sex.2",
        "alpha_detect.1", "alpha_detect.2",
        "detection_slope.1", "detection_slope.2",
        *[
            f"rho_male_subject.{subject}"
            for subject in range(1, len(score.PARTICIPANTS) + 1)
        ],
        *[
            f"thetaInd.{parameter}.{subject}"
            for subject in range(1, len(score.PARTICIPANTS) + 1)
            for parameter in range(1, 13)
        ],
    ]
    log_likelihood_columns = [
        f"log_lik_sex.{row}" for row in range(1, n_sex_obs + 1)
    ]
    frames = []
    for chain, path in enumerate(paths, start=1):
        frame = pd.read_csv(
            path, comment="#", usecols=[*parameter_columns, *log_likelihood_columns]
        )
        frame = pd.concat(
            [
                pd.DataFrame(
                    {"chain": chain, "draw": np.arange(1, len(frame) + 1)}
                ),
                frame,
            ],
            axis=1,
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    chain_sizes = [len(frame) for frame in frames]
    if len(set(chain_sizes)) != 1:
        raise RuntimeError(f"Unequal chain sizes in {directory}: {chain_sizes}")
    likelihood = np.stack(
        [
            frame[log_likelihood_columns].to_numpy(dtype=float)
            for frame in frames
        ],
        axis=0,
    )
    return combined, likelihood


def replay_subject(
    draws: pd.DataFrame,
    data: dict[str, Any],
    subject: int,
    output_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Python transcription of cao_sex_trajectory for one subject."""
    sample_count = len(draws)
    n_stages = int(data["nStages"])
    a_s = int(data["a_s"])
    theta = np.column_stack(
        [draws[f"thetaInd.{parameter}.{subject}"] for parameter in range(1, 13)]
    )
    stages = np.arange(1, n_stages + 1, dtype=float)[None, :]
    mu = theta[:, 1, None]
    sigma = theta[:, 2, None]
    normal_constant = -np.log(sigma) - 0.5 * math.log(2 * math.pi)
    first = (
        np.log(theta[:, 3, None]) + normal_constant
        - 0.5 * ((stages - 0.5 - mu) / sigma) ** 2
    )
    second = (
        normal_constant
        - 0.5 * ((stages - n_stages - 0.5 - mu) / sigma) ** 2
    )
    log_age = np.logaddexp(first, second)
    log_age -= np.max(log_age, axis=1, keepdims=True)
    age = np.exp(log_age)
    parasites = theta[:, 0, None] * age / np.sum(age, axis=1, keepdims=True)
    committed = np.zeros((sample_count, n_stages))
    immature = np.zeros((sample_count, 4))
    mature_total = np.zeros(sample_count)

    concentration = np.asarray(data["PIPconc"], dtype=float)[:, subject - 1]
    gamma = theta[:, 6]
    ec50_gamma = theta[:, 5] ** gamma
    parasite_survival = np.exp(-theta[:, 8])
    m = theta[:, 9]
    immature_rate = m + theta[:, 10]
    immature_survival = np.exp(-immature_rate)
    immature_flow = integrated_flow(immature_rate)
    mature_survival = np.exp(-theta[:, 11])
    mature_flow = integrated_flow(theta[:, 11])
    rho = draws[f"rho_male_subject.{subject}"].to_numpy(dtype=float)

    maximum = int(np.max(output_indices))
    positions = {int(index): slot for slot, index in enumerate(output_indices)}
    total_mature = np.zeros((sample_count, len(output_indices)))
    for stan_index in range(2, maximum + 1):
        prior_concentration = concentration[stan_index - 2]
        current_concentration = concentration[stan_index - 1]
        prior_power = prior_concentration ** gamma
        current_power = current_concentration ** gamma
        prior_killing = theta[:, 4] * prior_power / (ec50_gamma + prior_power)
        current_killing = theta[:, 4] * current_power / (ec50_gamma + current_power)
        treated_survival = np.exp(-0.5 * (prior_killing + current_killing)) * parasite_survival

        parasites_new = np.zeros_like(parasites)
        committed_new = np.zeros_like(committed)
        parasites_new[:, 1:] = parasites[:, :-1] * treated_survival[:, None]
        parasites_new[:, 0] = theta[:, 3] * parasites[:, -1] * treated_survival
        parasites_new[:, a_s] = (
            (1 - theta[:, 7]) * parasites[:, a_s - 1] * treated_survival
        )
        committed_new[:, (a_s + 1):] = (
            committed[:, a_s:-1] * parasite_survival[:, None]
        )
        committed_new[:, a_s] = (
            theta[:, 7] * parasites[:, a_s - 1] * treated_survival
        )
        committed_new[:, 0] = theta[:, 3] * committed[:, -1] * parasite_survival
        committed_new[:, 1:(a_s - 1)] = (
            committed[:, 0:(a_s - 2)] * parasite_survival[:, None]
        )
        source = committed[:, a_s - 2] * parasite_survival

        immature_new = np.empty_like(immature)
        immature_new[:, 0] = (
            immature[:, 0] * immature_survival + source * immature_flow
        )
        for stage in range(1, 4):
            immature_new[:, stage] = (
                immature[:, stage] * immature_survival
                + m * immature[:, stage - 1] * immature_flow
            )
        mature_total = (
            mature_total * mature_survival + m * immature[:, 3] * mature_flow
        )
        parasites = parasites_new
        committed = committed_new
        immature = immature_new
        slot = positions.get(stan_index)
        if slot is not None:
            total_mature[:, slot] = mature_total

    return total_mature * (1 - rho[:, None]), total_mature * rho[:, None]


def participant_blocks(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Define the qualified deletion endpoint without relying on legacy splits."""
    subject = np.asarray(data["sex_subject"], dtype=int)
    male = np.asarray(data["sex_is_male"], dtype=int)
    blocks: dict[str, dict[str, Any]] = {}
    for subject_index, participant in enumerate(score.PARTICIPANTS, start=1):
        scored = np.flatnonzero(subject == subject_index)
        female = scored[male[scored] == 0]
        male_rows = scored[male[scored] == 1]
        if not len(scored) or not len(female) or not len(male_rows):
            raise RuntimeError(f"Incomplete sex-marker block for {participant}")
        blocks[participant] = {
            "calibration_rows": [],
            "scored_rows": scored.tolist(),
            "female_scored_rows": female.tolist(),
            "male_scored_rows": male_rows.tolist(),
        }
    if sum(len(value["scored_rows"]) for value in blocks.values()) != int(data["nSexObs"]):
        raise RuntimeError("Participant deletion blocks do not partition all sex-marker rows")
    return blocks


def make_observations(
    data: dict[str, Any], participant: str
) -> list[dict[str, Any]]:
    participant_index = score.PARTICIPANTS.index(participant) + 1
    subject = np.asarray(data["sex_subject"], dtype=int)
    indices = np.flatnonzero(subject == participant_index)
    records = []
    for index in indices:
        is_male = int(data["sex_is_male"][index])
        records.append(
            {
                "source_row": int(index + 1),
                "day": float((int(data["sex_iObs"][index]) - 1) / 24),
                "marker": "PfMGET" if is_male else "Pfs25",
                "sex_stream": "male" if is_male else "female",
                "observed_assay_density": float(data["sex_count"][index]),
                "nondetection": bool(data["sex_is_nondetection"][index]),
                "phase": "held_out",
            }
        )
    return records


def summarize_arm(
    arm: str,
    draws: pd.DataFrame,
    likelihood_by_chain: np.ndarray,
    data: dict[str, Any],
    blocks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    flattened_likelihood = likelihood_by_chain.reshape(-1, likelihood_by_chain.shape[-1])
    participants: dict[str, Any] = {}
    max_replay_error = 0.0
    for subject, participant in enumerate(score.PARTICIPANTS, start=1):
        contract = blocks[participant]
        scored_rows = np.asarray(contract["scored_rows"], dtype=int)
        psis = score.psis_weights(
            np.sum(likelihood_by_chain[:, :, scored_rows], axis=2)
        )
        log_weights = np.asarray(psis.pop("log_weights"), dtype=float)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        visits = np.asarray(
            sorted(
                set(
                    int(data["sex_iObs"][index])
                    for index in contract["scored_rows"]
                )
            ),
            dtype=int,
        )
        # Cao-style panels show emergence before the first marker observation,
        # rather than beginning only once a marker happens to have been sampled.
        # Day 3 is Stan index 73 under the package's (index-1)/24 convention.
        grid_start = 73
        grid_end = min(int(data["Niter"]), int(visits.max()) + 24)
        regular = np.arange(grid_start, grid_end + 1, GRID_STEP_HOURS, dtype=int)
        output_indices = np.asarray(sorted(set(regular) | set(visits)), dtype=int)
        female_stock, male_stock = replay_subject(draws, data, subject, output_indices)
        stocks = (female_stock, male_stock)
        streams: dict[str, Any] = {}
        for sex_name, is_male, marker in SEXES:
            latent = float(data["latent_floor"]) + np.exp(
                draws["log_scale"].to_numpy(dtype=float)
            )[:, None] * stocks[is_male]
            alpha = draws[f"alpha_detect.{is_male + 1}"].to_numpy(dtype=float)
            slope = draws[f"detection_slope.{is_male + 1}"].to_numpy(dtype=float)
            probability = expit(
                alpha[:, None]
                + slope[:, None] * np.log1p(latent / float(data["sex_lod"]))
            )
            latent_quantiles = np.asarray(
                [weighted_quantile(latent[:, time], weights) for time in range(latent.shape[1])]
            ).T
            expected_detection = np.sum(weights[:, None] * probability, axis=0)
            streams[sex_name] = {
                "marker": marker,
                "latent_assay_density": {
                    "q025": latent_quantiles[0].tolist(),
                    "median": latent_quantiles[1].tolist(),
                    "q975": latent_quantiles[2].tolist(),
                },
                "probability_detected_mean": expected_detection.tolist(),
            }

            # Replay validation against the original generated log likelihood.
            participant_rows = [
                row for row in range(int(data["nSexObs"]))
                if int(data["sex_subject"][row]) == subject
                and int(data["sex_is_male"][row]) == is_male
            ]
            slot_by_index = {int(value): slot for slot, value in enumerate(output_indices)}
            for row in participant_rows:
                slot = slot_by_index[int(data["sex_iObs"][row])]
                p = probability[:, slot]
                eta = (
                    alpha
                    + slope * np.log1p(latent[:, slot] / float(data["sex_lod"]))
                )
                if int(data["sex_is_nondetection"][row]) == 1:
                    replayed = -np.logaddexp(0, eta)
                else:
                    ln_y = float(data["lnSexCount"][row])
                    sigma = np.exp(
                        draws[f"log_sigma_sex.{is_male + 1}"].to_numpy(dtype=float)
                    )
                    replayed = (
                        -np.logaddexp(0, -eta)
                        - np.log(sigma)
                        - 0.5 * math.log(2 * math.pi)
                        - 0.5 * ((ln_y - np.log(latent[:, slot])) / sigma) ** 2
                        - ln_y
                    )
                max_replay_error = max(
                    max_replay_error,
                    float(np.max(np.abs(replayed - flattened_likelihood[:, row]))),
                )

        participants[participant] = {
            "grid_day": ((output_indices - 1) / 24).tolist(),
            "psis": {
                "pareto_k": psis["pareto_k"],
                "good_k_threshold": psis["good_k_threshold"],
                "pareto_warning": psis["pareto_warning"],
                "relative_mcmc_efficiency": psis["relative_mcmc_efficiency"],
                "posterior_draws": int(len(draws)),
            },
            "streams": streams,
        }
        print(f"{arm} participant {participant}: k={psis['pareto_k']:.5f}", flush=True)
    return {
        "participants": participants,
        "maximum_replayed_log_likelihood_absolute_error": max_replay_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path, default=DEFAULT_ARM_DIRECTORIES["b0"])
    parser.add_argument("--b2", type=Path, default=DEFAULT_ARM_DIRECTORIES["b2"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite {args.output}; pass --overwrite")
    data_path = HERE / "data/stan_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    blocks = participant_blocks(data)
    observations = {
        participant: {
            "rows": make_observations(data, participant),
        }
        for participant in score.PARTICIPANTS
    }
    models: dict[str, Any] = {}
    for arm, directory in {"b0": args.b0, "b2": args.b2}.items():
        draws, likelihood = read_arm(directory, int(data["nSexObs"]))
        summary_path = directory / "mcmc_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing strict MCMC summary: {summary_path}")
        mcmc = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            mcmc.get("arm") != arm
            or mcmc.get("divergences") != 0
            or mcmc.get("max_treedepth_transitions") != 0
            or float(mcmc.get("maximum_rhat", math.inf)) > 1.01
            or float(mcmc.get("minimum_bulk_ess", -math.inf)) < 400
            or float(mcmc.get("minimum_tail_ess", -math.inf)) < 400
            or min(mcmc.get("ebfmi_by_chain", [-math.inf])) < 0.30
        ):
            raise RuntimeError(f"{arm} does not pass the strict MCMC gate")
        if int(mcmc["draws_sampling_per_chain"]) * 4 != len(draws):
            raise RuntimeError(f"{arm} retained-draw count differs from its MCMC summary")
        models[arm] = summarize_arm(arm, draws, likelihood, data, blocks)
        models[arm]["mcmc"] = {
            "chains": int(mcmc["chains"]),
            "draws_sampling_per_chain": int(mcmc["draws_sampling_per_chain"]),
            "draws_warmup_per_chain": int(mcmc["draws_warmup_per_chain"]),
            "divergences": int(mcmc["divergences"]),
            "max_treedepth_transitions": int(mcmc["max_treedepth_transitions"]),
            "maximum_rhat": float(mcmc["maximum_rhat"]),
            "minimum_bulk_ess": float(mcmc["minimum_bulk_ess"]),
            "minimum_tail_ess": float(mcmc["minimum_tail_ess"]),
            "ebfmi_by_chain": list(map(float, mcmc["ebfmi_by_chain"])),
        }

    artefact = {
        "schema": "cao-extension-sex-trajectory-summary-v2",
        "status": "valid_participant_block_psis_trajectory_summary",
        "performance_claim_permitted": True,
        "scope": {
            "participants": 17,
            "public_release_only": True,
            "protected_data_used": False,
            "sex_marker_calibration_rows": 0,
            "held_out_sex_assay_rows": sum(
                len(value["scored_rows"]) for value in blocks.values()
            ),
            "conditioning": "complete participant 18S/PK series; no sex-marker observations",
            "claim_ceiling": "internal conditional participant-block prediction; not external validation",
        },
        "method": {
            "case_deletion": (
                "For each arm and participant, PSIS weights are computed from minus the "
                "joint log likelihood of every sex-marker row for that participant and "
                "applied to all retained trajectory draws. Complete 18S/PK remains conditioned upon."
            ),
            "grid_step_hours": GRID_STEP_HOURS,
            "day_transform": "(Stan hour index - 1) / 24",
            "rendered_main_summary": "PSIS-weighted median and 95% interval of latent assay-scale density lambda_s(t)",
            "hurdle_summary": "PSIS-weighted posterior mean detection probability E[p_detect(t)]",
            "nondetection_display": (
                "open circle on the dotted 1/mL display line; categorical nondetection "
                "is not treated as a measured 1/mL value"
            ),
        },
        "data": {
            "sex_lod_reference_scale": data["sex_lod"],
            "latent_floor": data["latent_floor"],
        },
        "observations": observations,
        "models": models,
    }
    if (
        artefact["scope"]["sex_marker_calibration_rows"] != 0
        or artefact["scope"]["held_out_sex_assay_rows"] != 390
    ):
        raise RuntimeError("Unexpected qualified participant-block endpoint counts")
    for arm in DEFAULT_ARM_DIRECTORIES:
        replay_error = artefact["models"][arm]["maximum_replayed_log_likelihood_absolute_error"]
        if replay_error > 1e-8:
            raise RuntimeError(f"{arm} trajectory replay error too large: {replay_error}")
        if any(
            value["psis"]["pareto_warning"]
            for value in artefact["models"][arm]["participants"].values()
        ):
            raise RuntimeError(f"{arm} contains a high Pareto-k participant block")

    payload = (json.dumps(artefact, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "replay_error": {
            arm: models[arm]["maximum_replayed_log_likelihood_absolute_error"]
            for arm in DEFAULT_ARM_DIRECTORIES
        },
    }, indent=2))


if __name__ == "__main__":
    main()
