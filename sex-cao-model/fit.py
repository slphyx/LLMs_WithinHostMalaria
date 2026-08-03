#!/usr/bin/env python3
"""Reproduce the qualified matched Cao B0/B2 posterior fits.

The sampler settings and the four chain initializations for each arm are read
from the bundled qualified-run contract.  Scientific settings are deliberately
not exposed as command-line overrides: changing them creates a new analysis,
not a reproduction of the reported result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path


HERE = Path(__file__).resolve().parent
MODEL = HERE / "model/cao_sex_allocation.stan"
QUALIFIED = HERE / "results/qualified_run"
RUN_CONTRACT = QUALIFIED / "run_contract.json"
LOCAL_CMDSTAN_ROOT = HERE / ".cmdstan"
ARMS = ("b0", "b2")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def configure_cmdstan(path: Path | None) -> Path:
    if path is None and LOCAL_CMDSTAN_ROOT.is_dir():
        installations = sorted(
            candidate
            for candidate in LOCAL_CMDSTAN_ROOT.glob("cmdstan-*")
            if (candidate / "bin/stanc").is_file()
        )
        if installations:
            path = installations[-1]
    if path is not None:
        set_cmdstan_path(str(path.resolve()))
    try:
        installation = Path(cmdstan_path())
    except ValueError as error:
        raise SystemExit(
            "CmdStan is not installed. Install 2.39.0 as described in README.md "
            "or pass --cmdstan PATH."
        ) from error
    if not (installation / "bin/stanc").is_file():
        raise SystemExit(f"Invalid CmdStan installation: {installation}")
    return installation


def selected_arms(value: str) -> tuple[str, ...]:
    return ARMS if value == "both" else (value,)


def validate_qualified_inputs(contract: dict[str, Any], arm: str) -> tuple[Path, list[dict[str, Any]]]:
    arm_contract = contract["arms"][arm]
    data_path = QUALIFIED / arm_contract["data"]
    init_path = QUALIFIED / arm_contract["initializations"]
    for path in (data_path, init_path):
        if not path.is_file():
            raise RuntimeError(f"Missing qualified input: {path}")

    data = read_json(data_path)
    if data.get("allocation_active") != arm_contract["allocation_active"]:
        raise RuntimeError(f"Wrong allocation flag in {data_path}")
    if data.get("nSexObs") != contract["score"]["sex_rows"]:
        raise RuntimeError("Qualified data do not contain all 390 sex-marker rows")
    if any(value != 1 for value in data.get("sex_include", [])):
        raise RuntimeError("The full posterior fit must include every sex-marker row")

    initialization = read_json(init_path)
    chain_inits = initialization.get("chain_initializations")
    if not isinstance(chain_inits, list) or len(chain_inits) != contract["chains"]:
        raise RuntimeError("Qualified initialization artefact must contain four chains")
    return data_path, chain_inits


def run_arm(
    model: CmdStanModel,
    contract: dict[str, Any],
    output_root: Path,
    arm: str,
    *,
    no_progress: bool,
) -> None:
    arm_contract = contract["arms"][arm]
    data_path, chain_inits = validate_qualified_inputs(contract, arm)
    output = output_root.resolve() / arm
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing arm directory: {output}")
    output.mkdir(parents=True)

    fit = model.sample(
        data=str(data_path),
        inits=chain_inits,
        seed=int(arm_contract["seed"]),
        chains=int(contract["chains"]),
        parallel_chains=int(contract["parallel_chains"]),
        threads_per_chain=int(contract["threads_per_chain"]),
        iter_warmup=int(arm_contract["warmup"]),
        iter_sampling=int(arm_contract["sampling"]),
        adapt_delta=float(arm_contract["adapt_delta"]),
        max_treedepth=int(contract["max_treedepth"]),
        metric=str(contract["metric"]),
        output_dir=str(output),
        sig_figs=int(contract["sig_figs"]),
        show_progress=not no_progress,
        refresh=50,
    )

    csv_paths = [Path(path).resolve() for path in fit.runset.csv_files]
    if len(csv_paths) != contract["chains"] or any(not path.is_file() for path in csv_paths):
        raise RuntimeError("CmdStan did not produce all four posterior CSVs")
    manifest = {
        "schema": "sex-cao-fit-manifest-v1",
        "arm": arm,
        "allocation_active": arm_contract["allocation_active"],
        "configuration": {
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
        },
        "csv_files": [
            {"path": str(path), "size": path.stat().st_size}
            for path in csv_paths
        ],
        "public_release_only": True,
        "protected_data_used": False,
    }
    write_json(output / "fit_manifest.json", manifest)
    print(f"{arm}: completed qualified refit in {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("both", *ARMS), default="both")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/qualified_refit")
    parser.add_argument("--cmdstan", type=Path)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = read_json(RUN_CONTRACT)
    installation = configure_cmdstan(args.cmdstan)
    required_version = str(contract["cmdstan_version"])
    if installation.name != f"cmdstan-{required_version}":
        raise SystemExit(
            f"Qualified reproduction requires CmdStan {required_version}; found {installation}"
        )
    model = CmdStanModel(
        stan_file=str(MODEL),
        force_compile=args.force_compile,
        stanc_options={"O1": True},
        cpp_options={"STAN_THREADS": True, "STAN_CPP_OPTIMS": True},
    )
    for arm in selected_arms(args.arm):
        run_arm(model, contract, args.output, arm, no_progress=args.no_progress)


if __name__ == "__main__":
    main()
