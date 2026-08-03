#!/usr/bin/env python3
"""Build figures for the persistent effective-male Cao extension.

The default invocation is directory-local::

    .venv/bin/python build_figures.py

"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "figure_data/cao_extension_results.json"
DEFAULT_TRAJECTORIES = (
    HERE / "figure_data/cao_extension_sex_trajectory_summary.json.gz"
)
DEFAULT_OUTPUT = HERE / "figures"
PARTICIPANTS = [
    "101", "102", "103", "104", "105", "106", "201", "202", "203",
    "204", "301", "302", "303", "304", "305", "306", "307",
]

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#008B67"
PURPLE = "#8064B3"
DARK = "#263238"
MID = "#66757D"
LIGHT = "#E3E8EA"
PALE_BLUE = "#EEF6FA"
PALE_ORANGE = "#FCF2EC"
PALE_GREEN = "#EEF8F5"
PAPER = "#F8FAFA"
CHAIN_COLOURS = [BLUE, ORANGE, GREEN, PURPLE]
CHAIN_STYLES = ["solid", (0, (4, 1.5)), (0, (1, 1.3)), (0, (4, 1.3, 1, 1.3))]
PDF_METADATA = {
    "Creator": "sex-cao-model/build_figures.py",
    "Producer": f"Matplotlib {mpl.__version__}",
    "CreationDate": None,
    "ModDate": None,
}


def read_inputs(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "cao-extension-report-figure-input-v2":
        raise ValueError("Unexpected Cao figure-input schema.")
    if data["scope"]["public_release_only"] is not True:
        raise ValueError("Figure inputs are not certified public-release-only.")
    if data["scope"]["protected_data_used"] is not False:
        raise ValueError("Protected data must not enter these report figures.")
    if data["scope"]["participants"] != 17:
        raise ValueError("Expected the 17-participant public Cao cohort.")

    result = data["qualified_comparison"]
    if result.get("status") != "valid_primary_mcmc_psis_comparison":
        raise ValueError("The packaged comparison is not qualified.")
    if result.get("performance_claim_permitted") is not True:
        raise ValueError("The packaged comparison does not permit its scoped claim.")
    if data["scope"].get("held_out_sex_assay_rows") != 390:
        raise ValueError("Expected all 390 sex-marker rows to be held out by participant.")
    if data["scope"].get("sex_marker_calibration_rows") != 0:
        raise ValueError("Sex-marker observations must not enter participant calibration.")
    if any(result["high_pareto_k_participants"][arm] for arm in ("b0", "b2")):
        raise ValueError("A participant PSIS block requires an exact refit.")
    for arm in ("b0", "b2"):
        mcmc = result["mcmc"][arm]
        if (
            mcmc.get("passed") is not True
            or mcmc.get("divergences") != 0
            or mcmc.get("max_treedepth_hits") != 0
        ):
            raise ValueError(f"{arm} does not pass the packaged strict MCMC gate.")

    draws_path = path.parent / data["posterior_draws"]["path"]
    draws = pd.read_csv(draws_path)
    expected_shape = (
        data["posterior_draws"]["rows"],
        len(data["posterior_draws"]["columns"]),
    )
    if draws.shape != expected_shape:
        raise ValueError(f"Unexpected posterior shape {draws.shape}; expected {expected_shape}.")
    if list(draws.columns) != data["posterior_draws"]["columns"]:
        raise ValueError("Compact posterior columns differ from the packaged metadata.")
    draws_per_chain = int(data["posterior_draws"]["draws_per_chain"])
    if draws.groupby("chain").size().tolist() != [draws_per_chain] * 4:
        raise ValueError("Compact posterior chain sizes differ from the metadata.")
    if draws_per_chain != 2000:
        raise ValueError(
            "Expected the qualified hierarchical-extension posterior with "
            "2,000 draws per chain."
        )
    return data, draws


def read_trajectory_inputs(path: Path) -> dict[str, Any]:
    data = json.loads(gzip.decompress(path.read_bytes()))
    if data.get("schema") != "cao-extension-sex-trajectory-summary-v2":
        raise ValueError("Unexpected sex-trajectory input schema.")
    if data.get("status") != "valid_participant_block_psis_trajectory_summary":
        raise ValueError("Sex trajectories are not from the qualified endpoint.")
    if data.get("performance_claim_permitted") is not True:
        raise ValueError("Qualified sex trajectories do not permit their scoped claim.")
    scope = data.get("scope", {})
    if (
        scope.get("participants") != 17
        or scope.get("sex_marker_calibration_rows") != 0
        or scope.get("held_out_sex_assay_rows") != 390
        or scope.get("public_release_only") is not True
        or scope.get("protected_data_used") is not False
    ):
        raise ValueError("Unexpected sex-trajectory cohort/split scope.")
    if set(data["observations"]) != set(PARTICIPANTS):
        raise ValueError("Sex-trajectory artefact does not contain all participants.")
    for arm in ("b0", "b2"):
        participants = data["models"][arm]["participants"]
        if set(participants) != set(PARTICIPANTS):
            raise ValueError(f"{arm} trajectories do not contain all participants.")
        if any(record["psis"]["pareto_warning"] for record in participants.values()):
            raise ValueError(f"{arm} includes an invalid participant PSIS block.")
    if any(
        row.get("phase") != "held_out"
        for participant in data["observations"].values()
        for row in participant["rows"]
    ):
        raise ValueError("Every displayed sex-marker observation must be held out.")
    return data


def configure_style() -> None:
    sns.set_theme(context="paper", style="white", font_scale=1.0)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.7,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    *,
    pad_inches: float = 0.09,
    tight_bbox: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    bounds = (
        {"bbox_inches": "tight", "pad_inches": pad_inches}
        if tight_bbox else {}
    )
    figure.savefig(pdf, metadata=PDF_METADATA, **bounds)
    figure.savefig(
        png, dpi=300, facecolor="white", **bounds
    )
    plt.close(figure)
    return [pdf, png]


def diagram_box(
    axis: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str = LIGHT,
    face: str = "white",
    fontsize: float = 7.0,
    weight: str = "normal",
    linewidth: float = 0.9,
) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
        transform=axis.transAxes, clip_on=False,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, fontweight=weight,
        transform=axis.transAxes, linespacing=1.2,
    )


def diagram_arrow(
    axis: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = MID,
    linewidth: float = 1.05,
    connectionstyle: str = "arc3",
    linestyle: str | tuple[Any, ...] = "solid",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8.5,
            linewidth=linewidth, color=colour,
            connectionstyle=connectionstyle, transform=axis.transAxes,
            linestyle=linestyle, shrinkA=1.5, shrinkB=1.5,
        )
    )


def panel_frame(
    axis: mpl.axes.Axes,
    letter: str,
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            linewidth=0.8, edgecolor="#C8D1D5", facecolor="white",
            transform=axis.transAxes,
        )
    )
    if letter:
        axis.text(
            x + 0.018, y + height - 0.045, letter,
            ha="center", va="center", fontsize=8.4, fontweight="bold",
            color="white", transform=axis.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor=DARK, edgecolor="none"
            ),
        )
    axis.text(
        x + (0.048 if letter else 0.025), y + height - 0.045, title,
        ha="left", va="center", fontsize=9.1, fontweight="bold",
        transform=axis.transAxes,
    )


def plot_model_schematic(output_dir: Path) -> list[Path]:
    """Show the sex-allocation extension with compact Cao-core context."""
    figure, axis = plt.subplots(figsize=(6.3, 2.5))
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.99)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    panel_frame(
        axis, "", "Cao-derived hierarchical-allocation extension",
        0.01, 0.04, 0.98, 0.92,
    )
    diagram_box(
        axis, 0.025, 0.385, 0.15, 0.15,
        "Cao mechanistic\ncore (unchanged)",
        edge=GREEN, face=PALE_GREEN, fontsize=6.4, weight="bold",
    )
    diagram_box(
        axis, 0.27, 0.265, 0.18, 0.39,
        "Hierarchical allocation\n\n$\\mathrm{logit}(\\rho_i)$\n$=\\mathrm{logit}(\\rho)$\n$+\\tau_\\rho z_i$",
        edge=PURPLE, face="#F4F0F8", fontsize=6.8, weight="bold",
    )
    diagram_arrow(axis, (0.182, 0.46), (0.263, 0.46), colour=PURPLE)
    axis.text(
        0.218, 0.505, "$S_i(t)$", fontsize=6.8,
        color=PURPLE, ha="center", transform=axis.transAxes,
    )

    diagram_box(
        axis, 0.49, 0.53, 0.145, 0.18,
        "Female I–IV\n$(1-\\rho_i)S_i(t)$", edge=ORANGE,
        face=PALE_ORANGE, fontsize=6.9, weight="bold",
    )
    diagram_box(
        axis, 0.68, 0.53, 0.115, 0.18,
        "Female\nstage V", edge=ORANGE, face=PALE_ORANGE, fontsize=7.0,
    )
    diagram_box(
        axis, 0.84, 0.53, 0.13, 0.18,
        "Pfs25\nassay", edge=ORANGE, face=PALE_ORANGE,
        fontsize=7.0, weight="bold",
    )
    diagram_box(
        axis, 0.49, 0.18, 0.145, 0.18,
        "Male I–IV\n$\\rho_iS_i(t)$", edge=BLUE, face=PALE_BLUE,
        fontsize=6.9, weight="bold",
    )
    diagram_box(
        axis, 0.68, 0.18, 0.115, 0.18,
        "Male\nstage V", edge=BLUE, face=PALE_BLUE, fontsize=7.0,
    )
    diagram_box(
        axis, 0.84, 0.18, 0.13, 0.18,
        "PfMGET\nassay", edge=BLUE, face=PALE_BLUE,
        fontsize=7.0, weight="bold",
    )
    diagram_arrow(
        axis, (0.457, 0.535), (0.483, 0.62), colour=ORANGE,
        connectionstyle="arc3,rad=-0.10",
    )
    diagram_arrow(
        axis, (0.457, 0.385), (0.483, 0.27), colour=BLUE,
        connectionstyle="arc3,rad=0.10",
    )
    diagram_arrow(axis, (0.642, 0.62), (0.673, 0.62), colour=ORANGE)
    diagram_arrow(axis, (0.802, 0.62), (0.833, 0.62), colour=ORANGE)
    diagram_arrow(axis, (0.642, 0.27), (0.673, 0.27), colour=BLUE)
    diagram_arrow(axis, (0.802, 0.27), (0.833, 0.27), colour=BLUE)

    return save_figure(figure, output_dir, "cao_extension_model")


def plot_predictive_comparison(data: dict[str, Any], output_dir: Path) -> list[Path]:
    """Plot qualified participant-block sex-marker predictive gains."""
    figure, axis = plt.subplots(figsize=(3.7, 4.25))
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.84)

    primary = data["qualified_comparison"]
    gains = {key: float(value) for key, value in primary["participant_gains"].items()}
    ordered = sorted(gains, key=gains.get)
    values = np.asarray([gains[key] for key in ordered])
    y = np.arange(len(ordered))
    colours = np.where(values >= 0, BLUE, ORANGE)
    x_low = min(-0.01, float(values.min()) - 0.004)
    x_high = max(0.06, float(values.max()) + 0.004)
    axis.axvspan(0, x_high, color=PALE_BLUE, zorder=0)
    axis.axvline(0, color=DARK, linewidth=0.85, zorder=1)
    axis.hlines(y, 0, values, color=colours, linewidth=1.1, zorder=2)
    axis.scatter(
        values, y, s=28, c=colours, edgecolor=DARK, linewidth=0.45, zorder=3
    )
    axis.axvline(
        primary["mean_nlpd_gain"], color=BLUE, linewidth=0.9,
        linestyle=(0, (3, 2)), zorder=1,
    )
    axis.set_yticks(y, ordered)
    axis.set_ylim(-0.7, len(y) - 0.3)
    axis.set_xlim(x_low, x_high)
    axis.set_xticks([0, 0.02, 0.04, 0.06])
    axis.set_xlabel(
        "Participant NLPD gain\n(global-allocation comparator − hierarchical extension)",
        fontsize=7.4, labelpad=5,
    )
    axis.set_ylabel("Participant", fontsize=8.0)
    figure.text(
        0.18, 0.965, "Held-out participant sex-marker prediction",
        ha="left", va="top", fontsize=9.0, fontweight="bold",
    )
    figure.text(
        0.18, 0.918,
        f"Hierarchical extension wins in {primary['strict_participant_wins']}/17; "
        f"mean gain "
        f"{primary['mean_nlpd_gain']:.4f} "
        f"(95% CI {primary['participant_bootstrap_95'][0]:.4f}–"
        f"{primary['participant_bootstrap_95'][1]:.4f})",
        ha="left", va="top", fontsize=6.8, color=MID,
    )
    figure.text(
        0.18, 0.890, "All participant PSIS blocks valid",
        ha="left", va="top", fontsize=6.8, color=MID,
    )
    axis.text(
        x_high - 0.002, 0.03, "favours hierarchical extension",
        color=BLUE, fontsize=6.4,
        ha="right", va="bottom", transform=axis.get_xaxis_transform(),
    )
    axis.tick_params(axis="x", labelsize=7.2)
    axis.tick_params(axis="y", labelsize=7.2, length=0)
    axis.grid(axis="x", color=LIGHT, linewidth=0.65)
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis, left=True)
    return save_figure(
        figure, output_dir, "cao_extension_predictive_comparison", pad_inches=0
    )


def plot_diagnostics(
    _data: dict[str, Any], draws: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Plot chain behaviour and the participant allocation posterior."""
    figure, panels = plt.subplot_mosaic(
        [["tau_trace", "rho_trace"], ["forest", "forest"]],
        figsize=(6.3, 4.25),
        gridspec_kw={"height_ratios": [0.95, 1.15]},
    )
    figure.subplots_adjust(
        left=0.11, right=0.985, bottom=0.15, top=0.86,
        wspace=0.42, hspace=0.65,
    )

    for axis_key, parameter, title, ylabel in (
        ("tau_trace", "tau_rho", "A  Between-participant scale", "$\\tau_\\rho$"),
        ("rho_trace", "rho_male", "B  Population male allocation", "$\\rho$"),
    ):
        axis = panels[axis_key]
        for chain, colour, linestyle in zip(
            range(1, 5), CHAIN_COLOURS, CHAIN_STYLES, strict=True
        ):
            subset = draws.loc[draws["chain"] == chain]
            axis.plot(
                subset["draw"], subset[parameter], color=colour,
                linewidth=0.55, linestyle=linestyle, alpha=0.82,
            )
        values = draws[parameter].to_numpy()
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        axis.text(
            0.99, 0.96, f"median {median:.3f}  (95% {q025:.3f}–{q975:.3f})",
            transform=axis.transAxes, ha="right", va="top", fontsize=7.0,
            color=DARK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
        )
        final_draw = int(draws["draw"].max())
        axis.set_xlim(1, final_draw)
        axis.set_xticks(np.linspace(1, final_draw, 4, dtype=int))
        axis.set_xlabel("Sampling iteration", fontsize=7.4)
        axis.set_ylabel(ylabel, fontsize=8.2, rotation=0, labelpad=10)
        axis.set_title(title, loc="left", fontsize=8.7, fontweight="bold", pad=4)
        axis.tick_params(labelsize=7.0)
        axis.grid(color=LIGHT, linewidth=0.55)
        sns.despine(ax=axis)

    axis = panels["forest"]
    posterior = []
    for index, participant in enumerate(PARTICIPANTS, start=1):
        values = draws[f"rho_male_subject.{index}"].to_numpy()
        posterior.append(
            (
                participant,
                float(np.mean(values)),
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            )
        )
    posterior.sort(key=lambda item: item[1])
    x = np.arange(len(posterior))
    means = np.asarray([item[1] for item in posterior])
    lower = np.asarray([item[2] for item in posterior])
    upper = np.asarray([item[3] for item in posterior])
    global_values = draws["rho_male"].to_numpy()
    global_lower, global_median, global_upper = np.quantile(
        global_values, [0.025, 0.5, 0.975]
    )
    axis.axhspan(global_lower, global_upper, color="#DCE8EE", alpha=0.75, zorder=0)
    axis.axhline(global_median, color=MID, linewidth=0.85, linestyle=(0, (3, 2)))
    axis.errorbar(
        x, means, yerr=np.vstack([means - lower, upper - means]),
        fmt="o", markersize=4.5, color=BLUE, markerfacecolor=BLUE,
        markeredgecolor=DARK, markeredgewidth=0.4, ecolor="#60747E",
        elinewidth=0.9, capsize=2.2, zorder=3,
    )
    axis.set_xticks(x, [item[0] for item in posterior], rotation=45, ha="right")
    axis.set_xlim(-0.6, len(x) - 0.4)
    axis.set_ylim(0.06, 0.56)
    axis.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axis.set_ylabel("Effective male allocation $\\rho_i$", fontsize=7.8)
    axis.set_xlabel("Participant (ordered by posterior mean)", fontsize=7.5)
    axis.set_title(
        "C  Full-data posterior: marker-informed effective male allocation", loc="left",
        fontsize=8.7, fontweight="bold", pad=5,
    )
    axis.tick_params(axis="x", labelsize=7.0)
    axis.tick_params(axis="y", labelsize=7.0)
    axis.grid(axis="y", color=LIGHT, linewidth=0.6)
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    axis.text(
        0.99, 0.94, "shading: population $\\rho$ 95% interval",
        transform=axis.transAxes, fontsize=7.0, color=MID,
        ha="right", va="top",
    )

    handles = [
        Line2D(
            [0], [0], color=colour, linestyle=linestyle,
            linewidth=1.3, label=f"Chain {chain}"
        )
        for chain, colour, linestyle in zip(
            range(1, 5), CHAIN_COLOURS, CHAIN_STYLES, strict=True
        )
    ]
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
        frameon=False, ncol=4, fontsize=7.0, handlelength=1.5,
        columnspacing=1.1,
    )
    return save_figure(figure, output_dir, "cao_extension_diagnostics")


def plot_sex_trajectories(
    data: dict[str, Any], output_dir: Path
) -> list[Path]:
    """Cao Figure-3 analogue with PSIS case-deleted marker trajectories."""
    figure = plt.figure(figsize=(6.24, 8.45))
    outer = figure.add_gridspec(
        5, 4, left=0.075, right=0.988, bottom=0.06, top=0.93,
        wspace=0.23, hspace=0.34,
    )
    x_limits = (3, 31)
    y_limits = (0.018, 30000)
    nondetection_display = float(data["data"]["sex_lod_reference_scale"])
    marker_specs = {
        "female": (ORANGE, "Pfs25"),
        "male": (BLUE, "PfMGET"),
    }
    for participant_index, participant in enumerate(PARTICIPANTS):
        row, column = divmod(participant_index, 4)
        main_axis = figure.add_subplot(outer[row, column])
        observation = data["observations"][participant]

        for sex_stream, (colour, _) in marker_specs.items():
            b0_record = data["models"]["b0"]["participants"][participant]
            b2_record = data["models"]["b2"]["participants"][participant]
            b0_day = np.asarray(b0_record["grid_day"], dtype=float)
            day = np.asarray(b2_record["grid_day"], dtype=float)
            if not np.array_equal(b0_day, day):
                raise ValueError(
                    f"Comparator/extension trajectory grids differ for {participant}."
                )
            b0_median = np.asarray(
                b0_record["streams"][sex_stream]["latent_assay_density"]["median"],
                dtype=float,
            )
            stream = b2_record["streams"][sex_stream]
            latent = stream["latent_assay_density"]
            median = np.asarray(latent["median"], dtype=float)
            lower = np.asarray(latent["q025"], dtype=float)
            upper = np.asarray(latent["q975"], dtype=float)
            main_axis.fill_between(
                day, lower, upper, color=colour, alpha=0.13,
                linewidth=0, zorder=2,
            )
            main_axis.plot(
                b0_day, b0_median, color=colour, linestyle=(0, (3.2, 2.0)),
                linewidth=0.82, alpha=0.88, zorder=3,
            )
            main_axis.plot(
                day, median, color=colour, linestyle="solid",
                linewidth=1.05, alpha=0.98, zorder=4,
            )

        for record in observation["rows"]:
            colour = marker_specs[record["sex_stream"]][0]
            day = float(record["day"]) + (-0.08 if record["sex_stream"] == "female" else 0.08)
            nondetection = bool(record["nondetection"])
            value = (
                nondetection_display
                if nondetection
                else float(record["observed_assay_density"])
            )
            main_axis.scatter(
                [day], [value], s=10.0, marker="o",
                facecolor="white", edgecolor=colour,
                linewidth=0.65, zorder=5,
            )

        main_axis.axhline(
            nondetection_display, color="#AAB3B7",
            linewidth=0.45, linestyle=(0, (1.2, 2.2)), zorder=1,
        )
        main_axis.set_yscale("log")
        main_axis.set_xlim(*x_limits)
        main_axis.set_ylim(*y_limits)
        main_axis.set_title(
            f"Volunteer {participant}", loc="left", fontsize=6.4,
            fontweight="normal", pad=1.4,
        )
        main_axis.set_xticks([5, 15, 25])
        main_axis.set_xticklabels(["5", "15", "25"], fontsize=5.5)
        main_axis.tick_params(axis="x", length=1.6, width=0.4, pad=1.1)
        main_axis.set_yticks([0.1, 1, 10, 100, 1000, 10000])
        if column == 0:
            main_axis.set_yticklabels(
                ["0.1", "1", "10", "$10^2$", "$10^3$", "$10^4$"],
                fontsize=5.6,
            )
        else:
            main_axis.set_yticklabels([])
        main_axis.tick_params(axis="y", length=1.8, width=0.45, pad=1.3)
        main_axis.grid(False)
        sns.despine(ax=main_axis)

    # The unused cells carry a compact key; the report caption states the
    # evidence boundary and evaluation details.
    legend_axis = figure.add_subplot(outer[4, 1:4])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D([0], [0], color=ORANGE, linewidth=1.4, label="Female (Pfs25)"),
            Line2D([0], [0], color=BLUE, linewidth=1.4, label="Male (PfMGET)"),
            Line2D(
                [0], [0], color=DARK, linewidth=1.15, linestyle="solid",
                label="Hierarchical extension (solid)",
            ),
            Line2D(
                [0], [0], color=DARK, linewidth=1.0, linestyle=(0, (3.2, 2.0)),
                label="Global-allocation comparator (dashed)",
            ),
            Patch(
                facecolor=MID, alpha=0.17, edgecolor="none",
                label="Hierarchical latent 95% interval",
            ),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=MID, markersize=4.2, label="Held-out assay"),
            Line2D([0], [0], color="#9AA5AA", linewidth=0.7,
                   linestyle=(0, (1.2, 2.2)), label="Nondetection line"),
        ],
        loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False,
        fontsize=5.7, handlelength=2.1, borderpad=0, labelspacing=0.36,
        ncol=2, columnspacing=1.4,
    )
    legend_axis.text(
        0.02, 0.06,
        "All sex-marker rows for each volunteer are held out jointly; complete 18S/PK is retained.",
        ha="left", va="top", fontsize=5.8,
        color=DARK, transform=legend_axis.transAxes,
    )

    figure.text(
        0.5, 0.982,
        "Sex-marker trajectories for all 17 volunteers",
        ha="center", va="top", fontsize=9.4, fontweight="bold", color=DARK,
    )
    figure.text(
        0.008, 0.52, "Marker assay density (/mL)", rotation=90,
        ha="center", va="center", fontsize=6.4, color=DARK,
    )
    figure.text(
        0.5, 0.045, "Post-inoculation day",
        ha="center", va="center", fontsize=6.4, color=DARK,
    )
    return save_figure(
        figure, output_dir, "cao_extension_sex_trajectories", tight_bbox=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_path = args.data.resolve()
    trajectory_path = args.trajectories.resolve()
    output_dir = args.output_dir.resolve()
    configure_style()
    data, draws = read_inputs(data_path)
    trajectories = read_trajectory_inputs(trajectory_path)
    outputs = []
    outputs.extend(plot_model_schematic(output_dir))
    outputs.extend(plot_predictive_comparison(data, output_dir))
    outputs.extend(plot_diagnostics(data, draws, output_dir))
    outputs.extend(plot_sex_trajectories(trajectories, output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
