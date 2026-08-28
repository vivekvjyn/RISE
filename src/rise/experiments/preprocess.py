"""Build every dataset the other experiments consume.

Run once, before anything else. The datasets are written to the cache as fixed
train / validation / test splits so that no later experiment re-draws them.
"""

from __future__ import annotations

import argparse

from ..data.preprocessing import (
    build_classification_datasets,
    build_clustering_dataset,
    build_pattern_recognition_dataset,
    build_pretrain_dataset,
    build_synthesis_dataset,
)

DESCRIPTION = "Extract pitch contours and cut them into the datasets of each experiment"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.5,
        help="residual budget of the smoothing spline; higher smooths more",
    )
    parser.add_argument(
        "--interpolation-gap",
        type=float,
        default=0.02,
        help="longest run of unvoiced frames that is interpolated across",
    )


def run(args: argparse.Namespace) -> None:
    # The builders run in a fixed order because the two that sample from the beat
    # grid draw from the same seeded generator; reordering them would change which
    # segments each one gets.
    build_classification_datasets(args.smoothing_factor, args.interpolation_gap)
    build_clustering_dataset(args.smoothing_factor, args.interpolation_gap)
    build_pattern_recognition_dataset(args.smoothing_factor, args.interpolation_gap)
    build_pretrain_dataset(args.smoothing_factor, args.interpolation_gap)
    build_synthesis_dataset(args.smoothing_factor, args.interpolation_gap)
