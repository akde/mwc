#!/usr/bin/env python3
"""CLI entry point for the spatial alignment pipeline.

Subcommands
-----------
homography  Compute and save the RGB-to-thermal homography for one or all experiments.
match       Run frame-to-frame matching between warped-RGB and thermal videos.
resume      Continue a previously interrupted matching run from a partial CSV.

Examples
--------
python run_alignment.py homography --experiment 0798
python run_alignment.py homography --all
python run_alignment.py match --experiment 0798 --start-frame 2400 --stop-frame 2450
python run_alignment.py match --all
python run_alignment.py resume --csv path/to/partial.csv --stop-frame 10000
"""

import argparse
import os
import sys

import yaml

from material_classifier.alignment.experiment_metadata import (
    get_experiment_params,
    load_experiment_metadata,
)
from material_classifier.alignment.homography import compute_homography
from material_classifier.alignment.frame_matching import (
    match_video_frames,
    resume_matching,
)
from material_classifier.alignment.matcher import SuperGlueMatcher


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "material_classifier", "config", "alignment.yaml",
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_matcher(cfg):
    mc = cfg.get("matcher", {})
    return SuperGlueMatcher(
        match_threshold=mc.get("match_threshold", 0.2),
        use_gpu=mc.get("use_gpu", True),
        keypoint_threshold=mc.get("keypoint_threshold", 0.0001),
    )


# -----------------------------------------------------------------------
# homography subcommand
# -----------------------------------------------------------------------

def cmd_homography(args):
    cfg = load_config(args.config)
    base_dir = os.path.expanduser(args.downloads_dir or cfg.get("downloads_dir", "~/Downloads"))
    matcher = make_matcher(cfg)

    if args.all:
        metadata = load_experiment_metadata()
        exp_ids = sorted(metadata.keys())
    else:
        exp_ids = [args.experiment]

    for eid in exp_ids:
        print(f"\n{'='*60}")
        print(f"Experiment {eid}")
        print(f"{'='*60}")
        try:
            H, path = compute_homography(
                eid, base_dir, matcher,
                output_dir=args.output_dir,
                debug=args.debug,
            )
            print(f"  -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            if not args.all:
                sys.exit(1)


# -----------------------------------------------------------------------
# match subcommand
# -----------------------------------------------------------------------

def cmd_match(args):
    cfg = load_config(args.config)
    base_dir = os.path.expanduser(args.downloads_dir or cfg.get("downloads_dir", "~/Downloads"))
    fm_cfg = cfg.get("frame_matching", {})
    matcher = make_matcher(cfg)

    if args.all:
        metadata = load_experiment_metadata()
        exp_ids = sorted(metadata.keys())
    else:
        exp_ids = [args.experiment]

    for eid in exp_ids:
        print(f"\n{'='*60}")
        print(f"Experiment {eid}")
        print(f"{'='*60}")

        params = get_experiment_params(eid)
        exp_dir = os.path.join(base_dir, eid)

        # Determine video paths
        query_video = os.path.join(exp_dir, f"IMG_{eid}_synched_warped.mp4")
        target_video = os.path.join(exp_dir, f"FLIR{eid}_synched_cropped.mp4")

        if not os.path.exists(query_video):
            print(f"  WARNING: Warped RGB video not found: {query_video}")
            if not args.all:
                sys.exit(1)
            continue
        if not os.path.exists(target_video):
            print(f"  WARNING: Thermal video not found: {target_video}")
            if not args.all:
                sys.exit(1)
            continue

        # Determine start frame and search center from experiment metadata
        exp_range = params["experiment_range"]
        start_frame = args.start_frame if args.start_frame is not None else exp_range["eo_start"]
        search_center = args.search_center if args.search_center is not None else exp_range["th_start"]

        try:
            df, csv_path = match_video_frames(
                query_video_path=query_video,
                query_frame_number=start_frame,
                target_video_path=target_video,
                search_center=search_center,
                initial_window_length=args.window_length or fm_cfg.get("initial_window_length", 20),
                matcher=matcher,
                match_threshold=cfg.get("matcher", {}).get("match_threshold", 0.2),
                stop_frame=args.stop_frame,
                best_percent=args.best_percent or fm_cfg.get("best_percent", 50),
                use_adaptive_window=args.adaptive_window if args.adaptive_window is not None else fm_cfg.get("adaptive_window", False),
                save_interval=args.save_interval or fm_cfg.get("save_interval", 50),
                max_window_length=args.max_window_length or fm_cfg.get("max_window_length", 200),
            )
            print(f"  -> {csv_path} ({len(df)} frames)")
        except Exception as e:
            print(f"  ERROR: {e}")
            if not args.all:
                sys.exit(1)


# -----------------------------------------------------------------------
# resume subcommand
# -----------------------------------------------------------------------

def cmd_resume(args):
    cfg = load_config(args.config)
    fm_cfg = cfg.get("frame_matching", {})
    matcher = make_matcher(cfg)

    df, csv_path = resume_matching(
        csv_path=args.csv,
        matcher=matcher,
        stop_frame=args.stop_frame,
        save_interval=args.save_interval or fm_cfg.get("save_interval", 50),
        max_window_length=args.max_window_length or fm_cfg.get("max_window_length", 200),
    )
    print(f"  -> {csv_path} ({len(df)} frames total)")


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spatial alignment pipeline for RGB-thermal registration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, help="Path to alignment.yaml config file")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- homography --
    p_homo = sub.add_parser("homography", help="Compute homography matrix")
    grp = p_homo.add_mutually_exclusive_group(required=True)
    grp.add_argument("--experiment", type=str, help="Experiment ID (e.g. 0798)")
    grp.add_argument("--all", action="store_true", help="Process all experiments")
    p_homo.add_argument("--downloads-dir", type=str, default=None)
    p_homo.add_argument("--output-dir", type=str, default=None)
    p_homo.add_argument("--debug", action="store_true")
    p_homo.set_defaults(func=cmd_homography)

    # -- match --
    p_match = sub.add_parser("match", help="Run frame-to-frame matching")
    grp2 = p_match.add_mutually_exclusive_group(required=True)
    grp2.add_argument("--experiment", type=str, help="Experiment ID (e.g. 0798)")
    grp2.add_argument("--all", action="store_true", help="Process all experiments")
    p_match.add_argument("--downloads-dir", type=str, default=None)
    p_match.add_argument("--start-frame", type=int, default=None)
    p_match.add_argument("--stop-frame", type=int, default=None)
    p_match.add_argument("--search-center", type=int, default=None)
    p_match.add_argument("--window-length", type=int, default=None)
    p_match.add_argument("--best-percent", type=int, default=None)
    p_match.add_argument("--adaptive-window", action="store_true", default=None)
    p_match.add_argument("--save-interval", type=int, default=None)
    p_match.add_argument("--max-window-length", type=int, default=None)
    p_match.set_defaults(func=cmd_match)

    # -- resume --
    p_resume = sub.add_parser("resume", help="Resume interrupted matching")
    p_resume.add_argument("--csv", type=str, required=True, help="Path to partial CSV")
    p_resume.add_argument("--stop-frame", type=int, default=None)
    p_resume.add_argument("--save-interval", type=int, default=None)
    p_resume.add_argument("--max-window-length", type=int, default=None)
    p_resume.set_defaults(func=cmd_resume)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
