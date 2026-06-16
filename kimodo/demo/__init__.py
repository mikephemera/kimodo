# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001
import argparse

from kimodo.model import DEFAULT_MODEL
from kimodo.model.registry import resolve_model_name

from .app import Demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the kimodo demo UI.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Default model to load (e.g. Kimodo-SOMA-RP-v1, kimodo-soma-rp, or SOMA).",
    )
    parser.add_argument(
        "--auto-save-dir",
        type=str,
        default="/mnt/datafiles/Work-syncfree/unitree_sim2x/assets/motions/g1_29dof/kimodo_autosave",
        help="Auto-save directory for generated motions. Auto-save is enabled by default.",
    )
    parser.add_argument(
        "--auto-save-format",
        type=str,
        default="CSV",
        choices=["NPZ", "BVH", "CSV", "AMASS NPZ"],
        help="Format for auto-saved motions (default: CSV).",
    )
    parser.add_argument(
        "--headless-port",
        type=int,
        default=None,
        help="Enable headless HTTP API on this port (e.g. 9551).",
    )
    parser.add_argument(
        "--headless-host",
        type=str,
        default="0.0.0.0",
        help="Bind address for headless API (default: 0.0.0.0).",
    )
    args = parser.parse_args()

    resolved = resolve_model_name(args.model, "Kimodo")
    demo = Demo(
        default_model_name=resolved,
        auto_save_dir=args.auto_save_dir,
        auto_save_format=args.auto_save_format,
        headless_host=args.headless_host,
        headless_port=args.headless_port,
    )
    demo.run()


if __name__ == "__main__":
    main()
