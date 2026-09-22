"""Command-line entry point. Every subcommand takes URIs (local or gs://) and a config."""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("soccer-cv")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("run", help="full match: ingest -> tier A (all shots) -> tier B -> tier C")
    m.add_argument("--video-uri", required=True)
    m.add_argument("--run-uri", required=True)
    m.add_argument("--config", required=True)
    m.add_argument("--backend", default="local", choices=["local", "batch"])
    m.add_argument("--backend-kwargs", default="{}", help="JSON, e.g. '{\"project\":..., \"region\":..., \"image\":...}'")
    m.add_argument("--all-shots", action="store_true", help="also process non-main-camera shots")
    m.add_argument("--replay-detections", default=None, help="parquet of precomputed detections (no detector)")

    a = sub.add_parser("tier-a", help="one shot: detect -> track -> ball -> summarize (what a Batch task runs)")
    a.add_argument("--run-uri", required=True)
    a.add_argument("--shot-id", required=True)
    a.add_argument("--config", required=True)

    b = sub.add_parser("tier-b", help="identity resolution over all shots")
    b.add_argument("--run-uri", required=True)
    b.add_argument("--config", required=True)

    c = sub.add_parser("tier-c", help="fuse + render overlay")
    c.add_argument("--run-uri", required=True)
    c.add_argument("--config", required=True)

    i = sub.add_parser("ingest", help="stage 0 only")
    i.add_argument("--video-uri", required=True)
    i.add_argument("--run-uri", required=True)
    i.add_argument("--config", required=True)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .core import Storage, load_config
    if args.cmd == "run":
        from .pipeline import run_match
        out = run_match(args.video_uri, args.run_uri, args.config, backend=args.backend,
                        backend_kwargs=json.loads(args.backend_kwargs), only_main=not args.all_shots,
                        replay_uri=args.replay_detections)
        print(json.dumps(out, indent=1))
    elif args.cmd == "tier-a":
        from .pipeline import run_tier_a_shot
        cfg = load_config(Storage.localize(args.config))
        run_tier_a_shot(args.run_uri, args.shot_id, cfg)
    elif args.cmd == "tier-b":
        from .tiers.tier_b_identity import IdentityTier
        cfg = load_config(Storage.localize(args.config))
        print(json.dumps(IdentityTier(cfg).execute(Storage.join(args.run_uri, "tier_a"), Storage.join(args.run_uri, "tier_b")), indent=1))
    elif args.cmd == "tier-c":
        from .tiers.tier_c_render import FuseStage, RenderStage
        cfg = load_config(Storage.localize(args.config))
        FuseStage(cfg).execute(args.run_uri, Storage.join(args.run_uri, "tier_c"))
        print(json.dumps(RenderStage(cfg).execute(args.run_uri, Storage.join(args.run_uri, "tier_c")), indent=1))
    elif args.cmd == "ingest":
        from .stages.s0_ingest import IngestStage
        cfg = load_config(Storage.localize(args.config))
        print(json.dumps(IngestStage(cfg).execute(args.video_uri, Storage.join(args.run_uri, "ingest")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
