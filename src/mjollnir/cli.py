"""``mjollnir-params`` — shell-level hygiene for parameter artifacts.

Subcommands:

* ``show <path>``      — pretty-print parameters + metadata, verifying integrity
* ``verify <path>...`` — exit non-zero if any artifact fails hash verification
* ``derive <path>``    — apply --scale/--set shocks, provenance-chained
* ``scenario <path> <name>`` — apply a library scenario (vol_spike, ...)

Examples::

    mjollnir-params show data/params/spy_heston.json
    mjollnir-params verify data/params/*.json
    mjollnir-params derive spy.json --scale theta=4 --note "vol crisis" -o crisis.json
    mjollnir-params scenario spy.json vol_spike --severity 2.5 -o spike.json
"""

from __future__ import annotations

import argparse
import sys

from mjollnir.params import ParamSet, ParamSetIntegrityError
from mjollnir.scenarios import LIBRARY


def _kv_pairs(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not _ or not key:
            raise SystemExit(f"expected NAME=VALUE, got {item!r}")
        out[key] = float(value)
    return out


def _show(args) -> int:
    try:
        ps = ParamSet.load(args.path)
    except ParamSetIntegrityError as e:
        print(f"INTEGRITY FAILURE: {e}", file=sys.stderr)
        return 1
    print(f"model     : {ps.model}   measure: {ps.measure}   asset: {ps.asset or '-'}")
    print(f"window    : {ps.window or '-'}")
    print(f"source    : {ps.source or '-'}")
    if ps.note:
        print(f"note      : {ps.note}")
    print(f"created   : {ps.created_at}   mjollnir {ps.mjollnir_version}")
    print(f"hash      : {ps.content_hash()}")
    if ps.parent_hash:
        print(f"parent    : {ps.parent_hash}")
    print("params    :")
    for k, v in sorted(ps.params.items()):
        print(f"  {k:10s} {v:+.6g}")
    return 0


def _verify(args) -> int:
    failures = 0
    for path in args.paths:
        try:
            ParamSet.load(path)
            print(f"OK      {path}")
        except (ParamSetIntegrityError, KeyError, ValueError) as e:
            print(f"FAILED  {path}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def _derive(args) -> int:
    ps = ParamSet.load(args.path)
    child = ps.derive(
        scale=_kv_pairs(args.scale or []),
        set_=_kv_pairs(args.set or []),
        note=args.note,
    )
    out = child.save(args.output)
    print(f"wrote {out}  (parent {ps.content_hash()[:12]}…)")
    return 0


def _scenario(args) -> int:
    if args.name not in LIBRARY:
        raise SystemExit(f"unknown scenario {args.name!r}; "
                         f"available: {', '.join(sorted(LIBRARY))}")
    factory = LIBRARY[args.name]
    scenario = factory(args.severity) if args.severity is not None else factory()
    child = scenario.apply(ParamSet.load(args.path))
    out = child.save(args.output)
    print(f"applied {scenario.name}: {scenario.description}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mjollnir-params",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("show", help="pretty-print an artifact (verifies hash)")
    p.add_argument("path")
    p.set_defaults(fn=_show)

    p = sub.add_parser("verify", help="verify integrity of artifacts")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=_verify)

    p = sub.add_parser("derive", help="apply manual shocks, provenance-chained")
    p.add_argument("path")
    p.add_argument("--scale", action="append", metavar="NAME=FACTOR")
    p.add_argument("--set", action="append", metavar="NAME=VALUE")
    p.add_argument("--note")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=_derive)

    p = sub.add_parser("scenario", help="apply a library scenario")
    p.add_argument("path")
    p.add_argument("name")
    p.add_argument("--severity", type=float)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=_scenario)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
