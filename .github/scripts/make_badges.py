"""Write the shields.io endpoint file for the README test badge.

The badge is self-hosted: this writes a small JSON file into `website/`,
which the deploy job publishes to GitHub Pages alongside the site, and the
README points shields.io at that URL. That keeps the number under this
repository's control, with no third-party account, no upload token, and no
extra repository secret to rotate.

The one cost of that choice: the badge only changes when a deploy runs, which
is on a push to main. If a per-pull-request number ever becomes worth having,
Codecov is the drop-in.

Coverage is measured in CI and printed in the run log, but deliberately not
published as a badge; see the "Coverage" section of CONTRIBUTING.md for why.

The input is the machine-readable report pytest already produces, a JUnit XML
file (`--junitxml`). Standard library only, so the job needs nothing beyond
the test dependencies.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = 1


def endpoint(label, message, color):
    """One shields.io endpoint payload."""
    return {
        "schemaVersion": SCHEMA,
        "label": label,
        "message": message,
        "color": color,
    }


def read_junit(path):
    """Return (total, failed) from a pytest JUnit XML report.

    pytest writes a <testsuites> wrapper around one <testsuite>; the counts
    live on the inner element, so read that rather than the root.
    """
    root = ET.parse(path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        raise SystemExit(f"{path}: no <testsuite> element")

    def count(name):
        return int(suite.get(name, 0))

    total = count("tests") - count("skipped")
    failed = count("failures") + count("errors")
    return total, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--junit", required=True, help="pytest JUnit XML report")
    ap.add_argument("--out", required=True, help="directory to write into")
    args = ap.parse_args()

    total, failed = read_junit(args.junit)

    # A failing suite fails the job before this runs, so the red branch is
    # unreachable in CI. It exists so that running this by hand on a broken
    # local report cannot quietly produce a green badge.
    if failed:
        tests = endpoint("tests", f"{failed} failing", "red")
    else:
        tests = endpoint("tests", f"{total} passing", "brightgreen")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "tests.json"
    dest.write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")
    print(f"   {dest}: {tests['message']}", file=sys.stderr)


if __name__ == "__main__":
    main()
