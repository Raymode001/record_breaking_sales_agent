"""Commands for the sales agent homework.

    python run.py test                     free   check your setup and your two files
    python run.py prompt                   free   show exactly what the model reads
    python run.py run --case CASE          cents  one practice case
    python run.py run --suite PATH         paid   one campaign, e.g. development/full_campaign/suite.json
    python run.py final                    paid   your prompt and tools on the 40-account campaign, five times
    python run.py check OUTPUT             small  audit the quality of saved runs
    python run.py spend                    free   what your API calls have cost
    python run.py package OUTPUT           free   the ZIP you hand in: your two files and the five runs

--variant chooses what runs: baseline (supplied prompt and tools), prompt (your
prompt, supplied tools) or tools (your prompt and tools, the default).
"""

import argparse
import contextlib
import io
import json
import secrets
import subprocess
import sys
from pathlib import Path

try:
    from sales_sim import experiments, local
    from sales_sim.quality import check_folder, comparison_lines, summary_lines
    from sales_sim.response_cache import cache_directory, cap_from_environment, spending_report
    from sales_sim.student_setup import check_setup, course_settings, model_input, model_settings
    from sales_sim.student_submission import VARIANTS
    from sales_sim.submissions import REQUIRED_RUNS, package_submission
    from sales_sim.suites import FORMAL_SUITE
except ImportError as error:
    raise SystemExit(f"Could not load the supplied runtime: {error}\n"
                     "Run `python -m pip install -r requirements.txt` with this Python, "
                     "and restart the notebook kernel if you use one.") from None

ROOT = Path(__file__).resolve().parent


# The commands ------------------------------------------------------------------

def test(args):
    if not check_setup(ROOT)["ok"]:
        return 1
    print("\nRunning the offline tests in tests/ ...", flush=True)
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT).returncode


def prompt(args):
    seen = model_input(ROOT, variant=args.variant, suite=args.suite, case=args.case)
    print(f"=== your prompt: the first message ({seen['case_id']}, {seen['variant']}) ===\n{seen['developer']}\n")
    print(f"=== the note that opens day 1 ===\n{seen['user']}\n")
    print(f"=== {len(seen['tools'])} tools, sent with every model call ===")
    print(json.dumps(seen["tools"], indent=2, ensure_ascii=False))
    size = len(json.dumps(seen["tools"])) // 4
    print(f"\nRough size: prompt {len(seen['developer']) // 4:,} tokens, tool definitions {size:,} tokens.")


def run(args):
    args.name = None
    if args.agent == "reference":
        if not args.case:
            raise ValueError("The scripted demo runs on practice cases: use --case.")
        args.model = args.customer_model = None
        use_course_limits(args)
        print("Seller: offline scripted demonstration (no model); customers: fixed offline simulator")
    else:
        use_course_settings(args)
        if args.case:
            args.customer_model = None                   # practice customers are scripted
        print(f"Seller: {args.model}; customers: {'fixed offline simulator' if args.case else args.customer_model}")
    args.output = args.output or experiments.default_output(ROOT, args.agent or args.variant)
    code = local.run(args, ROOT)
    if (Path(args.output) / "runs").is_dir():
        with contextlib.redirect_stdout(io.StringIO()):
            local.inspect_output(args.output)            # writes inspect.html, the run report
    print(f"Saved: {args.output} (open inspect.html)")
    return code


def final(args):
    args.case, args.agent, args.variant = None, None, "tools"
    args.suite, args.repeats = ROOT / FORMAL_SUITE, REQUIRED_RUNS
    args.model = args.customer_model = None                  # the hand-in uses the course settings
    args.output = args.output or experiments.default_output(ROOT, "final")
    code = run(args)
    print(f"Hand in: python run.py package {_shown(args.output)}")
    return code


def check(args):
    reports = check_folder(args.output, ROOT, use_judge=not args.no_judge, model=args.judge_model, audit=args.audit)
    for report in reports:
        print("\n".join(summary_lines(report, details=args.details)))
    for line in comparison_lines(reports):
        print(line)
    if reports:
        print("Saved quality.md in each run folder. The judge's verdicts are feedback, not grades.")


def spend(args):
    print(spending_report(cache_directory(ROOT), cap_usd=cap_from_environment()))


def package(args):
    args.output = args.output or experiments.default_output(ROOT, "submission", ".zip")
    package_submission(ROOT, args.output, runs=args.runs, code_only=args.code_only)
    print(f"Submission: {args.output}")


def _shown(path):
    try:
        return Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return path


# Settings from course.json ---------------------------------------------------

def use_course_limits(args):
    for name, value in course_settings(ROOT)["limits"].items():
        setattr(args, name, value)


def use_course_settings(args):
    settings = model_settings(ROOT, model=args.model, customer_model=args.customer_model)
    args.model, args.customer_model = settings["model"], settings["customer_model"]
    use_course_limits(args)
    args.fresh_salt = secrets.token_hex(4) if args.fresh else None


# Arguments -----------------------------------------------------------------------

def paid_options(parser, *, repeats):
    parser.add_argument("--repeats", type=int, default=repeats)
    parser.add_argument("--output", type=Path, help="Default: a new dated folder under outputs/.")
    parser.add_argument("--fresh", action="store_true", help="Draw new samples instead of replaying saved ones.")
    parser.add_argument("--model", help="Override the seller model in course.json (exploration only).")
    parser.add_argument("--customer-model", help="Override the customer model in course.json (exploration only).")


def parser():
    main = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = main.add_subparsers(dest="command", required=True)

    commands.add_parser("test", help="Check your setup and your two files (free).")

    show = commands.add_parser("prompt", help="Show exactly what the model reads (free).")
    show.add_argument("--case", help="A practice case. Default: the 40-account campaign.")
    show.add_argument("--suite", type=Path, help="Another campaign's suite.json.")
    show.add_argument("--variant", choices=VARIANTS, default="tools")

    one = commands.add_parser("run", help="Run one practice case or one campaign (paid).")
    where = one.add_mutually_exclusive_group(required=True)
    where.add_argument("--case", help="A practice case, e.g. requirements_update.")
    where.add_argument("--suite", type=Path, help="A campaign, e.g. development/full_campaign/suite.json.")
    one.add_argument("--variant", choices=VARIANTS, default="tools")
    one.add_argument("--agent", choices=("reference",), help="The free scripted demo (practice cases only).")
    paid_options(one, repeats=1)

    hand_in_runs = commands.add_parser("final", help="Your prompt and tools on the 40-account campaign, "
                                                     "five times: the runs you hand in (paid).")
    hand_in_runs.add_argument("--output", type=Path, help="Default: a new dated folder under outputs/.")
    hand_in_runs.add_argument("--fresh", action="store_true", help="Draw new samples instead of replaying saved ones.")

    audit = commands.add_parser("check", help="Audit the quality of saved runs (small cost, cached).")
    audit.add_argument("output", type=Path, help="A run or batch folder under outputs/.")
    audit.add_argument("--details", action="store_true", help="Show the evidence for every item not met.")
    audit.add_argument("--no-judge", action="store_true", help="Only what the records settle; free.")
    audit.add_argument("--judge-model", help="Override the judge model in course.json.")
    audit.add_argument("--audit", type=Path, help="An edited audit.json. Default: the campaign's own.")

    commands.add_parser("spend", help="What your API calls have cost (free).")

    hand_in = commands.add_parser("package", help="Build the ZIP you hand in (free).")
    hand_in.add_argument("runs", nargs="?", type=Path, help="The folder python run.py final saved under outputs/.")
    hand_in.add_argument("--output", type=Path, help="Default: a new dated ZIP under outputs/.")
    hand_in.add_argument("--code-only", action="store_true", help="Check the two files only; not a hand-in.")
    return main


COMMANDS = {"test": test, "prompt": prompt, "run": run, "final": final,
            "check": check, "spend": spend, "package": package}


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        return COMMANDS[args.command](args) or 0
    except (OSError, ValueError, TypeError, SyntaxError) as error:
        arguments.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
