"""CLI entry point."""
import argparse
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agricultural-report-classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check the project setup")
    args = parser.parse_args(argv)
    if args.command == "hello": print("Agricultural Report Classifier is ready.")
if __name__ == "__main__": main()
