"""CLI entry point."""
import argparse, json
from app.core import train_model, classify_report
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agricultural-report-classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train"); train.add_argument("--epochs", type=int, default=8)
    single = sub.add_parser("classify"); single.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    if args.command == "train": print(f"Model saved to: {train_model(epochs=args.epochs)}")
    elif args.command == "classify": print(json.dumps(classify_report(args.text), indent=2))
if __name__ == "__main__": main()
