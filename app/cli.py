"""CLI entry point: python -m app.cli --help"""
import argparse
import json

from app.core import classify_file, classify_report, train_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agricultural-report-classifier",
        description="Fine-tune GPT-2 and classify short agricultural field reports.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the starter package is wired up")
    train = sub.add_parser("train", help="fine-tune GPT-2 on the bundled agricultural examples")
    train.add_argument("--epochs", type=int, default=8)
    single = sub.add_parser("classify", help="classify one report")
    single.add_argument("--text", required=True, help="farmer report text")
    batch = sub.add_parser("batch", help="classify each non-empty line in a text file")
    batch.add_argument("--input", required=True, help="input .txt file")
    batch.add_argument("--output", default="results.json", help="JSON output path (default: results.json)")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Agricultural Report Classifier is ready.")
    elif args.command == "train":
        path = train_model(epochs=args.epochs)
        print(f"Fine-tuned GPT-2 model saved to: {path}")
    elif args.command == "classify":
        print(json.dumps(classify_report(args.text), indent=2, ensure_ascii=False))
    elif args.command == "batch":
        results = classify_file(args.input, args.output)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(results)} classifications to {args.output}")


if __name__ == "__main__":
    main()
