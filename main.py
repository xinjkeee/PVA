import argparse
from pathlib import Path

from src.audio.preprocess_audio import process_random_files
from scripts.visualize_audio import visualize_before_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./data/raw/ru/clips")
    parser.add_argument("--processed-dir", default="./data/processed/ru/results")
    parser.add_argument("--visual-dir", default="./data/visualizations")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")

    args = parser.parse_args()

    processed = process_random_files(
        input_dir=args.input_dir,
        output_dir=args.processed_dir,
        n=args.n,
        seed=args.seed,
        augment=args.augment,
    )

    visual_dir = Path(args.visual_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(processed, start=1):
        out_png = visual_dir / f"sample_{idx}.png"
        visualize_before_after(
            before_path=str(item.source_path),
            after_path=str(item.output_path),
            sample_num=idx,
            save_path=str(out_png),
        )

        print(f"[{idx}] saved: {item.output_path.name}, {out_png.name}")


if __name__ == "__main__":
    main()
