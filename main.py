import argparse
from pathlib import Path


def run_process(args: argparse.Namespace) -> None:
    from src.audio.preprocess_audio import process_random_files
    from scripts.visualize_audio import visualize_before_after

    processed_items = process_random_files(
        input_dir=args.input_dir,
        output_dir=args.processed_dir,
        n=args.n,
        seed=args.seed,
        augment=args.augment,
    )

    visual_dir = Path(args.visual_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(processed_items, start=1):
        out_png = visual_dir / f"sample_{idx}.png"
        visualize_before_after(
            before_path=str(item.source_path),
            after_path=str(item.output_path),
            sample_num=idx,
            save_path=str(out_png),
        )

        print(f"[{idx}] saved: {item.output_path.name}, {out_png.name}")


def run_visualize(args: argparse.Namespace) -> None:
    from scripts.visualize_audio import visualize_before_after

    visualize_before_after(
        before_path=args.before,
        after_path=args.after,
        sample_num=args.sample_num,
        save_path=args.output,
    )
    print(f"saved={args.output}" if args.output else "visualization complete")


def run_compare_vad(args: argparse.Namespace) -> None:
    from scripts.compare_vad import compare_vad

    result = compare_vad(
        audio_path=args.audio,
        save_path=args.output,
        energy_frame_ms=args.energy_frame_ms,
        energy_hop_ms=args.energy_hop_ms,
        webrtc_frame_ms=args.webrtc_frame_ms,
        webrtc_mode=args.webrtc_mode,
    )

    print(f"audio_seconds={result['audio_seconds']:.2f}")
    print(
        f"energy_segments={len(result['energy_segments'])} "
        f"energy_speech_seconds={result['energy_speech_seconds']:.2f} "
        f"energy_threshold_db={result['energy_threshold_db']:.1f}"
    )
    print(
        f"webrtc_segments={len(result['webrtc_segments'])} "
        f"webrtc_speech_seconds={result['webrtc_speech_seconds']:.2f}"
    )
    print(f"saved={args.output}")


def run_prepare_asr(args: argparse.Namespace) -> None:
    from scripts.prepare_asr_dataset import prepare_dataset

    stats = prepare_dataset(
        metadata_dir=Path(args.metadata_dir),
        clips_dir=Path(args.clips_dir),
        output_dir=Path(args.output_dir),
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        limit_per_split=args.limit_per_split,
    )

    for split in ["train", "val", "test", "all"]:
        split_stats = stats[split]
        print(
            f"{split}: records={split_stats['records']} "
            f"hours={split_stats['duration_hours']} "
            f"missing_duration={split_stats['missing_duration']} "
            f"duplicate_texts={split_stats['duplicate_normalized_texts']}"
        )
    print(f"saved={args.output_dir}")


def run_eval_asr(args: argparse.Namespace) -> None:
    from scripts.evaluate_whisper_asr import evaluate_manifest

    output_path = args.output
    if output_path is None:
        manifest_path = Path(args.manifest)
        split_name = manifest_path.stem.replace("manifest_", "")
        output_path = manifest_path.with_name(
            f"whisper_predictions_{split_name}_{args.model}.jsonl"
        )

    summary = evaluate_manifest(
        manifest_path=Path(args.manifest),
        output_path=Path(output_path),
        model_name=args.model,
        download_root=Path(args.download_root),
        language=args.language,
        limit=args.limit,
        fp16=args.fp16,
    )

    print(
        f"summary: samples={summary['samples']} "
        f"WER={summary['wer']:.4f} CER={summary['cer']:.4f}"
    )
    print(f"saved={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio preprocessing, VAD, and ASR toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process random audio files")
    process_parser.add_argument("--input-dir", default="./data/raw/ru/clips")
    process_parser.add_argument("--processed-dir", default="./data/processed/ru/results")
    process_parser.add_argument("--visual-dir", default="./data/visualizations")
    process_parser.add_argument("--n", type=int, default=5)
    process_parser.add_argument("--seed", type=int, default=42)
    process_parser.add_argument("--augment", action="store_true")
    process_parser.set_defaults(func=run_process)

    visualize_parser = subparsers.add_parser("visualize", help="Compare two audio files visually")
    visualize_parser.add_argument("--before", required=True)
    visualize_parser.add_argument("--after", required=True)
    visualize_parser.add_argument("--sample-num", type=int, required=True)
    visualize_parser.add_argument("--output", default=None)
    visualize_parser.set_defaults(func=run_visualize)

    vad_parser = subparsers.add_parser("compare-vad", help="Compare energy VAD and WebRTC VAD")
    vad_parser.add_argument("--audio", required=True, help="Path to source audio file")
    vad_parser.add_argument("--output", default="./data/visualizations/vad_compare.png")
    vad_parser.add_argument("--webrtc-mode", type=int, default=2, choices=[0, 1, 2, 3])
    vad_parser.add_argument("--webrtc-frame-ms", type=int, default=30, choices=[10, 20, 30])
    vad_parser.add_argument("--energy-frame-ms", type=float, default=30.0)
    vad_parser.add_argument("--energy-hop-ms", type=float, default=10.0)
    vad_parser.set_defaults(func=run_compare_vad)

    prepare_parser = subparsers.add_parser("prepare-asr", help="Build ASR JSONL manifests")
    prepare_parser.add_argument("--metadata-dir", default="./data/raw/ru/validation")
    prepare_parser.add_argument("--clips-dir", default="./data/raw/ru/clips")
    prepare_parser.add_argument("--output-dir", default="./data/asr/common_voice_ru")
    prepare_parser.add_argument("--min-duration-ms", type=int, default=200)
    prepare_parser.add_argument("--max-duration-ms", type=int, default=30000)
    prepare_parser.add_argument("--limit-per-split", type=int, default=None)
    prepare_parser.set_defaults(func=run_prepare_asr)

    eval_parser = subparsers.add_parser("eval-asr", help="Run Whisper ASR and WER/CER")
    eval_parser.add_argument("--manifest", default="./data/asr/common_voice_ru/manifest_test.jsonl")
    eval_parser.add_argument("--output", default=None)
    eval_parser.add_argument("--model", default="tiny")
    eval_parser.add_argument("--download-root", default="./data/models/whisper")
    eval_parser.add_argument("--language", default="ru")
    eval_parser.add_argument("--limit", type=int, default=20)
    eval_parser.add_argument("--fp16", action="store_true")
    eval_parser.set_defaults(func=run_eval_asr)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
