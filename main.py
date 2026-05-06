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


def run_prepare_kws(args: argparse.Namespace) -> None:
    from src.kws.data import build_kws_manifest_from_asr

    stats = build_kws_manifest_from_asr(
        source_manifest=Path(args.source_manifest),
        keyword=args.keyword,
        output_path=Path(args.output),
        max_positives=args.max_positives,
        max_negatives=args.max_negatives,
        seed=args.seed,
        match=args.match,
    )
    print(
        f"keyword={stats['keyword']} positives={stats['positives']} "
        f"negatives={stats['negatives']} total={stats['total']}"
    )
    print(f"saved={stats['output']}")


def run_train_kws(args: argparse.Namespace) -> None:
    from src.asr.text import normalize_text
    from src.kws.data import split_kws_manifest
    from src.kws.pipeline import train_kws_model

    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train KWS manifest not found: {train_manifest}. "
            f"Create it with: python main.py prepare-kws --keyword \"{args.keyword}\" "
            f"--output {train_manifest}"
        )
    if not val_manifest.exists():
        if not args.auto_split_val:
            raise FileNotFoundError(
                f"Validation KWS manifest not found: {val_manifest}. "
                f"Create it with: python main.py prepare-kws --keyword \"{args.keyword}\" "
                f"--source-manifest data/asr/common_voice_ru/manifest_val.jsonl --output {val_manifest}"
            )

        autosplit_train = train_manifest.with_name(f"{train_manifest.stem}_autosplit.jsonl")
        split_stats = split_kws_manifest(
            source_manifest=train_manifest,
            train_output=autosplit_train,
            val_output=val_manifest,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )
        train_manifest = autosplit_train
        print(
            f"created validation split: train={split_stats['train_records']} "
            f"val={split_stats['val_records']} saved={val_manifest}"
        )

    checkpoint = args.checkpoint
    if checkpoint is None:
        keyword_slug = "_".join(normalize_text(args.keyword).split()) or "keyword"
        checkpoint = f"./data/kws/checkpoints/{keyword_slug}_cnn.pt"

    feature_config = {
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "f_min": args.f_min,
        "f_max": args.f_max,
    }
    result = train_kws_model(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        checkpoint_path=Path(checkpoint),
        keyword=args.keyword,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        feature_config=feature_config,
    )
    print(f"saved={result['checkpoint']} device={result['device']}")


def run_detect_kws(args: argparse.Namespace) -> None:
    from src.kws.pipeline import detect_keyword

    result = detect_keyword(
        audio_path=Path(args.audio),
        checkpoint_path=Path(args.checkpoint),
        threshold=args.threshold,
        device=args.device,
    )
    print(
        f"keyword={result['keyword']} probability={result['probability']:.4f} "
        f"threshold={result['threshold']:.2f} wake={result['wake']}"
    )


def run_tts(args: argparse.Namespace) -> None:
    from src.tts.system import synthesize_with_macos_say

    output_path = synthesize_with_macos_say(
        text=args.text,
        output_path=Path(args.output),
        voice=args.voice,
        rate=args.rate,
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

    prepare_kws_parser = subparsers.add_parser("prepare-kws", help="Build KWS labels from ASR manifest")
    prepare_kws_parser.add_argument("--source-manifest", default="./data/asr/common_voice_ru/manifest_train.jsonl")
    prepare_kws_parser.add_argument("--keyword", required=True)
    prepare_kws_parser.add_argument("--output", default="./data/kws/kws_train.jsonl")
    prepare_kws_parser.add_argument("--max-positives", type=int, default=None)
    prepare_kws_parser.add_argument("--max-negatives", type=int, default=2500)
    prepare_kws_parser.add_argument("--seed", type=int, default=42)
    prepare_kws_parser.add_argument("--match", choices=["contains", "exact"], default="contains")
    prepare_kws_parser.set_defaults(func=run_prepare_kws)

    train_kws_parser = subparsers.add_parser("train-kws", help="Train a small CNN keyword spotter")
    train_kws_parser.add_argument("--train-manifest", default="./data/kws/kws_train.jsonl")
    train_kws_parser.add_argument("--val-manifest", default="./data/kws/kws_val.jsonl")
    train_kws_parser.add_argument("--checkpoint", default=None)
    train_kws_parser.add_argument("--keyword", required=True)
    train_kws_parser.add_argument("--epochs", type=int, default=5)
    train_kws_parser.add_argument("--batch-size", type=int, default=16)
    train_kws_parser.add_argument("--lr", type=float, default=1e-3)
    train_kws_parser.add_argument("--device", default="auto")
    train_kws_parser.add_argument("--limit-train", type=int, default=None)
    train_kws_parser.add_argument("--limit-val", type=int, default=None)
    train_kws_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_kws_parser.add_argument("--split-seed", type=int, default=42)
    train_kws_parser.add_argument("--no-auto-split-val", dest="auto_split_val", action="store_false")
    train_kws_parser.add_argument("--sample-rate", type=int, default=16000)
    train_kws_parser.add_argument("--duration-sec", type=float, default=1.5)
    train_kws_parser.add_argument("--n-fft", type=int, default=400)
    train_kws_parser.add_argument("--hop-length", type=int, default=160)
    train_kws_parser.add_argument("--n-mels", type=int, default=40)
    train_kws_parser.add_argument("--f-min", type=float, default=80.0)
    train_kws_parser.add_argument("--f-max", type=float, default=7600.0)
    train_kws_parser.set_defaults(auto_split_val=True)
    train_kws_parser.set_defaults(func=run_train_kws)

    detect_kws_parser = subparsers.add_parser("detect-kws", help="Run keyword spotting on one file")
    detect_kws_parser.add_argument("--audio", required=True)
    detect_kws_parser.add_argument("--checkpoint", required=True)
    detect_kws_parser.add_argument("--threshold", type=float, default=0.5)
    detect_kws_parser.add_argument("--device", default="auto")
    detect_kws_parser.set_defaults(func=run_detect_kws)

    tts_parser = subparsers.add_parser("tts", help="Synthesize a spoken response")
    tts_parser.add_argument("--text", required=True)
    tts_parser.add_argument("--output", default="./data/tts/response.aiff")
    tts_parser.add_argument("--voice", default=None)
    tts_parser.add_argument("--rate", type=int, default=None)
    tts_parser.set_defaults(func=run_tts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
