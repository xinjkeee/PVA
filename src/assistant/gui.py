import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.assistant.pipeline import AssistantConfig, AssistantPipeline, AssistantResult

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_STATUS_COLORS = {
    "ok": "#22c55e",
    "empty": "#f59e0b",
    "skip": "#4b5563",
    "error": "#ef4444",
}

_STAGE_META = [
    ("VAD", "🎙"),
    ("KWS", "🔑"),
    ("ASR", "📝"),
    ("NLU", "🧠"),
    ("TTS", "🔊"),
    ("Speaker", "📢"),
]


class _StageCard(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTkFrame, name: str, icon: str, **kwargs) -> None:
        super().__init__(parent, corner_radius=10, border_width=1, border_color="#1e293b", **kwargs)

        ctk.CTkLabel(
            self,
            text=f"{icon}  {name}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e2e8f0",
        ).pack(pady=(12, 2), padx=8)

        self._status = ctk.CTkLabel(
            self,
            text="—",
            font=ctk.CTkFont(size=11),
            text_color="#4b5563",
        )
        self._status.pack(pady=(0, 2))

        self._detail = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=9),
            text_color="#64748b",
            wraplength=110,
        )
        self._detail.pack(pady=(0, 10), padx=6)

    def update(self, status: str, detail: str = "") -> None:
        color = _STATUS_COLORS.get(status, "#4b5563")
        self._status.configure(text=status, text_color=color)
        if len(detail) > 42:
            detail = detail[:39] + "…"
        self._detail.configure(text=detail)
        border = color if status in ("ok", "error", "empty") else "#1e293b"
        self.configure(border_color=border)

    def reset(self) -> None:
        self._status.configure(text="—", text_color="#4b5563")
        self._detail.configure(text="")
        self.configure(border_color="#1e293b")


class AssistantGUI(ctk.CTk):
    def __init__(self, config: AssistantConfig | None = None, record_seconds: float = 5.0) -> None:
        super().__init__()
        self.title("Voice Assistant")
        self.geometry("1120x740")
        self.minsize(920, 620)

        self.config = config or AssistantConfig()
        self.pipeline = AssistantPipeline(self.config)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._pulse_job: str | None = None
        self._pulse_state = False

        self.audio_path = tk.StringVar()
        self.record_seconds = tk.DoubleVar(value=record_seconds)
        self.whisper_model = tk.StringVar(value=self.config.whisper_model)
        self.kws_checkpoint = tk.StringVar(
            value=str(self.config.kws_checkpoint) if self.config.kws_checkpoint else ""
        )
        self.kws_threshold = tk.DoubleVar(value=self.config.kws_threshold)
        self.require_wake_word = tk.BooleanVar(value=self.config.require_wake_word)
        self.synthesize_response = tk.BooleanVar(value=self.config.synthesize_response)
        self.play_response = tk.BooleanVar(value=self.config.play_response)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.after(100, self._poll_events)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color="#0b0f1a")
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(99, weight=1)

        ctk.CTkLabel(
            sb,
            text="Voice\nAssistant",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#4f8ef7",
        ).grid(row=0, column=0, padx=22, pady=(28, 2), sticky="w")

        ctk.CTkLabel(
            sb,
            text="prototype",
            font=ctk.CTkFont(size=11),
            text_color="#334155",
        ).grid(row=1, column=0, padx=22, pady=(0, 24), sticky="w")

        row = 2

        def section(label: str) -> None:
            nonlocal row
            ctk.CTkLabel(
                sb,
                text=label,
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color="#334155",
            ).grid(row=row, column=0, padx=22, pady=(16, 0), sticky="w")
            row += 1

        def sep() -> None:
            nonlocal row
            ctk.CTkFrame(sb, height=1, fg_color="#1e293b").grid(
                row=row, column=0, padx=18, pady=(16, 0), sticky="ew"
            )
            row += 1

        # ASR
        section("ASR MODEL")
        ctk.CTkOptionMenu(
            sb,
            variable=self.whisper_model,
            values=["tiny", "base", "small", "medium", "turbo"],
            width=186,
            fg_color="#1e293b",
            button_color="#334155",
            button_hover_color="#475569",
        ).grid(row=row, column=0, padx=22, pady=(6, 0), sticky="w")
        row += 1

        sep()

        # KWS
        section("KWS CHECKPOINT")
        ctk.CTkEntry(
            sb,
            textvariable=self.kws_checkpoint,
            placeholder_text="Path to .pt file",
            width=186,
            fg_color="#1e293b",
            border_color="#334155",
        ).grid(row=row, column=0, padx=22, pady=(6, 0), sticky="w")
        row += 1

        section("KWS THRESHOLD")
        thr_row = ctk.CTkFrame(sb, fg_color="transparent")
        thr_row.grid(row=row, column=0, padx=22, pady=(6, 0), sticky="ew")
        row += 1

        self._thr_label = ctk.CTkLabel(
            thr_row,
            text=f"{self.kws_threshold.get():.2f}",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            width=34,
        )
        self._thr_label.pack(side="right")
        ctk.CTkSlider(
            thr_row,
            from_=0.0,
            to=1.0,
            variable=self.kws_threshold,
            command=lambda v: self._thr_label.configure(text=f"{v:.2f}"),
            width=148,
        ).pack(side="left")

        sep()

        # Options
        section("OPTIONS")
        for text, var in (
            ("Require wake word", self.require_wake_word),
            ("Synthesize response", self.synthesize_response),
            ("Play response", self.play_response),
        ):
            ctk.CTkSwitch(
                sb,
                text=text,
                variable=var,
                font=ctk.CTkFont(size=12),
            ).grid(row=row, column=0, padx=22, pady=(10, 0), sticky="w")
            row += 1

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)

        # ── File input bar ────────────────────────────────────────────────────
        file_bar = ctk.CTkFrame(main, fg_color="#1e293b", corner_radius=12)
        file_bar.grid(row=0, column=0, padx=22, pady=(22, 10), sticky="ew")
        file_bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            file_bar,
            text="File",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#475569",
        ).grid(row=0, column=0, padx=(16, 10), pady=12)

        ctk.CTkEntry(
            file_bar,
            textvariable=self.audio_path,
            placeholder_text="Select an audio file…",
            height=34,
            fg_color="#0f172a",
            border_color="#334155",
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=10)

        ctk.CTkButton(
            file_bar,
            text="Browse",
            width=76,
            height=34,
            command=self._choose_file,
            fg_color="#334155",
            hover_color="#475569",
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=2, padx=(0, 6), pady=10)

        ctk.CTkButton(
            file_bar,
            text="▶  Run",
            width=86,
            height=34,
            command=self._run_file,
            fg_color="#1d4ed8",
            hover_color="#2563eb",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=3, padx=(0, 14), pady=10)

        # ── Mic + duration row ────────────────────────────────────────────────
        mic_row = ctk.CTkFrame(main, fg_color="transparent")
        mic_row.grid(row=1, column=0, padx=22, pady=(0, 10), sticky="ew")
        mic_row.grid_columnconfigure(2, weight=1)

        self._mic_btn = ctk.CTkButton(
            mic_row,
            text="🎙  Record",
            width=160,
            height=50,
            corner_radius=25,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._run_microphone,
            fg_color="#be123c",
            hover_color="#e11d48",
        )
        self._mic_btn.grid(row=0, column=0)

        ctk.CTkLabel(
            mic_row,
            text="Duration",
            font=ctk.CTkFont(size=12),
            text_color="#475569",
        ).grid(row=0, column=1, padx=(20, 8))

        self._sec_label = ctk.CTkLabel(
            mic_row,
            text=f"{self.record_seconds.get():.0f}s",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            width=30,
        )

        ctk.CTkSlider(
            mic_row,
            from_=1,
            to=30,
            variable=self.record_seconds,
            command=lambda v: self._sec_label.configure(text=f"{v:.0f}s"),
            width=180,
        ).grid(row=0, column=2, sticky="w")

        self._sec_label.grid(row=0, column=3, padx=(6, 0))

        self._status_label = ctk.CTkLabel(
            mic_row,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=13),
            text_color="#475569",
        )
        self._status_label.grid(row=0, column=4, padx=(20, 0), sticky="e")
        mic_row.grid_columnconfigure(4, weight=1)

        # ── Pipeline stage cards ──────────────────────────────────────────────
        pipe_panel = ctk.CTkFrame(main, fg_color="#1e293b", corner_radius=12)
        pipe_panel.grid(row=2, column=0, padx=22, pady=(0, 10), sticky="ew")
        pipe_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            pipe_panel,
            text="PIPELINE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#334155",
        ).grid(row=0, column=0, padx=16, pady=(12, 6), sticky="w")

        cards_frame = ctk.CTkFrame(pipe_panel, fg_color="transparent")
        cards_frame.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        self._stage_cards: dict[str, _StageCard] = {}
        for i, (name, icon) in enumerate(_STAGE_META):
            cards_frame.grid_columnconfigure(i, weight=1)
            card = _StageCard(cards_frame, name, icon, fg_color="#0f172a")
            card.grid(row=0, column=i, padx=3, sticky="ew")
            self._stage_cards[name] = card

        # ── Transcript + response ─────────────────────────────────────────────
        out = ctk.CTkFrame(main, fg_color="transparent")
        out.grid(row=3, column=0, padx=22, pady=(0, 22), sticky="nsew")
        out.grid_columnconfigure(0, weight=1)
        out.grid_columnconfigure(1, weight=1)
        out.grid_rowconfigure(1, weight=1)

        for col, title in ((0, "TRANSCRIPT  (ASR)"), (1, "RESPONSE")):
            ctk.CTkLabel(
                out,
                text=title,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="#334155",
            ).grid(row=0, column=col, padx=(0 if col == 0 else 10, 0), pady=(0, 6), sticky="w")

        self._transcript = ctk.CTkTextbox(
            out,
            wrap="word",
            font=ctk.CTkFont(family="Menlo", size=13),
            corner_radius=10,
            fg_color="#1e293b",
            text_color="#e2e8f0",
            border_color="#334155",
            border_width=1,
        )
        self._transcript.grid(row=1, column=0, sticky="nsew")

        self._response = ctk.CTkTextbox(
            out,
            wrap="word",
            font=ctk.CTkFont(family="Menlo", size=13),
            corner_radius=10,
            fg_color="#1e293b",
            text_color="#e2e8f0",
            border_color="#334155",
            border_width=1,
        )
        self._response.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

    # ── Event loop ────────────────────────────────────────────────────────────

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "result":
                    self._render_result(payload)
                else:
                    self.status_var.set(f"Error: {payload}")
                    self._status_label.configure(text_color="#ef4444")
                    messagebox.showerror("Pipeline Error", str(payload))
                self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _render_result(self, result: AssistantResult) -> None:
        for stage, status, detail in result.stage_rows():
            if stage in self._stage_cards:
                self._stage_cards[stage].update(status, detail)

        self._transcript.configure(state="normal")
        self._transcript.delete("1.0", "end")
        self._transcript.insert("end", result.transcript)

        self._response.configure(state="normal")
        self._response.delete("1.0", "end")
        self._response.insert("end", result.response_text)

        msg = result.status
        if result.errors:
            msg += " — " + "; ".join(result.errors)
        self.status_var.set(msg)
        color = "#22c55e" if result.status == "ok" else "#f59e0b"
        self._status_label.configure(text_color=color)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=(("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aiff"), ("All", "*.*")),
        )
        if path:
            self.audio_path.set(path)

    def _run_file(self) -> None:
        path = self.audio_path.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please select an audio file first.")
            return
        self._start_worker("file", Path(path))

    def _run_microphone(self) -> None:
        self._start_worker("microphone", float(self.record_seconds.get()))

    def _start_worker(self, mode: str, payload: object) -> None:
        if self._busy:
            return
        self._sync_config()
        self._set_busy(True)
        self._clear_output()
        self.status_var.set("Running pipeline…")
        self._status_label.configure(text_color="#f59e0b")
        threading.Thread(target=self._worker, args=(mode, payload), daemon=True).start()

    def _worker(self, mode: str, payload: object) -> None:
        try:
            if mode == "microphone":
                result = self.pipeline.run_microphone(seconds=float(payload))
            else:
                result = self.pipeline.run_file(Path(payload))
            self.events.put(("result", result))
        except Exception as exc:
            self.events.put(("error", exc))

    def _clear_output(self) -> None:
        for card in self._stage_cards.values():
            card.reset()
        self._transcript.configure(state="normal")
        self._transcript.delete("1.0", "end")
        self._response.configure(state="normal")
        self._response.delete("1.0", "end")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._start_pulse()
        else:
            self._stop_pulse()
            self._mic_btn.configure(text="🎙  Record", fg_color="#be123c")

    def _start_pulse(self) -> None:
        self._pulse_state = False
        self._mic_btn.configure(text="⏳  Recording…")
        self._do_pulse()

    def _do_pulse(self) -> None:
        if not self._busy:
            return
        self._pulse_state = not self._pulse_state
        self._mic_btn.configure(fg_color="#7f1d1d" if self._pulse_state else "#be123c")
        self._pulse_job = self.after(550, self._do_pulse)

    def _stop_pulse(self) -> None:
        if self._pulse_job:
            self.after_cancel(self._pulse_job)
            self._pulse_job = None

    def _sync_config(self) -> None:
        self.config.whisper_model = self.whisper_model.get().strip() or "tiny"
        ckpt = self.kws_checkpoint.get().strip()
        self.config.kws_checkpoint = Path(ckpt) if ckpt else None
        self.config.kws_threshold = float(self.kws_threshold.get())
        self.config.require_wake_word = bool(self.require_wake_word.get())
        self.config.synthesize_response = bool(self.synthesize_response.get())
        self.config.play_response = bool(self.play_response.get())
        self.pipeline = AssistantPipeline(self.config)


def run_gui(config: AssistantConfig | None = None, record_seconds: float = 5.0) -> None:
    app = AssistantGUI(config=config, record_seconds=record_seconds)
    app.mainloop()
