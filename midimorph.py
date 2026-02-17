"""
MidiMorph: MP3 を 3 ステムに分離し、全パートをライブラリ音源へ置換するツール。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import mido
from pydub import AudioSegment


PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "input"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_SOUNDFONTS = PROJECT_ROOT / "assets" / "soundfonts"
DEMUCS_MODEL = "htdemucs_6s"
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

ROLE_KEYWORDS = {
    "drums": ["drum", "kit"],
    "piano": ["piano", "keys", "keyboard"],
    "accompaniment": ["accompaniment", "strings", "pad", "orch", "epiano", "synth"],
}


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir and Path(scripts_dir).exists():
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    return env


def _run_cmd(cmd: list[str], error_message: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise RuntimeError(error_message)


def welcome_msg() -> None:
    print(
        """
--- MidiMorph v0.2 ---
[1] Split MP3 to drums/piano/accompaniment...
[2] Transcribe each stem to MIDI...
[3] Render library sounds (SoundFont)...
[4] Mix and export MP3...
"""
    )


def ensure_dirs() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_SOUNDFONTS.mkdir(parents=True, exist_ok=True)


def resolve_input_file(input_path: str | None = None) -> Path:
    if input_path is not None:
        input_file = Path(input_path).expanduser().resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")
        return input_file

    ensure_dirs()
    audio_candidates = sorted(
        [
            path
            for path in INPUTS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not audio_candidates:
        raise FileNotFoundError(
            "入力ファイルが指定されていません。"
            f" {INPUTS_DIR} に mp3/wav 等の音声を置くか、CLI 引数で入力ファイルを指定してください。"
        )
    return audio_candidates[0]


def phase1_stem_separation(input_path: Path, workspace: Path) -> Path:
    print("--- Phase 1: Stem Separation (Demucs 6 stems) ---")
    out_root = workspace / "stems"
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "-o",
        str(out_root),
        str(input_path),
    ]
    _run_cmd(cmd, "Demucs の音源分離に失敗しました。demucs が 6stem モデル対応か確認してください。")

    stems_dir = out_root / DEMUCS_MODEL / input_path.stem
    if not stems_dir.exists():
        raise RuntimeError(f"Demucs の出力が見つかりません: {stems_dir}")
    if not (stems_dir / "drums.wav").exists():
        raise RuntimeError("drums.wav が見つかりません。Demucs 実行結果を確認してください。")
    if not (stems_dir / "piano.wav").exists():
        raise RuntimeError(
            "piano.wav が見つかりません。Demucs 6stem モデル (htdemucs_6s) が必要です。"
        )

    print(f"  -> {stems_dir}")
    return stems_dir


def build_accompaniment_stem(stems_dir: Path, workspace: Path, duration_ms: int) -> Path:
    accompaniment_wav = workspace / "accompaniment_source.wav"
    exclude = {"drums.wav", "piano.wav"}
    source_wavs = sorted([p for p in stems_dir.glob("*.wav") if p.name not in exclude])
    if not source_wavs:
        raise RuntimeError("伴奏用のステムが作れませんでした（drums/piano 以外の stem がありません）。")

    merged = AudioSegment.silent(duration=duration_ms)
    for wav in source_wavs:
        seg = AudioSegment.from_wav(str(wav))
        if len(seg) > duration_ms:
            seg = seg[:duration_ms]
        merged = merged.overlay(seg)
    merged.export(str(accompaniment_wav), format="wav")
    return accompaniment_wav


def transcribe_to_midi(stem_wav: Path, midi_dir: Path) -> Path:
    if not stem_wav.exists():
        raise FileNotFoundError(f"入力ステムが見つかりません: {stem_wav}")

    before = set(midi_dir.glob("*.mid"))
    cmd = ["basic-pitch", str(midi_dir), str(stem_wav)]
    _run_cmd(cmd, f"Basic Pitch の変換に失敗しました: {stem_wav.name}")
    after = set(midi_dir.glob("*.mid"))

    new_midis = sorted(after - before, key=lambda p: p.name)
    if new_midis:
        return new_midis[0]

    # 既存ファイルから stem 名を優先して探す
    preferred = [p for p in after if stem_wav.stem.lower() in p.stem.lower()]
    if preferred:
        return sorted(preferred, key=lambda p: p.name)[0]

    raise RuntimeError(f"MIDI が見つかりませんでした: {stem_wav.name}")


def find_soundfont_for_role(role: str) -> Path | None:
    if not ASSETS_SOUNDFONTS.exists():
        return None

    keywords = ROLE_KEYWORDS.get(role, [role])
    all_sf2 = sorted(ASSETS_SOUNDFONTS.glob("*.sf2"), key=lambda p: p.name.lower())
    for sf2 in all_sf2:
        lower_name = sf2.name.lower()
        if any(keyword in lower_name for keyword in keywords):
            return sf2
    # 役割キーワードで見つからない場合は先頭の sf2 を流用
    if all_sf2:
        return all_sf2[0]
    return None


def synthesize_midi_to_wav(midi_path: Path, sf2_path: Path, output_wav: Path) -> None:
    cmd = [
        "fluidsynth",
        "-ni",
        "-T",
        "wav",
        "-F",
        str(output_wav),
        str(sf2_path),
        str(midi_path),
    ]
    _run_cmd(cmd, f"FluidSynth のレンダリングに失敗しました: {midi_path.name}")


def convert_midi_to_drum_channel(midi_path: Path) -> Path:
    """MIDI のチャンネルメッセージを GM ドラムチャンネル(10ch=9) に統一する。"""
    midi_data = mido.MidiFile(str(midi_path))
    converted_path = midi_path.with_name(f"{midi_path.stem}_drums_ch10.mid")

    for track in midi_data.tracks:
        for message in track:
            if hasattr(message, "channel"):
                message.channel = 9

    midi_data.save(str(converted_path))
    return converted_path


def phase4_mix_and_export(
    drums_wav: Path, piano_wav: Path, accompaniment_wav: Path, output_path: Path, duration_ms: int
) -> None:
    print("--- Phase 4: Final Mix ---")
    mix = AudioSegment.silent(duration=duration_ms)

    for wav_path in [drums_wav, piano_wav, accompaniment_wav]:
        segment = AudioSegment.from_wav(str(wav_path))
        if len(segment) > duration_ms:
            segment = segment[:duration_ms]
        mix = mix.overlay(segment)

    mix.export(str(output_path), format="mp3", bitrate="192k")
    print(f"  -> 出力: {output_path}")


def resolve_sf2(role: str, user_path: str | None = None) -> Path:
    if user_path is not None:
        explicit = Path(user_path).expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"{role} 用 sf2 が見つかりません: {explicit}")
        return explicit

    auto = find_soundfont_for_role(role)
    if auto is not None:
        return auto
    raise FileNotFoundError(
        f"{role} 用 sf2 が見つかりません。"
        " assets/soundfonts に .sf2 を置くか、"
        f" --{role}-sf2 で明示指定してください。"
        " README の「SoundFont（.sf2）の準備」を参照してください。"
    )


def process_music(
    input_path: str | None = None,
    output_path: str | None = None,
    drums_sf2_path: str | None = None,
    piano_sf2_path: str | None = None,
    accompaniment_sf2_path: str | None = None,
    drums_only: bool = False,
) -> None:
    ensure_dirs()
    input_file = resolve_input_file(input_path)
    welcome_msg()

    track_name = input_file.stem
    workspace = WORKSPACE_DIR / track_name
    workspace.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_file = OUTPUTS_DIR / (f"{track_name}_drums.wav" if drums_only else f"{track_name}_morphed.mp3")
    else:
        output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    track_duration_ms = len(AudioSegment.from_file(str(input_file)))

    stems_dir = phase1_stem_separation(input_file, workspace)
    drums_source = stems_dir / "drums.wav"
    if drums_only:
        shutil.copy2(drums_source, output_file)
        print(f"\nSuccess! '{input_file.name}' -> {output_file} (drums only)")
        return

    piano_source = stems_dir / "piano.wav"
    accompaniment_source = build_accompaniment_stem(stems_dir, workspace, track_duration_ms)

    print("--- Phase 2: Audio to MIDI ---")
    midi_dir = workspace / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    drums_midi = transcribe_to_midi(drums_source, midi_dir)
    piano_midi = transcribe_to_midi(piano_source, midi_dir)
    accompaniment_midi = transcribe_to_midi(accompaniment_source, midi_dir)
    drums_midi = convert_midi_to_drum_channel(drums_midi)

    print("--- Phase 3: Library Sound Rendering ---")
    drums_sf2 = resolve_sf2("drums", drums_sf2_path)
    piano_sf2 = resolve_sf2("piano", piano_sf2_path)
    accompaniment_sf2 = resolve_sf2("accompaniment", accompaniment_sf2_path)

    synth_dir = workspace / "synths"
    synth_dir.mkdir(parents=True, exist_ok=True)
    drums_wav = synth_dir / "drums.wav"
    piano_wav = synth_dir / "piano.wav"
    accompaniment_wav = synth_dir / "accompaniment.wav"
    synthesize_midi_to_wav(drums_midi, drums_sf2, drums_wav)
    synthesize_midi_to_wav(piano_midi, piano_sf2, piano_wav)
    synthesize_midi_to_wav(accompaniment_midi, accompaniment_sf2, accompaniment_wav)

    phase4_mix_and_export(drums_wav, piano_wav, accompaniment_wav, output_file, track_duration_ms)
    print(f"\nSuccess! '{input_file.name}' -> {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MidiMorph: MP3 -> drums/piano/accompaniment morph tool")
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to input MP3/WAV file (省略時は input/ ディレクトリ内の最新音声を利用)",
    )
    parser.add_argument("-o", "--output", help="Output MP3 path (default: outputs/<name>_morphed.mp3)")
    parser.add_argument("--drums-sf2", help="Drums SoundFont path (.sf2)")
    parser.add_argument("--piano-sf2", help="Piano SoundFont path (.sf2)")
    parser.add_argument("--accompaniment-sf2", help="Accompaniment SoundFont path (.sf2)")
    parser.add_argument("--drums-only", action="store_true", help="Demucs で抽出した drums.wav のみを出力する")
    args = parser.parse_args()

    try:
        process_music(
            args.input,
            args.output,
            drums_sf2_path=args.drums_sf2,
            piano_sf2_path=args.piano_sf2,
            accompaniment_sf2_path=args.accompaniment_sf2,
            drums_only=args.drums_only,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
