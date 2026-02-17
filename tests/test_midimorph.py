from pathlib import Path

import pytest
from pydub import AudioSegment

import midimorph


def test_process_music_raises_when_input_file_missing():
    with pytest.raises(FileNotFoundError):
        midimorph.process_music("/path/to/not-found.mp3")


def test_find_soundfont_for_role(tmp_path, monkeypatch):
    soundfonts_dir = tmp_path / "assets" / "soundfonts"
    soundfonts_dir.mkdir(parents=True, exist_ok=True)
    (soundfonts_dir / "my_piano.sf2").write_text("dummy", encoding="utf-8")
    (soundfonts_dir / "great_drum_kit.sf2").write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(midimorph, "ASSETS_SOUNDFONTS", soundfonts_dir)

    assert midimorph.find_soundfont_for_role("piano").name == "my_piano.sf2"
    assert midimorph.find_soundfont_for_role("drums").name == "great_drum_kit.sf2"


def test_build_accompaniment_stem_merges_non_drum_piano(tmp_path):
    stems_dir = tmp_path / "stems"
    workspace = tmp_path / "workspace"
    stems_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    AudioSegment.silent(duration=500).export(str(stems_dir / "bass.wav"), format="wav")
    AudioSegment.silent(duration=500).export(str(stems_dir / "other.wav"), format="wav")
    AudioSegment.silent(duration=500).export(str(stems_dir / "drums.wav"), format="wav")
    AudioSegment.silent(duration=500).export(str(stems_dir / "piano.wav"), format="wav")

    accompaniment_wav = midimorph.build_accompaniment_stem(stems_dir, workspace, 500)
    assert accompaniment_wav.exists()
    assert accompaniment_wav.name == "accompaniment_source.wav"


def test_transcribe_to_midi_picks_new_file(tmp_path, monkeypatch):
    stem_wav = tmp_path / "piano.wav"
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    stem_wav.write_bytes(b"dummy")

    class DummyResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, env):  # noqa: ARG001
        (midi_dir / "piano_basic_pitch.mid").write_bytes(b"dummy")
        return DummyResult()

    monkeypatch.setattr(midimorph.subprocess, "run", fake_run)
    result = midimorph.transcribe_to_midi(stem_wav, midi_dir)
    assert result.name == "piano_basic_pitch.mid"


def test_process_music_orchestrates_pipeline(tmp_path, monkeypatch):
    input_file = tmp_path / "song.mp3"
    input_file.write_bytes(b"dummy")

    workspace_dir = tmp_path / "workspace"
    outputs_dir = tmp_path / "outputs"
    soundfonts_dir = tmp_path / "assets" / "soundfonts"
    soundfonts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(midimorph, "WORKSPACE_DIR", workspace_dir)
    monkeypatch.setattr(midimorph, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(midimorph, "ASSETS_SOUNDFONTS", soundfonts_dir)
    monkeypatch.setattr(midimorph, "welcome_msg", lambda: None)
    monkeypatch.setattr(midimorph.AudioSegment, "from_file", lambda _p: AudioSegment.silent(duration=1200))

    stems_dir = tmp_path / "stems_result"
    stems_dir.mkdir(parents=True, exist_ok=True)
    (stems_dir / "drums.wav").write_bytes(b"dummy")
    (stems_dir / "piano.wav").write_bytes(b"dummy")
    (stems_dir / "bass.wav").write_bytes(b"dummy")
    (stems_dir / "other.wav").write_bytes(b"dummy")

    monkeypatch.setattr(midimorph, "phase1_stem_separation", lambda _in, _ws: stems_dir)
    monkeypatch.setattr(
        midimorph,
        "build_accompaniment_stem",
        lambda _stems, _ws, _dur: tmp_path / "accompaniment_source.wav",
    )
    monkeypatch.setattr(midimorph, "transcribe_to_midi", lambda stem, _midi_dir: tmp_path / f"{stem.stem}.mid")
    monkeypatch.setattr(midimorph, "resolve_sf2", lambda _role, _path=None: tmp_path / "dummy.sf2")
    monkeypatch.setattr(midimorph, "synthesize_midi_to_wav", lambda _midi, _sf2, out: out.write_bytes(b"dummy"))

    called = {}

    def fake_mix(drums, piano, accompaniment, output, duration):
        called["drums"] = drums
        called["piano"] = piano
        called["accompaniment"] = accompaniment
        called["output"] = output
        called["duration"] = duration

    monkeypatch.setattr(midimorph, "phase4_mix_and_export", fake_mix)
    midimorph.process_music(str(input_file))

    assert called["duration"] == 1200
    assert called["output"] == outputs_dir / "song_morphed.mp3"
