from pathlib import Path

import mido
import pytest
from pydub import AudioSegment

import midimorph


def test_process_music_raises_when_input_file_missing():
    with pytest.raises(FileNotFoundError):
        midimorph.process_music("/path/to/not-found.mp3")


def test_resolve_input_file_uses_latest_file_in_input_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    old_file = input_dir / "old.mp3"
    new_file = input_dir / "new.wav"
    old_file.write_bytes(b"dummy-old")
    new_file.write_bytes(b"dummy-new")
    old_file.touch()
    new_file.touch()
    monkeypatch.setattr(midimorph, "INPUTS_DIR", input_dir)

    result = midimorph.resolve_input_file()

    assert result == new_file.resolve()


def test_resolve_input_file_raises_when_input_dir_is_empty(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(midimorph, "INPUTS_DIR", input_dir)

    with pytest.raises(FileNotFoundError):
        midimorph.resolve_input_file()


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


def test_transcribe_drums_to_midi_falls_back_to_basic_pitch(tmp_path, monkeypatch):
    stem_wav = tmp_path / "drums.wav"
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    stem_wav.write_bytes(b"dummy")

    monkeypatch.setattr(
        midimorph,
        "transcribe_drums_with_omnizart",
        lambda _stem, _dir: (_ for _ in ()).throw(RuntimeError("omnizart missing")),
    )
    monkeypatch.setattr(
        midimorph,
        "transcribe_to_midi",
        lambda stem, _midi_dir, **_kwargs: tmp_path / f"{stem.stem}_basic_pitch.mid",
    )

    result = midimorph.transcribe_drums_to_midi(stem_wav, midi_dir, drum_transcriber="omnizart", drum_midi_dense=True)

    assert result.name == "drums_basic_pitch.mid"


def test_convert_midi_to_drum_channel_sets_channel_10(tmp_path):
    midi_path = tmp_path / "drums.mid"
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.Message("program_change", program=1, channel=3, time=0))
    track.append(mido.Message("note_on", note=36, velocity=90, channel=3, time=0))
    track.append(mido.Message("note_off", note=36, velocity=0, channel=3, time=120))
    midi_file.save(str(midi_path))

    converted = midimorph.convert_midi_to_drum_channel(midi_path)
    converted_midi = mido.MidiFile(str(converted))
    channel_messages = [
        message for track in converted_midi.tracks for message in track if hasattr(message, "channel")
    ]

    assert converted.exists()
    assert channel_messages
    assert all(message.channel == 9 for message in channel_messages)


def test_augment_drum_midi_with_onsets_appends_overlay_track(tmp_path, monkeypatch):
    midi_path = tmp_path / "drums_ch10.mid"
    midi_file = mido.MidiFile()
    base_track = mido.MidiTrack()
    midi_file.tracks.append(base_track)
    base_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi_file.save(str(midi_path))

    monkeypatch.setattr(
        midimorph,
        "extract_onset_drum_events",
        lambda _stem_wav: [(0.0, 36, 100), (0.5, 38, 90)],
    )

    dense_path = midimorph.augment_drum_midi_with_onsets(midi_path, tmp_path / "drums.wav")
    dense_midi = mido.MidiFile(str(dense_path))
    overlay_track = dense_midi.tracks[-1]
    note_on_messages = [msg for msg in overlay_track if msg.type == "note_on" and msg.velocity > 0]

    assert dense_path.exists()
    assert dense_path.name.endswith("_onset_dense.mid")
    assert len(dense_midi.tracks) == 2
    assert [msg.note for msg in note_on_messages] == [36, 38]


def test_process_music_orchestrates_pipeline(tmp_path, monkeypatch):
    input_file = tmp_path / "song.mp3"
    input_file.write_bytes(b"dummy")

    input_dir = tmp_path / "input"
    workspace_dir = tmp_path / "workspace"
    outputs_dir = tmp_path / "outputs"
    soundfonts_dir = tmp_path / "assets" / "soundfonts"
    soundfonts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(midimorph, "INPUTS_DIR", input_dir)
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
    monkeypatch.setattr(midimorph, "transcribe_drums_to_midi", lambda stem, _midi_dir, _t, _d: tmp_path / f"{stem.stem}.mid")
    monkeypatch.setattr(midimorph, "transcribe_to_midi", lambda stem, _midi_dir, **_kwargs: tmp_path / f"{stem.stem}.mid")
    monkeypatch.setattr(midimorph, "convert_midi_to_drum_channel", lambda midi_path: midi_path)
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


def test_process_music_outputs_drums_only_and_skips_later_phases(tmp_path, monkeypatch):
    input_file = tmp_path / "song.mp3"
    input_file.write_bytes(b"dummy")

    input_dir = tmp_path / "input"
    workspace_dir = tmp_path / "workspace"
    outputs_dir = tmp_path / "outputs"
    soundfonts_dir = tmp_path / "assets" / "soundfonts"
    soundfonts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(midimorph, "INPUTS_DIR", input_dir)
    monkeypatch.setattr(midimorph, "WORKSPACE_DIR", workspace_dir)
    monkeypatch.setattr(midimorph, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(midimorph, "ASSETS_SOUNDFONTS", soundfonts_dir)
    monkeypatch.setattr(midimorph, "welcome_msg", lambda: None)
    monkeypatch.setattr(midimorph.AudioSegment, "from_file", lambda _p: AudioSegment.silent(duration=1200))

    stems_dir = tmp_path / "stems_result"
    stems_dir.mkdir(parents=True, exist_ok=True)
    drums_source = stems_dir / "drums.wav"
    drums_source.write_bytes(b"drums")
    (stems_dir / "piano.wav").write_bytes(b"dummy")
    monkeypatch.setattr(midimorph, "phase1_stem_separation", lambda _in, _ws: stems_dir)

    monkeypatch.setattr(
        midimorph,
        "transcribe_to_midi",
        lambda _stem, _midi_dir, **_kwargs: pytest.fail("drums_only では MIDI 変換は呼ばれない想定"),
    )

    midimorph.process_music(str(input_file), drums_only=True)

    out_file = outputs_dir / "song_drums.wav"
    assert out_file.exists()
    assert out_file.read_bytes() == b"drums"
