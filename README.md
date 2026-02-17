# MidiMorph

MP3/WAV を読み込み、`drums / piano / accompaniment` の 3 パートに分離し、  
**3パートすべてを SoundFont 音源に差し替えて** MP3 として書き出す CLI ツールです。

---

## 必要な環境

- **macOS**
- **Python 3.9 以上**
- **FFmpeg**（音声読み書き）
- **FluidSynth**（必須）

---

## セットアップ

### 1. システムツールのインストール

```bash
brew install ffmpeg
```

```bash
brew install fluidsynth
```

### 2. リポジトリのクローンと仮想環境

```bash
cd midi-morph
python3 -m venv venv
source venv/bin/activate
```

### 3. Python パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. SoundFont（.sf2）の準備

Phase 3 で MIDI を音声に変換するために、**drums / piano / accompaniment** 用の SoundFont が必要です。

1. 次のいずれかから無料の .sf2 を入手し、`assets/soundfonts/` に置いてください。
   - [GeneralUser GS](https://schristiancollins.com/generaluser.php)（総合）
   - [FluidSynth 公式のテスト用 sf2](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont) など
2. ファイル名に次のキーワードが含まれると自動で役割が割り当てられます。
   - **drums**: `drum`, `kit` を含むファイル名
   - **piano**: `piano`, `keys`, `keyboard` を含むファイル名
   - **accompaniment**: `accompaniment`, `strings`, `pad`, `orch`, `epiano`, `synth` を含むファイル名
3. 1 つしか .sf2 を置かない場合は、その 1 つが全パートに使われます。

例（3 つ置く場合）:
```text
assets/soundfonts/
  drums.sf2
  piano.sf2
  accompaniment.sf2
```

---

## 使い方

### 基本的な実行（`input/` から自動検出）

```bash
python midimorph.py
```

- 入力: `input/` ディレクトリ内の最新音声ファイル（mp3/wav/m4a/flac など）
- 出力: 自動で `outputs/<曲名>_morphed.mp3` に保存されます

音声ファイルを明示指定したい場合は、従来どおり引数で指定できます。

```bash
python midimorph.py path/to/あなたの曲.mp3
```

`drums` パートは MIDI 変換後に自動で GM ドラムチャンネル（10ch）へ補正してからレンダリングします。

### 出力先を指定する

```bash
python midimorph.py -o path/to/結果.mp3
```

### drums だけ抽出して出力する

```bash
python midimorph.py --drums-only
```

- 出力: `outputs/<曲名>_drums.wav`
- `-o` を使えば任意の出力パスに変更できます。

`assets/soundfonts/` からファイル名で自動検出します。

- `drum` / `kit` を含む `.sf2` → drums
- `piano` / `keys` / `keyboard` を含む `.sf2` → piano
- `accompaniment` / `strings` / `pad` / `orch` / `epiano` / `synth` を含む `.sf2` → accompaniment

### sf2 を明示指定して実行

```bash
python midimorph.py input.mp3 \
  --drums-sf2 assets/soundfonts/drums.sf2 \
  --piano-sf2 assets/soundfonts/piano.sf2 \
  --accompaniment-sf2 assets/soundfonts/strings.sf2 \
  -o outputs/result.mp3
```

### オプション一覧

| 引数 | 説明 |
|------|------|
| `input` | 変換したい音声ファイルのパス（省略時は `input/` の最新音声を使用） |
| `-o`, `--output` | 出力 MP3 のパス（省略時は `outputs/<曲名>_morphed.mp3`） |
| `--drums-sf2` | drums 用 SoundFont のパス |
| `--piano-sf2` | piano 用 SoundFont のパス |
| `--accompaniment-sf2` | accompaniment 用 SoundFont のパス |
| `--drums-only` | Demucs の `drums.wav` のみを出力（Phase 2 以降をスキップ） |

---

## 処理の流れ

1. **Phase 1** … Demucs (`htdemucs_6s`) で `drums/piano/...` に分離
2. **Phase 2** … `drums`, `piano`, `accompaniment` をそれぞれ Basic Pitch で MIDI 化
3. **Phase 3** … 3つの MIDI をそれぞれ FluidSynth + SoundFont で WAV 化
4. **Phase 4** … 3つの WAV をミックスして MP3 出力

1 曲あたり数分かかることがあります。

---

## ディレクトリ構造

実行時に次のディレクトリが自動作成されます。

```
midi-morph/
├── midimorph.py          # メインスクリプト
├── requirements.txt
├── input/                # 変換したい音声ファイルを置く（未指定実行時はここから最新を読む）
├── assets/
│   └── soundfonts/       # ここに .sf2 を置くと新音色で再生される
├── outputs/              # 変換後の MP3（ここに出力）
└── workspace/            # 分離音・MIDI 等の一時ファイル（曲名ごとにサブディレクトリ）
```

### SoundFont 配置

`assets/soundfonts/` に 3種類の `.sf2` を配置してください（または CLI 引数で明示指定）。

- drums 用
- piano 用
- accompaniment 用

---

## トラブルシューティング

- **「basic-pitch が見つからない」**  
  同じ仮想環境（`source venv/bin/activate` した状態）で `python midimorph.py` を実行してください。システムの Python だけで実行していると basic-pitch が PATH に無い場合があります。

- **「piano.wav が見つからない」**  
  `htdemucs_6s` の利用が必要です。Demucs のバージョンとモデル取得状態を確認してください。

- **「sf2 が見つからない」**  
  `assets/soundfonts/` に対象の `.sf2` を配置するか、`--drums-sf2` などで明示指定してください。

- **処理が重い・遅い**  
  Demucs と Basic Pitch は GPU が使えれば高速化されます。CPU のみでも動作しますが、1 曲あたり数分かかることがあります。
