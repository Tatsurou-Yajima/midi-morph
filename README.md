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

---

## 使い方

### 基本的な実行（自動検出）

```bash
python midimorph.py path/to/あなたの曲.mp3
```

- 入力: MP3 または WAV のパス
- 出力: 自動で `outputs/<曲名>_morphed.mp3` に保存されます

### 出力先を指定する

```bash
python midimorph.py path/to/曲.mp3 -o path/to/結果.mp3
```

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
| `input` | 変換したい音声ファイルのパス（必須） |
| `-o`, `--output` | 出力 MP3 のパス（省略時は `outputs/<曲名>_morphed.mp3`） |
| `--drums-sf2` | drums 用 SoundFont のパス |
| `--piano-sf2` | piano 用 SoundFont のパス |
| `--accompaniment-sf2` | accompaniment 用 SoundFont のパス |

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
