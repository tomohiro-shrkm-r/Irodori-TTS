# Server API (`server.py`)

`server.py` は Gradio UI と REST API を 1 プロセスで提供する。UI と API はキャッシュ済みの runtime を共有するため、モデルは 1 度だけロードされる。

| パス | 内容 |
|---|---|
| `/` | Gradio UI（`gradio_app.py` と同じ） |
| `GET /api/health` | 稼働確認 |
| `POST /api/tts` | JSON で合成 → `audio/wav` |
| `POST /api/tts/ref` | multipart で参照音声付き合成 → `audio/wav` |
| `/docs` | Swagger UI（OpenAPI から自動生成。ブラウザで試し打ち可） |

## 起動

```bash
uv run --no-sync python server.py --host 0.0.0.0 --port 7860 --preload
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--host` | `127.0.0.1` | 他端末から使う場合は `0.0.0.0` |
| `--port` | `7860` | |
| `--checkpoint` | UI と同じ | API が使うモデル（`.pt` / `.safetensors` / HF repo id） |
| `--model-device` / `--codec-device` | 利用可能な最初のデバイス | 例: `cuda`, `cpu` |
| `--model-precision` / `--codec-precision` | `fp32` | `fp32` / `bf16`（GPU のみ） |
| `--preload` | off | 起動時にモデルをロードする。無しだと初回リクエストでロード |
| `--api-key KEY` | なし | `/api/*` に認証を要求する（UI は対象外） |
| `--speaker-dir DIR` | なし | 指定時のみ `speaker_embedding` を受け付ける |
| `--lora-dir DIR` | なし | 指定時のみ `lora_adapter` を受け付ける |
| `--no-ui` | off | API のみ提供する |

UI 側で別のチェックポイント・デバイス・精度をロードすると、次の API リクエストで `--checkpoint` 等の設定に再ロードされる（runtime キャッシュは 1 つ）。

## 認証

`--api-key` 指定時は、次のいずれかのヘッダが必要。無い・不一致の場合は `401`。

```
X-API-Key: KEY
Authorization: Bearer KEY
```

## `GET /api/health`

```json
{"status": "ok", "checkpoint": "/path/to/model.safetensors"}
```

## `POST /api/tts`

参照音声なし（または `speaker_embedding` 指定）で合成する。

```bash
curl -X POST http://HOST:7860/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは、今日はいい天気ですね。", "seed": 42}' \
  -o out.wav
```

caption を指定すると声質・スタイルを制御できる（caption 対応チェックポイントのみ）。

```bash
curl -X POST http://HOST:7860/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "あははっ、それ本当に言ってるの？", "caption": "余裕のある大人の男性。くだけた雰囲気で楽しそうに話している。"}' \
  -o out.wav
```

### パラメータ

`text` 以外は省略可。既定値は Gradio UI と同じ。各パラメータの意味は [parameters.md](parameters.md) を参照。

| 名前 | 型 | 既定値 | 備考 |
|---|---|---|---|
| `text` | string | （必須） | 1 文字以上 |
| `caption` | string \| null | `null` | 声質・スタイルを指定する VoiceDesign キャプション。caption 対応チェックポイントが必要 |
| `max_caption_len` | int \| null | `null` | `null` はチェックポイントの既定値。1 以上 |
| `seed` | int \| null | `null` | `null` はランダム。使われた値はレスポンスヘッダ `X-Seed` |
| `num_steps` | int \| null | `null` | `null` はチェックポイントの推奨値。1 以上 |
| `seconds` | float \| null | `null` | `null` は自動推定 |
| `duration_scale` | float | `1.0` | 自動推定した長さへの倍率 |
| `t_schedule_mode` | `"linear"` \| `"sway"` | `"linear"` | |
| `sway_coeff` | float | `-1.0` | `sway` 時のみ有効 |
| `cfg_guidance_mode` | `"independent"` \| `"joint"` \| `"alternating"` | `"independent"` | |
| `cfg_scale_text` | float | `3.0` | |
| `cfg_scale_caption` | float | `3.0` | |
| `cfg_scale_speaker` | float | `5.0` | |
| `cfg_scale` | float \| null | `null` | 指定すると text/caption/speaker 全てを上書き |
| `cfg_min_t` | float | `0.5` | |
| `cfg_max_t` | float | `1.0` | |
| `context_kv_cache` | bool | `true` | |
| `truncation_factor` | float \| null | `null` | |
| `rescale_k` | float \| null | `null` | |
| `rescale_sigma` | float \| null | `null` | |
| `speaker_kv_scale` | float \| null | `null` | |
| `speaker_kv_min_t` | float \| null | `0.9` | |
| `speaker_kv_max_layers` | int \| null | `null` | |
| `speaker_embedding` | string \| null | `null` | `--speaker-dir` 直下の `.speaker.safetensors` のファイル名 |
| `lora_adapter` | string \| null | `null` | `--lora-dir` 直下のアダプタディレクトリ名 |
| `length_check` | bool | `false` | 2 パスで長さを調整する |

`speaker_embedding` / `lora_adapter` にはパスではなく名前を渡す。`../` や絶対パス、HF repo id は拒否される。

### レスポンス

- `200`: `audio/wav`（16-bit PCM、モノラル、48 kHz）
  - `X-Seed`: 使われた seed
  - `X-Total-To-Decode-Sec`: 生成〜デコードの所要秒数
- `400`: 合成時のエラー（`{"detail": "..."}`）
- `401`: API キー不一致
- `422`: パラメータの型・範囲エラー

## `POST /api/tts/ref`

アップロードした参照音声の声で合成する。`multipart/form-data` で、`/api/tts` のパラメータをフォームフィールドとして送り、`ref_audio` にファイルを付ける。`ref_audio` は複数指定でき、指定順に連結して参照に使われる。

```bash
curl -X POST http://HOST:7860/api/tts/ref \
  -F "text=こんにちは、今日はいい天気ですね。" \
  -F "seed=42" \
  -F "ref_audio=@voice1.wav" \
  -F "ref_audio=@voice2.wav" \
  -o out.wav
```

`ref_audio` と `speaker_embedding` は同時に指定できない（`400`）。

## クライアント例

### Python

```python
import requests

resp = requests.post(
    "http://HOST:7860/api/tts",
    json={"text": "こんにちは", "seed": 42},
    timeout=300,
)
resp.raise_for_status()
open("out.wav", "wb").write(resp.content)
print("seed:", resp.headers["X-Seed"])

with open("voice.wav", "rb") as f:
    resp = requests.post(
        "http://HOST:7860/api/tts/ref",
        data={"text": "こんにちは"},
        files=[("ref_audio", f)],
        timeout=300,
    )
```

## 注意

- リクエストは 1 件ずつ順に処理される（runtime 内部でロック）。同時に送ると後続は待たされる。
- Windows の Git Bash で curl の引数に日本語を直接書くと、Shift_JIS で送られて `400` になる（`-d` も `-F` も同じ）。JSON は UTF-8 のファイルを `--data-binary @req.json` で、multipart は `-F "text=<text.txt"` のように UTF-8 のファイルから値を読ませる。
- `--api-key` は Gradio UI を保護しない。UI からは任意のチェックポイント・LoRA パスを指定できるため、信頼できないネットワークに公開しない。
