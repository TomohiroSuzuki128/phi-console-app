# PhiConsoleApp
Phi モデルをローカルで検証できるコンソールアプリのサンプルです。
- ライブラリは ONNX です
- C# で書かれています

## 目的

- Phi3, Phi3.5 をローカル CPU で動かす検証の用途
- Phi3, Phi3.5 でエッジ AI アプリを作る前に Phi3, Phi3.5 で求めている精度が出るか確認できる
- Phi3, Phi3.5 でエッジ AI とりあえず遊んでみる

## 特長

### 設定ファイルは json

以下の項目を設定できます
- ローカルに保存したモデルの Path
- 翻訳の ON/OFF
- RAG の ON/OFF

### プロンプトを一度英語に翻訳して問い合わせ、結果を再度日本語にできる

Phi シリーズは英語に最適化されているため、プロンプトを一度英語にして問い合わせ結果を再度日本語にすることで精度を高められます。翻訳の ON/OFF は設定ファイルのオプションで指定します。

### RAG に対応

md, txt で付加情報を与えることでで RAG に対応しています。特に翻訳した時の固有名詞を与えることで、精度を高められます。

ベクトル検索は Build5Nines 氏の [SharpVector](https://github.com/Build5Nines/SharpVector) を使用しています。

#### 処理フロー
1. システムプロンプトを英語に翻訳する
1. ユーザープロンプトを英語に翻訳する
1. 英語にしたプロンプトで問い合わせ
1. レスポンスでベクトルデータベースに問合せし、RAG のデータ取得
1. RAG のデータを付加して日本語に翻訳

## 設定ファイルのフォーマット

```json:settings.json
{
  "modelPhi35Min128k": "<Your model path>",
  "modelPhi3Med4k": "<Your model path>",
  "modelPhi3Med128k": "<Your model path>",
  "modelPhi3Min4k": "<Your model path>",
  "modelPhi3Min128k": "<Your model path>",
  "modelPhi4Unofficial": "<Your model path>",
  "isTranslate": "<true or false>",
  "isUsingRag": "<true or false>",
  "systemPrompt": "<Your system prompt>",
  "userPrompt": "<Your user prompt>",
  "additionalDocumentsPath": "<Your documents path>" // RAG 用ファイルの Path
}
```

## モデルのダウンロード
Hugging Face からダウンロードします。
コマンドについては付属の download_model_cmd.txt を参考にしてください
