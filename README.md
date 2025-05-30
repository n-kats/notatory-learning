# Notatory Learning
## 概要
このツールは、カメラに映った内容をもとにアドバイスをしてくれるツールです。音声化には[VOICEVOX](https://voicevox.hiroshiba.jp/)を使用しています。

以下のようなウェブUIを提供します。

<!--![image](docs/images/screenshot.png)-->


## 使い方
### 想定環境
* 新しめのdocker
* cpuアーキテクチャ: x86
* linux（他環境での動作確認無し。wslでもいけるかも。arm/macは怪しい。）

### 事前準備(voicevox-engine)
[voicevox-engine](https://github.com/VOICEVOX/voicevox_engine) を利用します。
以下が実行例です。

#### gpu版の場合（十分なgpu環境設定が必要）
```
gpus all -d --rm \
  -p 50021:50021 \
  --gpus all \
  voicevox/voicevox_engine:nvidia-latest
```

#### cpu版の場合
```
gpus all -d --rm \
  -p 50021:50021 \
  voicevox/voicevox_engine:cpu-latest
```


### 環境変数設定
.env ファイルを .env.example からコピーして環境変数を設定します。

特に、OPENAI_API_KEYの設定が必要です。

### 起動
```
$ bash ./scripts/launch.sh
```

起動したら http://localhost:8100 にアクセスしてください。


<!--
### 操作方法

![image](docs/images/screenshot.png)
-->

## 制限
* 音声は ずんだもん 固定です。