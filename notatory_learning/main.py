import argparse
import time
from pathlib import Path

import gradio as gr
import numpy as np
import openai
from PIL import Image

from notatory_learning.utils.gpt_4o_utils import run_gpt_4o, to_image_content
from notatory_learning.utils.voice_utils import VoiceVoxSpeaker, text_to_wav

client = openai.Client()
speaker = None

prompts = {
    "説明": "次の画像の状況を日本語で説明してください。",
    "ノート作成アドバイス": "画像に写っているノートの内容をもとに、関係する事項について説明してください。",
    "読書アドバイス": "画像に写っている書籍の内容を解説してください。横にいてアドバイスをしているように話してください（返答はいきなりアドバイスから始めてください。）。",
}


def create_frame_descriptor(frame: np.ndarray, model, prompt: str):
    image = Image.fromarray(frame)
    print("Image size:", image.size)
    return run_gpt_4o(
        client,
        [
            {
                "role": "user",
                "content": [
                    prompts[prompt],
                    to_image_content(image, "png"),
                ],
            },
        ],
        model=model,
    )


dt = 10


def process_frame(
    mode: str,
    frame: np.ndarray | None,
    model: str,
    prompt: str,
    last_timestamp: float | None,
):
    if frame is None or frame.size == 0:
        return gr.update(), gr.update(), last_timestamp, gr.update(), gr.update()
    now = time.perf_counter()
    if mode == "playing" or (last_timestamp is not None and now - last_timestamp < dt):
        return gr.update(), gr.update(), last_timestamp, gr.update(), gr.update()
    frame = frame[::-1, ::-1]
    description = create_frame_descriptor(frame, model, prompt)
    voice_path = Path(f"_tmp/voice/{now}.mp3")
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    text_to_wav(description, speaker, voice_path)
    return "playing", frame, now, voice_path, description


# def apply_mode(mode: str):
#     show = gr.update(visible=True)
#     hide = gr.update(visible=False)
#     print(f"apply_mode: {mode}")
#     if mode == "selecting":
#         return show, hide, hide, hide, hide
#     elif mode in ["waiting", "playing"]:
#         return hide, show, show, show, show
#     else:
#         raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Webカメラを使った画像認識アプリ")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ssl_keyfile", type=str, required=True)
    parser.add_argument("--ssl_certfile", type=str, required=True)
    parser.add_argument("--voicevox_url", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    global speaker
    speaker = VoiceVoxSpeaker(
        speaker_id="1",
        url=args.voicevox_url,
    )

    with gr.Blocks() as app:
        last_timestamp = gr.State(value=None)
        mode = gr.State(value="selecting")

        prompt = gr.Dropdown(
            list(prompts.keys()),
            label="プロンプト",
        )
        model = gr.Dropdown(
            ["gpt-4.1", "gpt-4.1-mini"],
            value="gpt-4.1",
            label="モデル",
        )
        audio = gr.Audio(label="処理結果（音声）", autoplay=True)
        text = gr.Markdown(label="説明")
        with gr.Row():
            camera = gr.Image(
                sources=["webcam"],
                streaming=False,
                label="Webカメラ",
                webcam_options=gr.WebcamOptions(
                    mirror=False,
                    constraints={"video": {"width": 1600, "height": 1200}},
                ),
                width=256,
                height=256,
            )
            preview = gr.Image(label="Webカメラプレビュー", width=256, height=256)
        # select_camera = gr.Button("Webカメラを選択")

        camera.stream(
            fn=process_frame,
            inputs=[mode, camera, model, prompt, last_timestamp],
            outputs=[mode, preview, last_timestamp, audio, text],
        )
        # select_camera.click(
        #     fn=lambda: "selecting",
        #     outputs=[mode],
        # )
        audio.stop(
            fn=lambda: "waiting",
            outputs=[mode],
        )
        # mode.change(
        #     fn=apply_mode,
        #     inputs=[mode],
        #     outputs=[camera, select_camera, preview, audio, text],
        #     stream_every=0.01,
        # )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        ssl_verify=False,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
