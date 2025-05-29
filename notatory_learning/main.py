import cv2
from pathlib import Path
import wave
import os
import time
import gradio as gr
import numpy as np
from PIL import Image
from notatory_learning.utils.gpt_4o_utils import run_gpt_4o, to_image_content
from notatory_learning.utils.voice_utils import text_to_wav, VoiceVoxSpeaker

import openai
client = openai.Client()
speaker = VoiceVoxSpeaker(
    speaker_id="1",
    url="http://localhost:50021",
)

prompts = {
    "説明": "次の画像の状況を日本語で説明してください。",
    "ノート作成アドバイス": "画像に写っているノートの内容をもとに、関係する事項について説明してください。",
    "読書アドバイス": "画像に写っている書籍の内容を解説してください。横にいてアドバイスをしているように話してください（返答はいきなりアドバイスから始めてください。）。",
}


def create_frame_descriptor(frame: np.ndarray, prompt: str):
    image = Image.fromarray(frame)
    print("Image size:", image.size)
    return run_gpt_4o(
        client, [
            {
                "role": "user",
                "content": [
                        prompts[prompt],
                        to_image_content(image, "png"),
                ],
            },
        ],
        model="gpt-4.1",
    )


dt = 10


def process_frame(
    mode: str,
    frame: np.ndarray | None,
    prompt: str,
    last_timestamp: float | None,
):
    if frame is None or frame.size == 0:
        return gr.update(), gr.update(), last_timestamp, gr.update(), gr.update()
    now = time.perf_counter()
    if mode == "playing" or (last_timestamp is not None and now - last_timestamp < dt):
        return gr.update(), gr.update(), last_timestamp, gr.update(), gr.update()
    frame = frame[::-1, ::-1]
    description = create_frame_descriptor(frame, prompt)
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


with gr.Blocks() as app:
    last_timestamp = gr.State(value=None)
    mode = gr.State(value="selecting")

    gr.Markdown("## １分ごとにカメラ映像をキャプチャして音声を再生")
    prompt = gr.Dropdown(
        list(prompts.keys()), label="プロンプト",
    )
    audio = gr.Audio(label="処理結果（音声）", autoplay=True)
    text = gr.Markdown(label="説明")
    with gr.Row():
        camera = gr.Image(
            sources=["webcam"], streaming=True,
            label="Webカメラ", webcam_options=gr.WebcamOptions(
                mirror=False,
            ),
            width=256, height=256,
        )
        preview = gr.Image(label="Webカメラプレビュー",
                           width=256, height=256)
    # select_camera = gr.Button("Webカメラを選択")

    camera.stream(
        fn=process_frame,
        inputs=[mode, camera, prompt, last_timestamp],
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
    server_name="192.168.11.2",
    server_port=33333,
    ssl_verify=False,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem",
)
