import asyncio
import json
import os
import uuid

from pydub import AudioSegment
from websockets.legacy.server import Serve, WebSocketServerProtocol

from tts_interface import TTS


def get_audio(data):
    sr, wav = tts.get_voice(data["text"], data["lang"])
    audio = AudioSegment(
        data=wav.tobytes(), sample_width=wav.dtype.itemsize, frame_rate=sr, channels=1
    )
    filename = os.path.realpath(f"wav/{uuid.uuid4()}.wav")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    audio.export(filename, format="wav")
    return filename


async def serve(websocket: WebSocketServerProtocol, _: str):
    while True:
        data = json.loads(await websocket.recv())
        if data["type"] == "get":
            await websocket.send(get_audio(data))
        elif data["type"] == "model":
            tts.set_model_from_dataset(data["data"])
            await websocket.send("success")
        elif data["type"] == "stop":
            break
        else:
            await websocket.send("error")


if __name__ == "__main__":
    tts = TTS()
    start_server = Serve(serve, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
