import base64
from threading import Lock, Thread
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
import cv2
import openai
from mss import mss
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()


class ScreenStream:
    def __init__(self):
        self.running = False
        self.lock = Lock()
        self.frame = None

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        with mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            while self.running:
                screenshot = sct.grab(monitor)
                
                self.lock.acquire()
                self.frame = np.array(screenshot)
                self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy() if self.frame is not None else np.zeros((1080, 1920, 4), dtype=np.uint8)
        self.lock.release()

        # Convert from BGRA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions about what's on their screen.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


screen_stream = ScreenStream().start()

# Choose which model to use:
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
model = ChatOpenAI(model="gpt-4o")
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, screen_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)
try:
    while True:
        cv2.imshow("screen", screen_stream.read())
        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break
except KeyboardInterrupt:
    pass

screen_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)