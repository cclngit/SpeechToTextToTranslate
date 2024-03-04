import os
import glob
import time
import datetime
import sounddevice as sd
import wavio as wv
import numpy as np
import speech_recognition as sr
import whisper
import torch
from collections import deque
import queue
import json
import threading


def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")

def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def record_and_save(queue1, recordings_dir, freq, duration):
    try:
        while True:
            recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
            sd.wait()
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
            wv.write(f"{recordings_dir}/{filename}", recording, freq, sampwidth=2)
            print(f"Recording saved as {filename}")
            queue1.put(f"{recordings_dir}/{filename}")
            delete_old_files(f"{recordings_dir}/*.wav")
    
    except Exception as e:
        print(f"Error recording: {e}")


def transcribe(queue1, model, energy_threshold, record_timeout, phrase_timeout):
    try:
        audio_model = whisper.load_model(model)

        phrase_time = None
        transcription = deque(maxlen=10)
        recorder = sr.Recognizer()
        recorder.energy_threshold = energy_threshold
        recorder.dynamic_energy_threshold = False

        with sr.Microphone(sample_rate=16000) as source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            queue1.put(data)

        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        while True:
            now = datetime.datetime.now(datetime.UTC)
            if not queue1.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > datetime.timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                audio_data = b''.join(queue1.queue)
                queue1.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                transcription_text = result['text'].strip()

                if phrase_complete:
                    transcription.append(transcription_text)
                else:
                    transcription[-1] = transcription_text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
                time.sleep(0.25)
    
    except Exception as e:
        print(f"Error transcribing: {e}")

def pipeline(config):
    queue1 = queue.Queue()
    recordings_dir = config['recordings_dir']
    freq = config['freq']
    duration = config['duration']
    model = config['model']
    energy_threshold = config['energy_threshold']
    record_timeout = config['record_timeout']
    phrase_timeout = config['phrase_timeout']
    delete_old_files(recordings_dir)
    record_thread = threading.Thread(target=record_and_save, args=(queue1, recordings_dir, freq, duration))
