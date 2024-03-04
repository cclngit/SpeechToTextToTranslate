import os
import glob
import queue
import threading
import json
import time
import datetime
import sounddevice as sd
import wavio as wv
import numpy as np
import speech_recognition as sr
import ctranslate2
import transformers
import whisper
import torch
from collections import deque
import server


def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")


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


def translate(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device="cpu"):
    try:
        translator = ctranslate2.Translator(translator, device=device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, src_lang=src_lang)
        translations = deque(maxlen=10)

        while True:
            transcription = queue2.get()
            if not transcription in translations:
                temp_file = f"temp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                with open(temp_file, "w") as f:
                    f.write(transcription)

                source = tokenizer.convert_ids_to_tokens(tokenizer.encode(open(temp_file, "r").read()))
                target_prefix = [tgt_lang]
                results = translator.translate_batch([source], target_prefix=[target_prefix])
                target = results[0].hypotheses[0][1:]

                print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
                
                queue3.put(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))

                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
                with open(f"{translations_dir}/{filename}", "w") as f:
                    f.write(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))

                translations.append(transcription)

                os.remove(temp_file)
    
    except Exception as e:
        print(f"Error translating: {e}")


def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")


def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def record_and_save_wrapper(queue1, recordings_dir, freq, duration):
    while True:
        try:
            record_and_save(queue1, recordings_dir, freq, duration)
        except Exception as e:
            print(f"Recording thread failed: {e}")


def transcribe_wrapper(queue1, model, energy_threshold, record_timeout, phrase_timeout):
    while True:
        try:
            transcribe(queue1, model, energy_threshold, record_timeout, phrase_timeout)
        except Exception as e:
            print(f"Transcription thread failed: {e}")


def translate_wrapper(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device):
    while True:
        try:
            translate(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device)
        except Exception as e:
            print(f"Translation thread failed: {e}")


def server_wrapper(queue3):
    while True:
        try:
            server.run_server(queue3)
        except Exception as e:
            print(f"Server thread failed: {e}")


def run_pipeline(config):
    recordings_dir = config["recordings_dir"]
    transcriptions_dir = config["transcriptions_dir"]
    translations_dir = config["translations_dir"]
    num_to_keep = config["num_to_keep"]

    record_config = config["record"]
    freq = record_config["freq"]
    duration = record_config["duration"]

    transcribe_config = config["transcribe"]
    model = transcribe_config["model"]
    energy_threshold = transcribe_config["energy_threshold"]
    record_timeout = transcribe_config["record_timeout"]
    phrase_timeout = transcribe_config["phrase_timeout"]

    translate_config = config["translate"]
    translator = translate_config["translator"]
    tokenizer = translate_config["tokenizer"]
    device = translate_config["device"]
    src_lang = translate_config["src_lang"]
    tgt_lang = translate_config["tgt_lang"]

    os.makedirs(recordings_dir, exist_ok=True)
    os.makedirs(transcriptions_dir, exist_ok=True)
    os.makedirs(translations_dir, exist_ok=True)

    queue1 = queue.Queue()
    queue2 = queue.Queue()
    queue3 = queue.Queue()

    threads = [
        threading.Thread(target=record_and_save_wrapper, args=(queue1, recordings_dir, freq, duration)),
        threading.Thread(target=transcribe_wrapper, args=(queue1, model, energy_threshold,
                                                          record_timeout, phrase_timeout)),
        threading.Thread(target=translate_wrapper, args=(queue2, queue3, translations_dir, src_lang,
                                                         tgt_lang, translator, tokenizer, device)),
        threading.Thread(target=server_wrapper, args=(queue3,))
    ]

    for thread in threads:
        thread.start()

    while True:
        for thread in threads:
            if not thread.is_alive():
                thread.start()
        time.sleep(10)


if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.json')
        config = load_config(config_file)
        run_pipeline(config)
        
    except KeyboardInterrupt:
        print('Pipeline stopped')
        exit(0)