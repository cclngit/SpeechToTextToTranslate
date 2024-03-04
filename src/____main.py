import os
import glob
import queue
import threading
import json
import time
import sounddevice as sd
import wavio as wv
import datetime
import transformers
import ctranslate2
from collections import deque
import librosa
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


def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")


def transcribe(queue1, queue2, processor, model_path):
    try:
        processor = transformers.WhisperProcessor.from_pretrained(processor)
        model = ctranslate2.models.Whisper(model_path)
        transcribed = deque(maxlen=10)

        while True:
            recording = queue1.get()
            if not recording in transcribed:
                audio, _ = librosa.load(recording, sr=16000, mono=True)
                inputs = processor(audio, return_tensors="np", sampling_rate=16000)
                features = ctranslate2.StorageView.from_array(inputs.input_features)

                results = model.detect_language(features)
                language, probability = results[0][0]
                print("Detected language %s with probability %f" % (language, probability))

                prompt = processor.tokenizer.convert_tokens_to_ids([
                    "<|startoftranscript|>",
                    "<|en|>",
                    "<|transcribe|>",
                    "<|notimestamps|>",
                ])

                results = model.generate(features, [prompt])
                transcription = processor.decode(results[0].sequences_ids[0])
                print(transcription)

                queue2.put(transcription)
                transcribed.append(recording)
                
    except Exception as e:
        print(f"Error transcribing: {e}")


def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")


def translate(queue1,queue2, translations_dir, src_lang, tgt_lang, translator, tokenizer, device="cpu"):
    
    try:
        translator = ctranslate2.Translator(translator, device=device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, src_lang=src_lang)
        translations = deque(maxlen=10)

        while True:
            transcription = queue1.get()
            if not transcription in translations:
                temp_file = f"temp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                with open(temp_file, "w") as f:
                    f.write(transcription)

                source = tokenizer.convert_ids_to_tokens(tokenizer.encode(open(temp_file, "r").read()))
                target_prefix = [tgt_lang]
                results = translator.translate_batch([source], target_prefix=[target_prefix])
                target = results[0].hypotheses[0][1:]

                print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
                
                queue2.put(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))

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

def run_pipeline(config):
    recordings_dir = config["recordings_dir"]
    transcriptions_dir = config["transcriptions_dir"]
    translations_dir = config["translations_dir"]
    num_to_keep = config["num_to_keep"]

    record_config = config["record"]
    freq = record_config["freq"]
    duration = record_config["duration"]

    transcribe_config = config["transcribe"]
    model_path = transcribe_config["model_path"]
    processor = transcribe_config["processor"]

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
        threading.Thread(target=record_and_save, args=(queue1, recordings_dir, freq, duration)),
        threading.Thread(target=transcribe, args=(queue1, queue2, processor, model_path)),
        threading.Thread(target=translate, args=(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device)),
        threading.Thread(target=server.run_server, args=(queue3,))
    ]

    for thread in threads:
        thread.start()

    # Monitor threads and restart if they fail
    while True:
        for thread in threads:
            if not thread.is_alive():
                thread.start()
        time.sleep(10)  # Check every 10 seconds

if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.json')
        config = load_config(config_file)
        run_pipeline(config)
        
    except KeyboardInterrupt:
        print('Pipeline stopped')
        exit(0)
