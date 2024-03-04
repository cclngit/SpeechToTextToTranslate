import os
import queue
import threading
import record_and_save
import translate
import transcribe
import utils
import server
import time

def record_and_save_wrapper(queue1, recordings_dir, freq, duration):
    while True:
        try:
            record_and_save.record_and_save(queue1, recordings_dir, freq, duration)
        except Exception as e:
            print(f"Recording thread failed: {e}")

def transcribe_wrapper(queue1, queue2, processor, model_path):
    while True:
        try:
            transcribe.transcribe(queue1, queue2, processor, model_path)
        except Exception as e:
            print(f"Transcription thread failed: {e}")

def translate_wrapper(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device):
    while True:
        try:
            translate.translate(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device)
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
        threading.Thread(target=record_and_save_wrapper, args=(queue1, recordings_dir, freq, duration)),
        threading.Thread(target=transcribe_wrapper, args=(queue1, queue2, processor, model_path)),
        threading.Thread(target=translate_wrapper, args=(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device)),
        threading.Thread(target=server_wrapper, args=(queue3,))
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
        config = utils.load_config(config_file)
        run_pipeline(config)
        
    except KeyboardInterrupt:
        print('Pipeline stopped')
        exit(0)
