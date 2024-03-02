import os
import glob
import queue
import threading
import record_and_save
import translate
import transcribe
import json
import server

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
        threading.Thread(target=record_and_save.record_and_save, args=(queue1, recordings_dir, freq, duration)),
        threading.Thread(target=transcribe.transcribe, args=(queue1, queue2, processor, model_path)),
        threading.Thread(target=translate.translate, args=(queue2, queue3, translations_dir, src_lang, tgt_lang, translator, tokenizer, device)),
        threading.Thread(target=server.run_server, args=(queue3,))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.json')
        config = load_config(config_file)
        run_pipeline(config)
        
    except KeyboardInterrupt:
        print('Pipeline stopped')
        exit(0)
