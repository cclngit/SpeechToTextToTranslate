import os
import queue
import threading
import server
import utils
import translate, transcribe

def run_pipeline(config):
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

    queue1 = queue.Queue()
    queue2 = queue.Queue()

    threads = [
        threading.Thread(target=transcribe.transcribe, args=(queue1, model, energy_threshold, record_timeout, phrase_timeout)),
        threading.Thread(target=translate.translate, args=(queue1, queue2, src_lang, tgt_lang, translator, tokenizer, device)),
        threading.Thread(target=server.run_server, args=(queue2,))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.json')
        config = utils.load_config(config_file)
        run_pipeline(config)
        
    except KeyboardInterrupt:
        print('Pipeline stopped')
        exit(0)