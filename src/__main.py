import sounddevice as sd
import wavio as wv
import datetime
import os
import glob
import librosa
import transformers
import ctranslate2
from collections import deque
import queue
import threading


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
                    language,
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


def translate(queue2, translations_dir, src_lang, tgt_lang, translator, tokenizer, device="cpu"):
    
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

                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
                with open(f"{translations_dir}/{filename}", "w") as f:
                    f.write(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))

                translations.append(transcription)

                os.remove(temp_file)
    
    except Exception as e:
        print(f"Error translating: {e}")
            

def run_pipeline():
    
    # record
    freq = 44100
    duration = 3  # in seconds
    recordings_dir = './recordings'

    # transcribe
    transcriptions_dir = './transcriptions'
    model_path = "models/faster-whisper-tiny" # The model to use for transcription
    processor = "openai/whisper-tiny" # The processor to use for transcription
    
    # translate
    translations_dir = './translations'
    translator = "models/nllb-200-distilled-600M" # The model to use for translation
    tokenizer = "facebook/nllb-200-distilled-600M" # The tokenizer to use for translation
    device = "cpu" # The device to use for translation
    src_lang = "fra_Latn"
    tgt_lang = "eng_Latn"

    os.makedirs(recordings_dir, exist_ok=True)
    os.makedirs(transcriptions_dir, exist_ok=True)
    os.makedirs(translations_dir, exist_ok=True)

    queue1 = queue.Queue()
    queue2 = queue.Queue()

    threads = [
        threading.Thread(target=record_and_save, args=(queue1, recordings_dir, freq, duration)),
        threading.Thread(target=transcribe, args=(queue1, queue2, processor, model_path)),
        threading.Thread(target=translate, args=(queue2, translations_dir, src_lang, tgt_lang, translator, tokenizer, device))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print('Pipeline stopped')
        exit(0)

