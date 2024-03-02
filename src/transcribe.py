import os
import glob
import librosa
import transformers
import ctranslate2
from collections import deque


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
        
if __name__ == "__main__":
    pass
