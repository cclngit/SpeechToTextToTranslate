import librosa
import transformers
import ctranslate2
import datetime
import os
import glob
import multiprocessing

def transcribe_audio(audio_file):
    try:
        # Load the model on CPU.
        model = ctranslate2.models.Whisper("./models/faster-whisper-tiny")

        # Load and resample the audio file.
        audio, _ = librosa.load(audio_file, sr=16000, mono=True)

        # Compute the features of the first 30 seconds of audio.
        processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
        inputs = processor(audio, return_tensors="np", sampling_rate=16000)
        features = ctranslate2.StorageView.from_array(inputs.input_features)

        # Detect the language.
        results = model.detect_language(features)
        language, probability = results[0][0]
        print("Detected language %s with probability %f" % (language, probability))

        # Describe the task in the prompt.
        # See the prompt format in https://github.com/openai/whisper.
        prompt = processor.tokenizer.convert_tokens_to_ids(
            [
                "",
                language,
                "",
                "",  # Remove this token to generate timestamps.
            ]
        )

        # Run generation for the 30-second window.
        results = model.generate(features, [prompt])
        transcription = processor.decode(results[0].sequences_ids[0])

        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        with open(f"./transcriptions/{filename}", "w") as f:
            f.write(transcription)

        print(f"Transcription saved for {audio_file}")
    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")

if __name__ == '__main__':
    # find most recent files in a directory
    recordings_dir = os.path.join('./recordings', '*')
    transcribed = []

    try:
        while True:
            files = sorted(glob.iglob(recordings_dir), key=os.path.getctime, reverse=True)
            if len(files) < 1:
                continue
            
            latest_recording = files[0]

            if os.path.exists(latest_recording) and latest_recording not in transcribed:
                p = multiprocessing.Process(target=transcribe_audio, args=(latest_recording,))
                p.start()
                transcribed.append(latest_recording)
                
    except KeyboardInterrupt:
        print('Transcription stopped')
