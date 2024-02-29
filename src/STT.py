import librosa
import transformers
import ctranslate2
import datetime
import os, glob

# find most recent files in a directory
recordings_dir = os.path.join('./recordings', '*')

# Load the model on CPU.
model = ctranslate2.models.Whisper("./models/faster-whisper-tiny")

# list to store which wav files have been transcribed
transcribed = []
try:
    while True:
        # get most recent wav recording in the recordings directory
        files = sorted(glob.iglob(recordings_dir), key=os.path.getctime, reverse=True)
        if len(files) < 1:
            continue
        
        latest_recording = files[0]
        latest_recording_split = latest_recording.split('/')
        if len(latest_recording_split) > 1:
            latest_recording_filename = latest_recording_split[1]
        else:
            print("Error: latest_recording does not contain '/'")
            latest_recording_filename = None

        if os.path.exists(latest_recording) and not latest_recording in transcribed:
            
            # Load and resample the audio file.
            audio, _ = librosa.load(latest_recording, sr=16000, mono=True)
            
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
                    "<|startoftranscript|>",
                    language,
                    "<|transcribe|>",
                    "<|notimestamps|>",  # Remove this token to generate timestamps.
                ]
            )
            
            # Run generation for the 30-second window.
            results = model.generate(features, [prompt])
            transcription = processor.decode(results[0].sequences_ids[0])
            
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
            #with open(f"./transcriptions/{filename}", "w") as f:
                #f.write(transcription)

            # append text to transcript file
            with open('./transcriptions/transcript.txt', 'a') as f:
                f.write(transcription + '\n')
            
except KeyboardInterrupt:
    print('Transcription stopped')
    exit(0)

# Path: src/STT.py