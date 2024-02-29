import ctranslate2
import transformers
import librosa

# Load and resample the audio file.
audio, _ = librosa.load("audios/POV_Mini-Stories.mp3", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Load the model on CPU.
model = ctranslate2.models.Whisper("models/faster-whisper-tiny")

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
print(transcription)

src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

translator = ctranslate2.Translator("models/nllb-200-600M-onmt", device="cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

source = tokenizer.convert_ids_to_tokens(tokenizer.encode(transcription))
target_prefix = [tgt_lang]
results = translator.translate_batch([source], target_prefix=[target_prefix])
target = results[0].hypotheses[0][1:]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))