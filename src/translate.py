import ctranslate2
import transformers
import datetime
import os, glob

src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

translator = ctranslate2.Translator("models/nllb-200-distilled-600M", device="cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

# find most recent files in a directory
transcriptions_dir = os.path.join('./transcriptions', '*')

# list to store which txt files have been transcribed
transcriptions = []

try:
    while True:
        # get most recent txt transcription in the transcriptions directory
        files = sorted(glob.iglob(transcriptions_dir), key=os.path.getctime, reverse=True)
        if len(files) < 1:
            continue
        
        latest_transcription = files[0]
        latest_transcription_split = latest_transcription.split('/')
        if len(latest_transcription_split) > 1:
            latest_transcription_filename = latest_transcription_split[1]
        else:
            print("Error: latest_transcription does not contain '/'")
            latest_transcription_filename = None

        if os.path.exists(latest_transcription) and not latest_transcription in transcriptions:
        
            source = tokenizer.convert_ids_to_tokens(tokenizer.encode(open(latest_transcription, "r").read()))
            target_prefix = [tgt_lang]
            results = translator.translate_batch([source], target_prefix=[target_prefix])
            target = results[0].hypotheses[0][1:]
            print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
            
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
            with open(f"./translations/{filename}", "w") as f:
                f.write(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
            
            # save list of transcription so that we don't translate the same one again
            transcriptions.append(latest_transcription)
            print(f"Translation saved for {latest_transcription_filename}")
        
        
except KeyboardInterrupt:
    print('Translation stopped')
    exit(0)

