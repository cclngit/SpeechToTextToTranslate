import ctranslate2
import transformers
import datetime
import os, glob

src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

translator = ctranslate2.Translator("models/nllb-200-distilled-600M", device="cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

try:
    while True:
        # get most recent word in the transcripts directory
        files = sorted(glob.iglob('./transcriptions/transcript.txt'), key=os.path.getctime, reverse=True)
        if len(files) < 1:
            continue
        
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(open(files[0], "r").read()))
        
        target_prefix = [tgt_lang]
        results = translator.translate_batch([source], target_prefix=[target_prefix])
        target = results[0].hypotheses[0][1:]
        print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
        
        with open('./translations/translations.txt', "w") as f:
            f.write(tokenizer.decode(tokenizer.convert_tokens_to_ids(target))+ "\n")
    
except KeyboardInterrupt:
    print('Translation stopped')
    exit(0)

