import datetime
import os
import glob
import transformers
import ctranslate2
from collections import deque
import utils


def translate(queue1,queue2, translations_dir, src_lang, tgt_lang, translator, tokenizer, device="cpu"):
    
    try:
        translator = ctranslate2.Translator(translator, device=device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, src_lang=src_lang)
        translations = deque(maxlen=10)

        while True:
            transcription = queue1.get()
            if not transcription in translations:
                temp_file = f"temp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                with open(temp_file, "w") as f:
                    f.write(transcription)

                source = tokenizer.convert_ids_to_tokens(tokenizer.encode(open(temp_file, "r").read()))
                target_prefix = [tgt_lang]
                results = translator.translate_batch([source], target_prefix=[target_prefix])
                target = results[0].hypotheses[0][1:]
                translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))

                print(translated_text)
                
                queue2.put(translated_text)

                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
                with open(f"{translations_dir}/{filename}", "w") as f:
                    f.write(translated_text)

                translations.append(transcription)

                os.remove(temp_file)
                utils.delete_old_files(f"{translations_dir}/*.txt")
    
    except Exception as e:
        print(f"Error translating: {e}")
            

if __name__ == "__main__":
    pass