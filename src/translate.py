import datetime
import os
import glob
import transformers
import ctranslate2
from collections import deque


def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")


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

                print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
                
                queue2.put(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))

                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
                with open(f"{translations_dir}/{filename}", "w") as f:
                    f.write(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))

                translations.append(transcription)

                os.remove(temp_file)
    
    except Exception as e:
        print(f"Error translating: {e}")
            

if __name__ == "__main__":
    pass