import os
import transformers
import ctranslate2
from datetime import datetime


def translate(queue1, queue2, src_lang, tgt_lang, translator, tokenizer, device="cpu"):
    try:
        translator = ctranslate2.Translator(translator, device=device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, src_lang=src_lang)
        translations = set()

        while True:
            transcription = queue1.get()
            if transcription not in translations:
                temp_file = f"temp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                with open(temp_file, "w") as f:
                    f.write(transcription)

                source = tokenizer.convert_ids_to_tokens(tokenizer.encode(open(temp_file, "r").read()))
                target_prefix = [tgt_lang]
                results = translator.translate_batch([source], target_prefix=[target_prefix])
                target = results[0].hypotheses[0][1:]
                translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))

                print(translated_text)
                queue2.put(translated_text)

                translations.add(transcription)

                os.remove(temp_file)
    
    except Exception as e:
        print(f"Error translating: {e}")



if __name__ == '__main__':
    pass
