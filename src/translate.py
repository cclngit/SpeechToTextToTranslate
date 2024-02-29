import ctranslate2
import transformers

src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

translator = ctranslate2.Translator("models/nllb-200-600M-onmt", device="cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello and welcome to the effortless English podcast This is AJ Ho talking to you from San Francisco, California Today's topic is called TPR storytelling and point of view mini stories Now this starts with an article by Blaine Ray Blaine Ray developed the TPR storytelling technique TPR means total physical response. We had a podcast about that a few weeks ago."))
target_prefix = [tgt_lang]
results = translator.translate_batch([source], target_prefix=[target_prefix])
target = results[0].hypotheses[0][1:]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))