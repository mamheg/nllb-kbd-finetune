from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from tqdm import trange

# model_load_name = 'models/NLLB_v7/nllb_kbd_resize'
model_load_name = "facebook/nllb-200-distilled-600M"
model_load_name = "models/NLLB_v10/nllb_kbd"

model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)

# print(model)
def translate(
    text, src_lang='rus_Cyrl', tgt_lang='zul_Latn', a=32, b=3, num_return_sequences=4, max_input_length=1024, num_beams=10, **kwargs):
    """Turn a text or a list of texts into a list of translations"""

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True,
        max_length=max_input_length
    )
    model.eval() # turn off training mode
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, num_return_sequences=num_return_sequences, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

def batched_translate(text, src_lang='rus_Cyrl', tgt_lang='zul_Latn', batch_size=16, **kwargs):
    """Translate texts in batches of similar length"""
    text = text.replace('1', 'Ӏ')
    texts = text.split('.')
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in trange(0, len(texts2), batch_size):
        results.extend(translate(texts2[i: i+batch_size], src_lang, tgt_lang))
    return [p for i, p in sorted(zip(idxs, results))]

while True:
    try:
        text = input("text>>> ")
        print(translate(text, 'rus_Cyrl', 'kbd_Cyrl'))
        print(translate(text, 'kbd_Cyrl', 'rus_Cyrl'))
    except: print('ОШибка, еще раз')
    # print(batched_translate(text, 'zul_Latn', 'rus_Cyrl'))
    # print(batched_translate(text, 'rus_Cyrl', 'eng_Latn'))