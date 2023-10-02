#%%

# Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
#%%

from transformers import MarianMTModel, MarianTokenizer
import torch

# src_text = [
#     ">>ara<< I can't help you because I'm busy.",
#     ">>ara<< I have to write a letter. Do you have some paper?"
# ]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

src_text = [
    ">>ara<< a street sign in the middle of the night",
]

model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
model.to(device)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))

for t in translated:
    print( tokenizer.decode(t, skip_special_tokens=True) )

# expected output:
#     لا أستطيع مساعدتك لأنني مشغول.
#     يجب أن أكتب رسالة هل لديك بعض الأوراق؟

# %%
# from transformers import pipeline
# pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ar")
# print(pipe("I can't help you because I'm busy.",))

# expected output: لا أستطيع مساعدتك لأنني مشغول.
# %%
