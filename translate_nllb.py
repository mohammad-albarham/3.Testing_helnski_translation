from transformers import NllbMoeForConditionalGeneration, NllbTokenizer
from logger import logger
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
import json
from accelerate import Accelerator
import pandas as pd

logger.info(f"Start the script")

# Print the current time
current_time = datetime.datetime.now()
accelerator = Accelerator()

logger.info(f"Current Time: {current_time}")

# Importing the dataset 


english_caption = []
data = []

input_file = "SecretAgent.csv"
output_file = "SecretAgent_translated_ar2en.csv"
model_name = "facebook/nllb-200-distilled-600M"


# importing pandas as pd 
import pandas as pd 
  


df = pd.read_csv(input_file)


print(df.columns)



for i, itm in df.iterrows():
    if not(isinstance(itm["Arabic"], str)):
      df.loc[i, "Arabic"] =  str(itm["Arabic"])



logger.info(f"total dataset size is {len(english_caption)}")


# Testing the device needed 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]


logger.info(f"The available gpus available is {available_gpus}")


class CaptionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return self.df["Arabic"][index]
    


# Use a pipeline as a high-level helper
from transformers import pipeline

import torch 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="arb_Arab", tgt_lang="eng_Latn")

from transformers import pipeline

pipe = pipeline("translation", model=model_name,src_lang="arb_Arab", tgt_lang="eng_Latn", tokenizer=tokenizer, device_map="auto")


logger.info(f"The model used on the translation is {model_name}")




test_data = CaptionDataset(df)

logger.info(f"The dataset for training length {len(test_data)}")


# Define the dataloader for the dataset 

test_dataloader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    # collate_fn=custom_collate_fn,
)

tot_test_dataloader = len(test_dataloader)

# model,training_dataloader = accelerator.prepare(model,test_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_res = []
with torch.no_grad():

    for i, batch in enumerate(tqdm(test_dataloader, total=tot_test_dataloader)):

        # batch = {k: v.to(device) for k, v in batch.items()}
        # output_tokens = # model.generate(**batch)
        # decoded_tokens += tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)
        res = pipe(batch)
        for itm in res:
            out_res.append(itm["translation_text"])

df["en_translated"] = out_res


df.reset_index(drop=True, inplace=True)

df.to_csv(output_file)

current_time_now = datetime.datetime.now()

time_difference = current_time_now - current_time

logger.info(f"Script time {time_difference}")

