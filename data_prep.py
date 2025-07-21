# import os 
# import urllib.request 
# from tokenizer import SimpleTokenizerV1
"""
 Dowloading the dataset : 
 """

# if not os.path.exists("the-verdict.txt"):
#     url = ("https://raw.githubusercontent.com/rasbt/"
#            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#            "the-verdict.txt")
#     file_path = "the-verdict.txt"
#     # urllib.request.urlretrieve(url,file_path)
    

# with open("the-verdict.txt" , "r", encoding="utf-8") as f : 
#     raw_text = f.read()

# # print("Total number of character:", len(raw_text))
# # print(raw_text[:99])



# '''
# Let's split the sentence into tokens : 
# '''
# import re 
# result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# result = [item for item in result if item.strip()]
# # print(result)

# all_words = sorted(set(result))
# vocab_size = len(all_words)
# # print(all_words[:10])

# vocab = {token:integer for integer , token in enumerate(all_words)}

# """
# Let's handle in the unknown token which is not present in the vocab
# by adding the special tokens
# """
# all_tokens = sorted(list(set(result)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# vocab = {token:integer for integer,token in enumerate(all_tokens)}

# """
# Convert the above code into class to reuseability
# """
# tokenizer = SimpleTokenizerV1(vocab)

# text = """"It's the last he painted, you know," 

#            Mrs. Gisburn said with pardonable pride."""
           
# text = "Hi there my name is Saurabh !!"
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."

# text = " <|endoftext|> ".join((text1, text2))
# ids = word_tokenizer.encode(text)
# print(ids)
# text = word_tokenizer.decode(ids)
# print(text)

import torch
import tiktoken
from src.dataloader import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader
tokenizer = tiktoken.get_encoding("gpt2")


with open("the-verdict.txt", "r", encoding="utf-8") as f :
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]

context_size = 4 

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

# print(f"x: {x}")
# print(f"y:      {y}")


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)

# print(token_embeddings.shape)

max_length = 4 
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# print(pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings

print(input_embeddings.shape)