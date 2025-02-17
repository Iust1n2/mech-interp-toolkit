from typing import List, Tuple
from transformers import AutoTokenizer
import os
import torch
import json
import numpy as np
import copy
import re
import warnings
from transformer_lens import HookedTransformer

def load_dataset(dataset_path: str): 
    with open(dataset_path, "r") as f:
        json_dataset = json.load(f)
        
        clean_prompts = []
        corr_prompts = []
        clean_answer_toks = []
        wrong_toks_1 = []
        wrong_toks_2 = []
        
        for i, data in enumerate(json_dataset):
            clean_prompt = data['1']
            corr_prompt = data['2']
            correct_token = data['correct_token']
            wrong_1 = data['wrong_1']
            wrong_2 = data['wrong_2']

            clean_prompts.append(clean_prompt)
            corr_prompts.append(corr_prompt)
            clean_answer_toks.append(correct_token)
            wrong_toks_1.append(wrong_1)
            wrong_toks_2.append(wrong_2)
        
        dataset = {
            'clean_prompts': clean_prompts,
            'corr_prompts': corr_prompts,
            'correct_answer_toks': clean_answer_toks,
            'wrong_toks_1': wrong_toks_1,
            'wrong_toks_2': wrong_toks_2
        }
        # Check if all lists have the same length
        assert len(dataset['clean_prompts']) == len(dataset['corr_prompts']) == len(dataset['wrong_toks_1']) == len(dataset['wrong_toks_2']) == len(dataset['correct_answer_toks'])
        
        return dataset
    
def tokenize_answers(model, answers, device):
        answer_toks = []
        for answer in answers:
            token_pair = []
            for ans in answer:
                # Convert the answer string to token and then to its corresponding token ID
                token = model.to_single_token(ans)
                token_pair.append(token)
            answer_toks.append(token_pair)
        # Convert the list of token pairs to a PyTorch tensor
        answer_toks = torch.tensor(answer_toks, device=device)
        return answer_toks
    
def zip_and_tokenize_all_answers(model, clean_answers, corr_answers, device):
    all_answer_toks = []
    all_answers = []
    assert len(clean_answers) == len(corr_answers)
    for i in range(len(clean_answers)):
        all_answers.append(clean_answers[i])
        all_answers.append(corr_answers[i])

    for pair in all_answers:
        token_pair = []
        for ans in pair:
            token = model.to_single_token(ans)
            token_pair.append(token)
        
        all_answer_toks.append(token_pair)
    
    answer_tokens_ids = torch.tensor(all_answer_toks).to(device)
    return answer_tokens_ids, all_answers
    
def tokenize_prompts(model, prompts_strings):
    all_tokens = model.to_tokens(prompts_strings, prepend_bos=True, padding_side="left")
    return all_tokens

class Dataset:
    def __init__(
        self, 
        model: HookedTransformer, 
        file_path: str, 
        tokenizer: AutoTokenizer, 
        device: torch.device = torch.device("cuda")
        ):
        if tokenizer is None:
            try: 
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except OSError as e:
                print(f"Error: {e}")
                print("Please provide a valid tokenizer")
            else: 
                self.tokenizer = tokenizer

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.dataset = load_dataset(file_path)
        self.model = model
        self.device = device
        
        self.clean_prompts = self.dataset['clean_prompts']
        self.corr_prompts = self.dataset['corr_prompts']

        # Both clean and correct prompts
        self.all_prompts = list(zip(self.clean_prompts, self.corr_prompts))
        self.all_prompts_strings = [prompt for prompt_pair in self.all_prompts for prompt in prompt_pair]

        self.correct_answer_toks = self.dataset['correct_answer_toks']
        self.clean_wrong_toks = self.dataset['wrong_toks_1']
        self.corr_wrong_toks = self.dataset['wrong_toks_2']

        # Adding correct answer tokens to the end of each prompt
        self.clean_texts = [f"{prompt}{answer}" for prompt, answer in zip(self.clean_prompts, self.correct_answer_toks)]
        self.corr_texts = [f"{prompt}{answer}" for prompt, answer in zip(self.corr_prompts, self.correct_answer_toks)]

        # Both clean and wrong answer for Logit Diff
        self.clean_answers = [(correct, wrong) for correct, wrong in zip(self.correct_answer_toks, self.clean_wrong_toks)]
        self.corr_answers = [(correct, wrong) for correct, wrong in zip(self.correct_answer_toks, self.corr_wrong_toks)]

        # Tokenize answers
        self.clean_answer_tok_ids = tokenize_answers(self.model, self.clean_answers, self.device)
        self.corr_answer_tok_ids = tokenize_answers(self.model, self.corr_answers, self.device)
        self.all_answer_tok_ids, self.all_answer_strings = zip_and_tokenize_all_answers(self.model, self.clean_answers, self.corr_answers, self.device)

        # Tokenize all prompts
        self.all_token_ids = tokenize_prompts(self.model, self.all_prompts_strings)

    def __len__(self):
        return len(self.all_prompts)
    
if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2")
    dataset = Dataset(file_path="data/dataset.json", model= model, tokenizer=None, device=torch.device("cuda"))
    print(len(dataset))
    print(dataset.all_prompts)
    print(dataset.clean_texts)
    print(dataset.corr_texts)
    print(dataset.clean_answers)
    print(dataset.corr_answers)
    print(dataset.clean_answer_tok_ids)
    print(dataset.corr_answer_tok_ids)
    print(dataset.all_answer_tok_ids)
    print(dataset.all_token_ids)


    