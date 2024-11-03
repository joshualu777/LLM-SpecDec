# import unicodedata
# from transformers import AutoTokenizer
# from concurrent.futures import ProcessPoolExecutor, as_completed

# def normalize(text: str) -> str:
#     normalized_text = unicodedata.normalize('NFKD', text).lower()
#     return ''.join(char for char in normalized_text if char.isalnum())

# def align_token(draft_model_name: str, target_model_name: str, max_workers: int = 100):
#     tokenizer_draft = AutoTokenizer.from_pretrained(draft_model_name)
#     vocab_draft = tokenizer_draft.get_vocab()

#     tokenizer_draft_normalized = {}
#     for token, index in vocab_draft.items():
#         tokenizer_draft_normalized[normalize(token)] = index
#         print(index)

#     tokenizer_target = AutoTokenizer.from_pretrained(target_model_name)
#     vocab_target = tokenizer_target.get_vocab()

#     tokenizer_target_normalized = {}
#     for token, index in vocab_target.items():
#         tokenizer_target_normalized[normalize(token)] = index

#     draft_to_target = {}
#     print(len(tokenizer_draft_normalized))

#     # Use ProcessPoolExecutor to parallelize the get_best computation
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         future_to_token = {
#             executor.submit(get_best, token, tokenizer_target_normalized): token
#             for token in tokenizer_draft_normalized.keys()
#         }

#         for cnt, future in enumerate(as_completed(future_to_token)):
#             token = future_to_token[future]
#             try:
#                 best_match = future.result()
#                 draft_to_target[token] = best_match
#                 print(cnt, token, tokenizer_draft.convert_tokens_to_ids(token), best_match)
#             except Exception as exc:
#                 print(f"Token {token} generated an exception: {exc}")

# def get_best(cur_token: str, vocab: dict) -> list:
#     best_score = 0
#     best_match = []
#     for token, index in vocab.items():
#         cur_score = compute_score(token, cur_token)
#         if cur_score > best_score:
#             best_score = cur_score
#             best_match = [token]

#     if best_score > 0:
#         index = cur_token.find(best_match[0])
#         best_lower = get_best(cur_token[:index], vocab)
#         best_upper = get_best(cur_token[index + len(best_match[0]):], vocab)
#         return best_lower + best_match + best_upper
#     return best_match

# def compute_score(str1: str, str2: str) -> float:
#     if str1 in str2:
#         if len(str2) == 0:
#             return 0
#         return len(str1) / len(str2)
#     return 0

import unicodedata
import multiprocessing
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer

class TokenMapper:

    def __init__(self, draft_name, target_name) -> None:
        self.draft_index_mapping = {}
        self.align_token(draft_name, target_name)

    def get_correspond(self, index: int) -> list:
        if index in self.draft_index_mapping:
            self.draft_index_mapping[index]
        return []

    def normalize(self, text: str) -> str:
        normalized_text = unicodedata.normalize('NFKD', text).lower()
        return ''.join(char for char in normalized_text if char.isalnum())

    def align_token(self, draft_model_name: str, target_model_name: str):
        if os.path.exists("mapping.json"):
            with open("mapping.json", "r") as infile:
                self.draft_index_mapping = json.load(infile)
            return

        tokenizer_draft = AutoTokenizer.from_pretrained(draft_model_name)
        vocab = tokenizer_draft.get_vocab()

        tokenizer_draft_normalized = {}
        for token, index in vocab.items():
            tokenizer_draft_normalized[self.normalize(token)] = index
        
        tokenizer_target = AutoTokenizer.from_pretrained(target_model_name)
        vocab = tokenizer_target.get_vocab()

        tokenizer_target_normalized = {}
        for token, index in vocab.items():
            tokenizer_target_normalized[self.normalize(token)] = index
        
        draft_to_target = {}

        for token, index in tqdm(tokenizer_draft_normalized.items(), desc="Aligning tokens", unit="token"):
            lst = self.get_best(token, tokenizer_target_normalized)
            if len(lst) == 0:
                draft_to_target[token] = []
                self.draft_index_mapping[index] = []
            else:
                word, num = zip(*lst)
                draft_to_target[token] = list(word)
                self.draft_index_mapping[index] = list(num)
            #print(token, draft_to_target[token])
            #print(index, self.draft_index_mapping[index])
        
        json_object = json.dumps(self.draft_index_mapping, indent=4)
        
        with open("mapping.json", "w") as outfile:
            outfile.write(json_object)

        MAX_SPLIT = 10
        statistics = {
            "exact_match": 0,
            "no_match": 0,
            "split": {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                MAX_SPLIT: 0
            }
        }

        for token, lst in draft_to_target.items():
            if len(lst) == 0:
                statistics["no_match"] += 1
            elif len(lst) == 1:
                statistics["exact_match"] += 1
            else:
                if len(lst) < MAX_SPLIT:
                    statistics["split"][len(lst)] += 1
                else:
                    statistics["split"][MAX_SPLIT] += 1

        json_object = json.dumps(statistics, indent=4)
        
        with open("statistics.json", "w") as outfile:
            outfile.write(json_object)


    def get_best(self, cur_token: str, vocab: dict) -> list:
        best_score = 0
        best_match = []
        for token, index in vocab.items():
            cur_score = self.compute_score(token, cur_token)
            if cur_score > best_score:
                best_score = cur_score
                best_match = [(token, index)]

        if best_score > 0:
            index = cur_token.find(best_match[0][0])
            best_lower = self.get_best(cur_token[:index], vocab)
            best_upper = self.get_best(cur_token[index + len(best_match[0][0]):], vocab)
            return best_lower + best_match + best_upper
        return best_match

    def compute_score(self, str1: str, str2: str) -> float:
        if str1 in str2:
            if len(str2) == 0:
                return 0
            return len(str1) / len(str2)
        return 0

# reawakening

# Daft: reawaken ing'
# Target: re awakening

# Add statistics for the alignment
    # How many tokens found exact, match, how many split, how many didn't have match

# Want general matching: learn a mapping, take into account of byte-level sampling (letter by letter)
# Learn a mapping function, use histogram
# Neural network for generating mapping

# Read more about the token alignment

# Look at LLM Fusion, Zero-shot Tokenizer
# Look at papers that cited LLM Fusion