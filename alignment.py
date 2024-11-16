import unicodedata
import multiprocessing
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer

class TokenMapper:

    def __init__(self, draft_name, target_name) -> None:
        self.tokenizer_draft = AutoTokenizer.from_pretrained(draft_name)
        self.tokenizer_target = AutoTokenizer.from_pretrained(target_name)
        self.draft_tag = (draft_name + "x" + target_name).replace("/", "_")
        self.target_tag = (target_name + "x" + draft_name).replace("/", "_")
        self.draft_index_mapping = self.align_token(self.tokenizer_draft, self.tokenizer_target, self.draft_tag)
        self.target_index_mapping = self.align_token(self.tokenizer_target, self.tokenizer_draft, self.target_tag)

    def get_correspond(self, index: int) -> list:
        index = str(index)
        if index in self.draft_index_mapping:
            return self.draft_index_mapping[index]
        print("not found in draft", index)
        return []
    
    def get_reverse(self, index: int) -> list:
        index = str(index)
        if index in self.target_index_mapping:
            return self.target_index_mapping[index]
        print("not found in target", index)
        return []

    def normalize(self, text: str) -> str:
        normalized_text = unicodedata.normalize('NFKD', text).lower()
        return ''.join(char for char in normalized_text if char.isalnum())

    def align_token(self, tokenizer_first: AutoTokenizer, tokenizer_second: AutoTokenizer, tag: str) -> dict:
        print(tag + ".json")
        if os.path.exists(tag + ".json"):
            with open(tag + ".json", "r") as infile:
                return json.load(infile)

        vocab = tokenizer_first.get_vocab()

        tokenizer_first_normalized = {}
        for token, index in vocab.items():
            tokenizer_first_normalized[tokenizer_first.convert_tokens_to_string([token])] = index
        
        vocab = self.tokenizer_target.get_vocab()

        tokenizer_second_normalized = {}
        for token, index in vocab.items():
            tokenizer_second_normalized[tokenizer_second.convert_tokens_to_string([token])] = index
        
        first_to_second = {}

        for str, index in tqdm(tokenizer_first_normalized.items(), desc="Aligning tokens", unit="token"):
            # lst = self.get_best(str, tokenizer_second_normalized)
            lst = []
            if len(lst) == 0:
                first_to_second[index] = tokenizer_second(str)["input_ids"]
            else:
                word, num = zip(*lst)
                first_to_second[index] = list(num)
        
        json_object = json.dumps(first_to_second, indent=4)
        
        with open(tag + ".json", "w") as outfile:
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

        for index, lst in first_to_second.items():
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