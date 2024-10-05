import unicodedata
from transformers import AutoTokenizer

def normalize(text: str) -> str:
    normalized_text = unicodedata.normalize('NFKD', text).lower()
    return ''.join(char for char in normalized_text if char.isalnum())

def align_token(draft_model_name: str, target_model_name: str):
    tokenizer_draft = AutoTokenizer.from_pretrained(draft_model_name)
    vocab = tokenizer_draft.get_vocab()

    tokenizer_draft_normalized = {}
    for token, index in vocab.items():
        tokenizer_draft_normalized[normalize(token)] = index
    
    tokenizer_target = AutoTokenizer.from_pretrained(target_model_name)
    vocab = tokenizer_target.get_vocab()

    tokenizer_target_normalized = {}
    for token, index in vocab.items():
        tokenizer_target_normalized[normalize(token)] = index
    
    draft_to_target = {}
    print(len(tokenizer_draft_normalized))
    cnt = 0
    for token, index in tokenizer_draft_normalized.items():
        draft_to_target[token] = get_best(token, tokenizer_target_normalized)
        print(cnt, token, draft_to_target[token])
        cnt += 1

def get_best(cur_token: str, vocab: dict) -> list:
    best_score = 0
    best_match = []
    for token, index in vocab.items():
        cur_score = compute_score(token, cur_token)
        if cur_score > best_score:
            best_score = cur_score
            best_match = [token]

    if best_score > 0:
        index = cur_token.find(best_match[0])
        best_lower = get_best(cur_token[:index], vocab)
        best_upper = get_best(cur_token[index + len(best_match[0]):], vocab)
        return best_lower + best_match + best_upper
    return best_match

def compute_score(str1: str, str2: str) -> float:
    if str1 in str2:
        if len(str2) == 0:
            return 0
        return len(str1) / len(str2)
    return 0