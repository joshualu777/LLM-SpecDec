import torch
from tqdm import tqdm
import torch

# import editdistance

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder
from alignment import TokenMapper
from transformers import AutoTokenizer

torch.set_printoptions(precision=10)

@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix


@torch.no_grad()
def speculative_sampling_v2(token_map : TokenMapper, draft_prefix : torch.Tensor, target_prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = draft_prefix.shape[1]
    T = seq_len + max_len

    total = 0
    accepted = 0
    
    assert draft_prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        print("top_k, top_p", top_k, top_p)
        while draft_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            skip = []
            draft_seq = draft_prefix
            target_seq = target_prefix
            draft_prefix_len = draft_prefix.shape[1]
            target_prefix_len = target_prefix.shape[1]

            indexes = []

            test_tok = 1
            #print("prefix length", draft_prefix_len)
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(draft_seq).logits # the logits at indexes 9 are used to predict the 11th token
                #print("shape", q.shape)
                # print("logits", q[:, -1, :], max(q[:, -1, :]), "end")
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                # indexes.append(next_tok)
                # cur_index = draft_prefix_len + _
                #print("shape", q.shape)
                #print("draft token probability", cur_index, q[:, cur_index - 1,next_tok])
                #test_tok = next_tok
                # print("logit prob", q[:, -1, next_tok])
                mapped_toks = token_map.get_correspond(next_tok)
                draft_seq = torch.cat((draft_seq, next_tok), dim=1)
                skip.append(max(1, len(mapped_toks)))
                if len(mapped_toks) == 0:
                    target_seq = torch.cat((target_seq, next_tok), dim=1) # TODO need better default!!!
                    #print(_, next_tok, "not mapped")
                for tok in mapped_toks:
                    tok_tensor = torch.tensor(tok).view(1, 1).to(target_seq.device)
                    target_seq = torch.cat((target_seq, tok_tensor), dim=1)
                    #print(_, tok)
                #print(next_tok)

            tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
            tokenizer_draft = AutoTokenizer.from_pretrained("facebook/opt-125m")

            # Convert indexes to tokens
            # tokens = [tokenizer_draft.convert_ids_to_tokens(index) for index in draft_seq]
            # print("Start", tokens)
            # tokens = [tokenizer.convert_ids_to_tokens(index) for index in target_seq]
            # print("Start Target", tokens)

            # normalize the logits
            #print(q.shape, len(indexes), draft_prefix_len - 1)
            #q = approx_model(draft_seq).logits # added by me
            #print(q.shape, len(indexes), draft_prefix_len - 1)
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
                # if i >= draft_prefix_len and i < draft_prefix_len + gamma:
                #     pass
                    #print("normalized draft token probability", i, q[:,i - 1,indexes[i - draft_prefix_len]])
                
                # print("testing prob", q[:,i,test_tok])
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(target_seq).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
                # for j in range(p.shape[2]):
                #     if p[:,i,j] > 0:
                #         print("not zero", i, j, p[:,i,j])

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            # n = draft_prefix_len - 1

            # print("seq", draft_seq)
            # print([tokenizer_draft.convert_ids_to_tokens(index) for index in draft_seq])
            draft_index = draft_prefix_len
            target_index = target_prefix_len
            # print("draft_index is", draft_index, draft_seq.shape, gamma)
            for i in range(0, gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = draft_seq[:, draft_prefix_len + i]
                # print("j is", j)
                
                #print(Decoder().decode(j))

                prob_p = 0
                for _ in range(skip[i]):
                    #print("cur probability is", p[:, target_index - 1, target_seq[:, target_index]], target_seq[:, target_index])
                    prob_p = max(prob_p, p[:, target_index - 1, target_seq[:, target_index]].item())
                    print(tokenizer.convert_ids_to_tokens(target_seq[:, target_index]))
                    target_index += 1

                print("the probability is", prob_p, q[:, draft_prefix_len + i - 1, j])

                # print(q.shape, draft_prefix_len + i)
                if r < torch.min(torch.tensor([1], device=q.device), prob_p / q[:, draft_prefix_len + i - 1, j]): # check the math here
                    # accept, and update n
                    print("accepted")
                    draft_index += 1
                    accepted += 1
                    total += 1
                else:
                    # reject
                    #t = sample(max_fn(p[:, n, :] - q[:, n, :])) # problematic, the index order is not the same
                    #is_all_accept = False
                    testing = sample(p[:, target_index - skip[i] - 1, :])
                    print("desired tok", testing, p[:, target_index - skip[i] - 1, testing], tokenizer.convert_ids_to_tokens(testing))
                    target_index -= skip[i]
                    total += 1
                    break
         
            draft_prefix = draft_seq[:, :draft_index]
            target_prefix = target_seq[:, :target_index]
            # tokens = [tokenizer_draft.convert_ids_to_tokens(index) for index in draft_prefix]
            # print("End", tokens)
            # tokens = [tokenizer.convert_ids_to_tokens(index) for index in target_prefix]
            # print("End Target", tokens)

            # Convert indexes to tokens
            #tokens = [tokenizer.convert_ids_to_tokens(index) for index in indexes]
            # print("final", tokens)
            
            # if is_all_accept:
            #     t = sample(p[:, -1, :])
            t = sample(p[:, target_index - 1, :])
            print("prob of t", p[:,target_index - 1,t])
            target_prefix = torch.cat((target_prefix, t), dim=1)

            new_draft_tokens = token_map.get_reverse(t)
            for tok in new_draft_tokens:
                tok_tensor = torch.tensor(tok).view(1, 1).to(draft_prefix.device)
                draft_prefix = torch.cat((draft_prefix, tok_tensor), dim=1)

            #draft_prefix = torch.cat((draft_prefix, t), dim=1)
            # tokens = [tokenizer_draft.convert_ids_to_tokens(index) for index in draft_prefix]
            # print("Extra End", tokens)
            # tokens = [tokenizer.convert_ids_to_tokens(index) for index in target_prefix]
            # print("End End Target", tokens)
            pbar.update(draft_index - pbar.n)

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

    #tokens = [tokenizer.convert_ids_to_tokens(index) for index in target_prefix]
    # print(tokens)
    print("accepted:", accepted, "out of", total)
    return target_prefix

# Sch' ool' is' fun'.
# School' is' fun'.

# Berkeley is very hot.

# Find task for LLM Fusion