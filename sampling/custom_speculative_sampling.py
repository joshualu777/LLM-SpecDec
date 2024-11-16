import torch
from tqdm import tqdm
import torch


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder
from alignment import TokenMapper
from transformers import AutoTokenizer

torch.set_printoptions(precision=10)

@torch.no_grad()
def speculative_sampling_merging(token_map : TokenMapper, draft_prefix : torch.Tensor, target_prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """

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
        while draft_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            skip = []
            draft_seq = draft_prefix
            target_seq = target_prefix
            draft_prefix_len = draft_prefix.shape[1]
            target_prefix_len = target_prefix.shape[1]

            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(draft_seq).logits # the logits at indexes 9 are used to predict the 11th token
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                mapped_toks = token_map.get_correspond(next_tok.item())
                draft_seq = torch.cat((draft_seq, next_tok), dim=1)
                skip.append(max(1, len(mapped_toks)))
                if len(mapped_toks) == 0:
                    target_seq = torch.cat((target_seq, next_tok), dim=1) # TODO need better default!!!
                for tok in mapped_toks:
                    tok_tensor = torch.tensor(tok).view(1, 1).to(target_seq.device)
                    target_seq = torch.cat((target_seq, tok_tensor), dim=1)

            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
                
            p = target_model(target_seq).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            
            is_all_accept = True

            draft_index = draft_prefix_len
            target_index = target_prefix_len
            for i in range(0, gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = draft_seq[:, draft_prefix_len + i]

                prob_p = 0
                for _ in range(skip[i]):
                    prob_p = max(prob_p, p[:, target_index - 1, target_seq[:, target_index]].item())
                    target_index += 1


                if r < torch.min(torch.tensor([1], device=q.device), prob_p / q[:, draft_prefix_len + i - 1, j]):
                    # accept
                    draft_index += 1
                    accepted += 1
                    total += 1
                else:
                    # reject
                    target_index -= skip[i]
                    total += 1
                    break
         
            draft_prefix = draft_seq[:, :draft_index]
            target_prefix = target_seq[:, :target_index]

            t = sample(p[:, target_index - 1, :])
            target_prefix = torch.cat((target_prefix, t), dim=1)

            new_draft_tokens = token_map.get_reverse(t.item())
            for tok in new_draft_tokens:
                tok_tensor = torch.tensor(tok).view(1, 1).to(draft_prefix.device)
                draft_prefix = torch.cat((draft_prefix, tok_tensor), dim=1)

            pbar.update(draft_index - pbar.n)

    print("accepted:", accepted, "out of", total)
    return target_prefix