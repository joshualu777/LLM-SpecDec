
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2, speculative_sampling_merging
from globals import Decoder
import json
from  tqdm import tqdm
from alignment import TokenMapper
from datasets import load_dataset
from collections import defaultdict


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Suggest at least five related search terms to \"Mạng neural nhân tạo\".")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama2-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama2-70b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


def benchmark(fn, info, *args, **kwargs):

    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    threads = defaultdict(list)
    for message in dataset:
        if "message_tree_id" in message:
            threads[message["message_tree_id"]].append(message)

    test_sample_num = 5
    with contexttimer.Timer() as t:
        total_tokens = 0
        with tqdm(total=test_sample_num, desc=f"{info} benchmarking") as pbar:
            for tree_id, messages in list(threads.items())[:test_sample_num]:
                # Sort messages by hierarchy (if necessary)
                messages = sorted(messages, key=lambda x: x["parent_id"] or "")

                # Reconstruct the conversation
                content = "\n".join([msg["role"] + ": " + msg["text"] for msg in messages])
                # print("content", content)
                input_ids = Decoder().encode(content, return_tensors='pt').to('mps')
                if len(input_ids[0]) > 2048 :
                    continue
                output_ids = fn(input_ids, *args, **kwargs)
                generated_text = Decoder().decode(output_ids)
                # print("generated_text", generated_text)
                # print("end_generated_text")
                total_tokens += (len(output_ids[0]) - len(input_ids[0]))
                test_sample_num -= 1
                if test_sample_num < 0:
                    break
                
                pbar.update(1)
                    
    print(f"\n [benchmark] {info} tokens/sec: {total_tokens / t.elapsed}, {t.elapsed} sec generates {total_tokens} tokens")

def benchmark_merge(fn, info, mapper, *args, **kwargs):

    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    print(dataset[0].keys())

    # Group messages by thread_id
    threads = defaultdict(list)
    for message in dataset:
        if "message_tree_id" in message:
            threads[message["message_tree_id"]].append(message)

    test_sample_num = 5
    with contexttimer.Timer() as t:
        total_tokens = 0
        with tqdm(total=test_sample_num, desc=f"{info} benchmarking") as pbar:
            for tree_id, messages in list(threads.items())[:test_sample_num]:
                # Sort messages by hierarchy (if necessary)
                messages = sorted(messages, key=lambda x: x["parent_id"] or "")

                # Reconstruct the conversation
                content = "\n".join([msg["role"] + ": " + msg["text"] for msg in messages])
                # print("content: ", content)
                input_ids_draft = Decoder().encode_draft(content, return_tensors='pt').to('mps')
                input_ids_target = Decoder().encode_target(content, return_tensors='pt').to('mps')
                if len(input_ids_draft[0]) > 2048 or len(input_ids_target[0]) > 2048 :
                    print("too long")
                    continue
                output_ids = fn(mapper, input_ids_draft, input_ids_target, *args, **kwargs)
                # print(len(output_ids))
                generated_text = Decoder().decode_target(output_ids)
                # print("generated_text", generated_text)
                # print("end_generated_text")
                total_tokens += (len(output_ids[0]) - len(input_ids_target[0]))
                test_sample_num -= 1
                if test_sample_num < 0:
                    break
                
                pbar.update(1)
                    
    print(f"\n [benchmark] {info} tokens/sec: {total_tokens / t.elapsed}, {t.elapsed} sec generates {total_tokens} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=100, gamma = 4,
             random_seed = None):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    small_tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True, use_auth_token=True)
    large_tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True, use_auth_token=True)
  
    Decoder().set_tokenizer(small_tokenizer, large_tokenizer)
    
    # print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    # print("finish loading models")
    
    input_ids = small_tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 5000
    top_p = 0

    token_map = TokenMapper(approx_model_name, target_model_name)

    torch.manual_seed(123)
    benchmark(autoregressive_sampling, "AS_large", large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    benchmark(autoregressive_sampling, "AS_small", small_model, num_tokens, top_k = top_k, top_p=top_p)

    #torch.manual_seed(123)
    #benchmark(speculative_sampling, "SP", small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

    torch.manual_seed(123)
    benchmark_merge(speculative_sampling_merging, "SP", token_map, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma)
