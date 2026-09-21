import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


_IB_MOD = None
_IB_INITED = False
send_recv_idx = 0

def _ib_mod():
    global _IB_MOD
    if _IB_MOD is not None:
        return _IB_MOD
 
    import importlib
    _IB_MOD = importlib.import_module("pyextlxlink._L")
    return _IB_MOD



def _ib_init_once():
    global _IB_INITED
    if _IB_INITED:
        return
    _ib_mod().init_ib_py()
    _IB_INITED = True


def _bitcast_to_u16_cuda(t: torch.Tensor) -> torch.Tensor:
    if not t.is_contiguous():
        t = t.contiguous()

    return t.view(torch.uint16)

IB_RECV_BUFFER = None
def ib_send_recv(t: torch.Tensor, data_num, clear_kv, all_dummy, is_prefill):
    global IB_RECV_BUFFER
    if IB_RECV_BUFFER is None:
        IB_RECV_BUFFER = torch.empty(3584, device=t.device, dtype=torch.uint16)

    _ib_mod().gpuSend_Recv_py(_bitcast_to_u16_cuda(t), IB_RECV_BUFFER, data_num, clear_kv, all_dummy, is_prefill) 
    u16 = IB_RECV_BUFFER.unsqueeze(0).unsqueeze(0)
    return u16.view(t.dtype)


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @staticmethod
    def from_model_dir(model_dir):
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return Qwen2Config(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw.get("num_key_value_heads", raw["num_attention_heads"])),
            rms_norm_eps=float(raw.get("rms_norm_eps", raw.get("layer_norm_eps", 1e-6))),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 32768)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        x32 = x32 * self.weight.float()
        return x32.to(dtype=x_dtype)


class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = None
    def forward(
        self,
        input_ids,
        *,
        kv_caches=None,
        start_pos=0,
        is_prefill
    ):
        global send_recv_idx
        x = self.embed_tokens(input_ids)
        x = x.squeeze(0)
        
        new_caches = []
        return x, new_caches


class Qwen2LMHeadPart(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = nn.Module()
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, hidden_states):
        np.save("norm_lm_in.npy", hidden_states.detach().cpu().numpy())
        h = self.model.norm(hidden_states)
        lm_out = self.lm_head(h)
        np.save("norm_lm_out.npy", lm_out.detach().cpu().numpy())
        return lm_out

def _set_module_tensor_as_param(module, name, tensor):
    parts = name.split(".")
    parent = module
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return False
        parent = getattr(parent, p)
        if not isinstance(parent, nn.Module):
            return False
    leaf = parts[-1]
    if not hasattr(parent, leaf):
        return False
    cur = getattr(parent, leaf)
    if isinstance(cur, nn.Parameter):
        setattr(parent, leaf, nn.Parameter(tensor, requires_grad=False))
        return True
    return False


def load_weights_safetensors_direct_to_gpu(
    model: nn.Module,
    model_dir,
    *,
    device,
    dtype,
    key_strip_prefix: str = "",
):
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    elif os.path.exists(single_path):
        shard_files = ["model.safetensors"]
    else:
        return

    missing = {k for k, _ in model.named_parameters()}
    unexpected = []

    for _si, fn in enumerate(shard_files):
        shard_path = os.path.join(model_dir, fn)
        shard_sd = load_file(shard_path, device=str(device))
        for k, t in shard_sd.items():
            if k.endswith("rotary.inv_freq"):
                continue
            if t.dtype != dtype:
                t = t.to(dtype=dtype)
            load_k = k
            if key_strip_prefix and load_k.startswith(key_strip_prefix):
                load_k = load_k[len(key_strip_prefix) :]
            ok = _set_module_tensor_as_param(model, load_k, t)
            if ok:
                missing.discard(k)
                print(f"==============  finish loading {k}  ==============")
            else:
                unexpected.append(k)

    if "lm_head.weight" in missing and hasattr(model, "lm_head"):
        if hasattr(model, "embed_tokens"):
            model.lm_head.weight = model.embed_tokens.weight
            missing.discard("lm_head.weight")
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight = model.model.embed_tokens.weight
            missing.discard("lm_head.weight")

    if missing or unexpected:
        msg = []
        if missing:
            m = sorted(missing)
            msg.append(f"missing keys: {m[:20]}{' ...' if len(m) > 20 else ''}")
        if unexpected:
            msg.append(f"unexpected keys: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
        return


def _load_generation_config(model_dir):
    p = os.path.join(model_dir, "generation_config.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_eos_ids(gc, args_eos_id):
    eos = []
    v = gc.get("eos_token_id", None)
    if isinstance(v, int):
        eos.append(int(v))
    elif isinstance(v, list):
        for x in v:
            eos.append(int(x))
    if int(args_eos_id) >= 0:
        eos.append(int(args_eos_id))
    eos = [int(x) for x in eos if int(x) >= 0]
    return set(eos) if eos else None


def _apply_repetition_penalty_(logits, generated, penalty):
    if not generated:
        return logits
    if penalty is None:
        return logits
    p = float(penalty)
    if p == 1.0 or p <= 0:
        return logits
    idx = torch.tensor(list({int(x) for x in generated}), device=logits.device, dtype=torch.long)
    selected = logits.index_select(0, idx)
    selected = torch.where(selected < 0, selected * p, selected / p)
    return logits.scatter(0, idx, selected)


def sample_token(
    logits,
    *,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    generated,
):
    logits = logits.detach()
    logits = _apply_repetition_penalty_(logits, generated, repetition_penalty)

    t = float(temperature)
    if t <= 0:
        return int(torch.argmax(logits).item())
    logits = logits.float() / max(t, 1e-5)

    k = int(top_k)
    if k > 0 and k < int(logits.shape[-1]):
        v, _ = torch.topk(logits, k)
        cutoff = v[-1]
        logits = torch.where(logits < cutoff, torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype), logits)

    probs = torch.softmax(logits, dim=-1)
    p = float(top_p)
    if p >= 1.0:
        return int(torch.multinomial(probs, 1).item())

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cut = torch.searchsorted(cdf, torch.tensor(p, device=logits.device)).item()
    cut = max(int(cut), 0)
    keep = cut + 1
    sp = sorted_probs[:keep]
    si = sorted_idx[:keep]
    sp = sp / sp.sum()
    pick = int(torch.multinomial(sp, 1).item())
    return int(si[pick].item())


def _build_input_text(tokenizer, prompt, system, use_chat_template):
    if not use_chat_template:
        return prompt
    prompt_template = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt_template.format(system=system, input_text=prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True, help="Qwen2模型路径")
    ap.add_argument("--prompt", type=str, default="", help="输入提示词")
    ap.add_argument("--system", type=str, default="You are a helpful assistant.", help="系统提示词")
    ap.add_argument("--no-template", "--no-chat-template", dest="no_template", action="store_true", help="不使用聊天模板")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="最大生成token数")
    ap.add_argument("--temperature", type=float, default=0.7, help="温度系数")
    ap.add_argument("--top-p", type=float, default=0.9, help="核采样")
    ap.add_argument("--top-k", type=int, default=0, help="Top-K采样")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="重复惩罚")
    ap.add_argument("--do-sample", action="store_true", default=False, help="是否采样")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--device", type=str, default="auto", help="运行设备 cuda/cpu")
    ap.add_argument("--dtype", type=str, default="bf16", help="精度 bf16/fp16/fp32")
    ap.add_argument("--prefill-chunk-size", type=int, default=0)
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--input-ids", type=str, default="")
    ap.add_argument("--eos-id", type=int, default=-1)
    ap.add_argument("--loop-num", type = int, default = 73, help="和内部保持一致")
    args = ap.parse_args()

    prompt = args.prompt
    if not prompt.strip():
        prompt = sys.stdin.read()
    prompt = prompt.strip()

    if args.seed:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device.strip().lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device.strip().lower())

    dtype_s = args.dtype.strip().lower()
    if dtype_s in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif dtype_s in ("fp16", "float16", "half"):
        dtype = torch.float16
    elif dtype_s in ("fp32", "float32"):
        dtype = torch.float32
    else:
        return 1

    gen_cfg = _load_generation_config(args.model_path)
    cfg = Qwen2Config.from_model_dir(args.model_path)
    eos_ids = _parse_eos_ids(gen_cfg, int(args.eos_id))
    if "temperature" in gen_cfg:
        args.temperature = float(gen_cfg["temperature"])
    if "top_p" in gen_cfg:
        args.top_p = float(gen_cfg["top_p"])
    if "top_k" in gen_cfg:
        args.top_k = int(gen_cfg["top_k"])
    if "repetition_penalty" in gen_cfg:
        args.repetition_penalty = float(gen_cfg["repetition_penalty"])

    backbone = Qwen2Model(cfg).to(device=device, dtype=dtype)
    head = Qwen2LMHeadPart(cfg).to(device=device, dtype=dtype)
    load_weights_safetensors_direct_to_gpu(
        backbone,
        args.model_path,
        device=device,
        dtype=dtype,
        key_strip_prefix="model.",
    )
    load_weights_safetensors_direct_to_gpu(
        head,
        args.model_path,
        device=device,
        dtype=dtype,
    )
    if cfg.tie_word_embeddings:
        head.lm_head.weight = backbone.embed_tokens.weight
    backbone.eval()
    head.eval()

    if args.no_tokenizer:
        ids = [int(x.strip()) for x in args.input_ids.split(",") if x.strip()]
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        tokenizer = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
        input_text = _build_input_text(
            tokenizer=tokenizer,
            prompt=prompt,
            system=args.system,
            use_chat_template=(not args.no_template),
        )
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)

    kv = [None] * cfg.num_hidden_layers
    start_pos = 0
    
    # warm up 
    if device.type == "cuda":
        with torch.inference_mode():
            for _ in range(10):
                a = torch.randn(512, 512, device=device, dtype=dtype)
                b = torch.randn(512, 512, device=device, dtype=dtype)
                _ = a @ b
                _w_ids = input_ids[:, :1].contiguous()
                _w_h, _ = backbone(_w_ids, kv_caches=kv, start_pos=0, is_prefill=1)
                while _w_h.dim() < 3:
                    _w_h = _w_h.unsqueeze(0)
                _ = head(_w_h)
        torch.cuda.synchronize()
    #----

    _ib_init_once()  # ib 初始化

    prefill_cache = None
    dummy_data = torch.randn(1,3584, device = "cuda", dtype=torch.float16)
    ib_send_buffer = []
    #t_prefill0 = time.perf_counter()
    t_prefill0 = None
    with torch.inference_mode():
        logits = None
        total = int(input_ids.shape[1])
        torch.cuda.synchronize()
        for off in range(total):
            step_ids = input_ids[:, off : off + 1].contiguous()
            hidden, kv = backbone(step_ids, kv_caches=kv, start_pos=start_pos, is_prefill=1)
            ib_send_buffer.append(hidden)
            start_pos += 1
        for _ in range(72):
            global send_recv_idx 
            ib_send_buffer.append(dummy_data)
        ib_send_in = torch.cat(ib_send_buffer)
        torch.cuda.synchronize()
        
        t_prefill0 = time.perf_counter()
        prefill_cache = ib_send_recv(ib_send_in, data_num=total + 72, clear_kv=0, all_dummy=0, is_prefill=1)
        torch.cuda.synchronize()
        print("prefill number: ", total + 72)
        
        
        logits = head(prefill_cache)
        torch.cuda.synchronize()
        send_recv_idx = send_recv_idx + total + 72
        assert logits is not None

    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_prefill0
    print("prefill_s: ", prefill_s)

    generated = []
    if args.do_sample:
        next_id = sample_token(
            logits[0, -1],
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=float(args.repetition_penalty),
            generated=generated,
        )
    else:
        next_id = int(torch.argmax(logits[0, -1]).item())
    generated.append(int(next_id))
    print("输出的字：", tokenizer.decode([int(next_id)], skip_special_tokens=True))
    
    t_decode0 = time.time()
    if eos_ids is None or int(next_id) not in eos_ids:
        last_token = int(next_id)
        for step in range(1, args.max_new_tokens):
            if int(start_pos) >= int(cfg.max_position_embeddings):
                break
            step_ids = torch.tensor([[last_token]], device=device, dtype=torch.long)
            with torch.inference_mode():
                hidden, kv = backbone(step_ids, kv_caches=kv, start_pos=start_pos, is_prefill=0)
                ib_recv_t = ib_send_recv(hidden, data_num=1, clear_kv=0, all_dummy=0, is_prefill=0)
                send_recv_idx += 73
                step_logits = head(ib_recv_t)
            start_pos += 1
            if args.do_sample:
                next_id = sample_token(
                    step_logits[0, -1],
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    repetition_penalty=float(args.repetition_penalty),
                    generated=generated,
                )
            else:
                next_id = int(torch.argmax(step_logits[0, -1]).item())
            

            generated.append(int(next_id))
            last_token = int(next_id)
            if eos_ids is not None and last_token in eos_ids:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_s = time.time() - t_decode0

    gen_tokens = len(generated)
    prompt_tokens = int(input_ids.shape[1])
    tps_decode = (gen_tokens / decode_s) if decode_s > 0 else float("inf")
    tps_prefill = (prompt_tokens / prefill_s) if prefill_s > 0 else float("inf")
    

    # 填充剩下的 loop_num 
    loop_num = args.loop_num
    if loop_num > send_recv_idx :
        offset = loop_num -  send_recv_idx
        dummy_data = torch.randn(1, 3584, device= "cuda", dtype= torch.float16)
        for _ in range(offset):
            ib_send_recv(dummy_data, data_num=1, clear_kv=0, all_dummy=1, is_prefill=1)

    print(
        "[benchmark] "
        f"prefill: {prompt_tokens} tokens, time {prefill_s:.3f} s, "
        f"throughput {tps_prefill:.2f} tps; "
        f"generation: {gen_tokens} tokens, time {decode_s:.3f} s, "
        f"throughput {tps_decode:.2f} tps",
        file=sys.stderr,
        flush=True,
    )


    print(tokenizer.decode(generated, skip_special_tokens=True))
    return 0

if __name__ == "__main__":
    main()
