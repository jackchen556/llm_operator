import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib
from tqdm import tqdm

_FCA = importlib.import_module(os.environ.get("FCA_MHA_PYMODULE", "pyextlxlink._C"))  # type: ignore


def InitializeLink(cfg):
    _FCA.initialize(FCA_LINK_IDS, cfg.hidden_size, cfg.hidden_size)


def _parse_xlink_ids(s):
    s = (s or "").strip()
    s = s.replace(" ", ",")
    return [int(x.strip()) for x in s.split(",") if x.strip()]


FCA_LINK_IDS = [0]
replace_layer = 0


sys.path.insert(0, "/data/chenjiaken/lynxkit-master/demo/golden_model_test/test/")
from model_load_lynxkit import model_load  # type: ignore
from types import SimpleNamespace


def _require_openai_humaneval_parquet_dir(dataset_dir: str) -> str:
    """
    仅支持 HuggingFace datasets 导出的 parquet 结构：
      {dataset_dir}/openai_humaneval/test-*.parquet
    """
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    parquet_dir = os.path.join(dataset_dir, "openai_humaneval")
    if not os.path.isdir(parquet_dir):
        raise ValueError(f"未找到目录: {parquet_dir}")
    for fn in os.listdir(parquet_dir):
        if fn.startswith("test-") and fn.endswith(".parquet") and os.path.isfile(os.path.join(parquet_dir, fn)):
            return parquet_dir
    raise ValueError(f"未找到 test-*.parquet: {parquet_dir}")


def _load_humaneval_samples(dataset_dir: str) -> List[dict]:
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    parquet_dir = _require_openai_humaneval_parquet_dir(dataset_dir)
    parquet_files = sorted(
        os.path.join(parquet_dir, fn)
        for fn in os.listdir(parquet_dir)
        if fn.startswith("test-") and fn.endswith(".parquet") and os.path.isfile(os.path.join(parquet_dir, fn))
    )
    # 你当前只有一个 test-00000-of-00001.parquet，但这里保持通用性
    rows: List[dict] = []

    def _maybe_raise_lfs_pointer(fp: str, err: Exception) -> None:
        try:
            st = os.stat(fp)
            # LFS 指针文件通常非常小（几百字节）
            if st.st_size <= 4096:
                with open(fp, "rb") as f:
                    head = f.read(256)
                if b"git-lfs.github.com/spec/v1" in head or head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(
                        "检测到 Git LFS 指针文件，实际 parquet 未被拉取。"
                        f"请在仓库目录执行 `git lfs pull` 或重新下载数据集文件后重试。file={fp}"
                    ) from err
        except ValueError:
            raise
        except Exception:
            return

    try:
        import pyarrow.parquet as pq  # type: ignore

        for fp in parquet_files:
            try:
                table = pq.read_table(fp)
            except Exception as e:
                _maybe_raise_lfs_pointer(fp, e)
                raise
            rows.extend(table.to_pylist())
    except Exception:
        try:
            import pandas as pd  # type: ignore

            for fp in parquet_files:
                try:
                    df = pd.read_parquet(fp)
                except Exception as e:
                    _maybe_raise_lfs_pointer(fp, e)
                    raise
                rows.extend(df.to_dict(orient="records"))
        except Exception as e:
            raise ValueError(f"读取 parquet 失败（需要 pyarrow 或 pandas）: {parquet_dir}: {e}") from e
    samples = [r for r in rows if isinstance(r, dict)]
    if not samples:
        raise ValueError(f"parquet 数据集为空: {parquet_dir}")
    return samples


def _run_python_snippet(code: str, *, timeout_s: float = 3.0) -> Tuple[bool, str]:
    fp = os.path.abspath("openai_humaneval_prog.py")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(code)
        f.write("\n")
    try:
        cp = subprocess.run(
            [sys.executable, "-I", fp],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired:
        return False, f"Timeout({timeout_s}s)"
    if cp.returncode == 0:
        return True, ""
    msg = (cp.stderr or "").strip() or (cp.stdout or "").strip()
    try:
        dump_path = os.path.abspath("openai_humaneval_fail_last.py")
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(code)
            f.write("\n")
    except Exception:
        pass
    return False, (msg if msg else f"Non-zero exit code: {cp.returncode}")


def _generate_completion_for_prompt(model, tokenizer, cfg, device, args, prompt_text: str) -> str:
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
    if args.eos_id >= 0:
        eos_id = args.eos_id
    eos_ids = set([int(eos_id)]) if eos_id is not None else None

    kv = [None] * cfg.num_hidden_layers
    start_pos = 0
    generated: List[int] = []

    with torch.inference_mode():
        logits = None
        total = int(input_ids.shape[1])
        for off in range(total):
            step_ids = input_ids[:, off : off + 1].contiguous()
            logits, kv = model(step_ids, kv_caches=kv, start_pos=start_pos)
            start_pos += 1
        assert logits is not None

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

        if eos_ids is None or int(next_id) not in eos_ids:
            last_token = int(next_id)
            for _step in range(1, int(args.max_new_tokens)):
                if int(start_pos) >= int(cfg.max_position_embeddings):
                    break
                step_ids = torch.tensor([[last_token]], device=device, dtype=torch.long)
                step_logits, kv = model(step_ids, kv_caches=kv, start_pos=start_pos)
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

    text = tokenizer.decode(generated, skip_special_tokens=True)
    text = text.replace("\r\n", "\n")
    if "```" in text:
        text = text.split("```", 1)[0]

    lines = text.splitlines()
    kept: List[str] = []
    started = False
    for ln in lines:
        if not started:
            if ln.strip() == "":
                continue
            started = True
        if ln.strip() == "":
            kept.append(ln)
            continue
        if ln[:1].isspace():
            kept.append(ln)
            continue
        break
    kept_text = "\n".join(kept).rstrip() + "\n"

    prompt_part = str(prompt_text).replace("\r\n", "\n").rstrip() + "\n"
    comp_lines = kept_text.splitlines()

    for _ in range(min(200, len(comp_lines) + 5)):
        candidate = "\n".join(comp_lines).rstrip() + "\n"
        if not candidate.strip():
            candidate = "    pass\n"
        try:
            ast.parse(prompt_part + candidate)
            return candidate
        except SyntaxError:
            if comp_lines:
                comp_lines.pop()
            else:
                return "    pass\n"
    # fallback
    return "    pass\n"


def run_humaneval_and_print(model, tokenizer, cfg, device, args, dataset_dir: str = "./openai_humaneval/") -> float:
    samples = _load_humaneval_samples(dataset_dir)
    required = ("task_id", "prompt", "test", "entry_point")
    seen = set()
    for i, s in enumerate(samples, start=1):
        miss = [k for k in required if k not in s]
        if miss:
            raise ValueError(f"缺少字段 {miss}: idx={i}")
        tid = str(s["task_id"])
        if tid in seen:
            raise ValueError(f"task_id 重复: {tid}")
        seen.add(tid)

    passed = 0
    total = len(samples)
    for i, s in enumerate(samples, start=1):
        tid = str(s["task_id"])
        prompt = str(s["prompt"]).replace("\r\n", "\n")
        test = str(s["test"]).replace("\r\n", "\n")
        entry_point = str(s["entry_point"])
        print(f"\n===== {i}/{total} | {tid} | entry_point={entry_point} =====")
        print(prompt.rstrip())
        completion = _generate_completion_for_prompt(model, tokenizer, cfg, device, args, prompt)
        # 强制分段之间有明确换行，避免出现 `... )import ...` 的黏连语法错误
        prompt_part = prompt.rstrip() + "\n"
        completion_part = completion.lstrip("\n").rstrip() + "\n"
        test_body = test.lstrip("\n").rstrip() + "\n"

        # 确保测试会被执行：有些数据/变体可能只给出 `check()` 定义与断言，但没调用入口。
        # 这里在末尾补一个统一的 runner（若 test 已经包含 main/调用则不重复补）。
        needs_runner = ("def check" in test_body) and ("check(" not in test_body or "__main__" not in test_body)
        runner = ""
        if needs_runner:
            runner = (
                "\n\nif __name__ == \"__main__\":\n"
                f"    check({entry_point})\n"
            )

        test_part = "\n" + test_body + runner
        code = prompt_part + completion_part + test_part
        ok, err = _run_python_snippet(code, timeout_s=3.0)
        if ok:
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
            print(f"ERROR: {err}")

    score = passed / total if total else 0.0
    print("\n===== SUMMARY =====")
    print(f"passed: {passed}/{total}")
    print(f"score: {score:.4f}")
    return float(score)



class ApuLoader:
    def __init__(
        self,
        xdma_ids_mha=None,
        model_path_mha=None,
        model_paths_mha=None,
        *,
        device_config_file="",
        load_version=1,
    ):
        self.xdma_ids_mha = xdma_ids_mha
        self.model_path_mha = model_path_mha
        self.model_paths_mha = list(model_paths_mha) if model_paths_mha is not None else None
        self.device_config_file = device_config_file
        self.load_version = int(load_version)
        self.mha_loaders: list = []

    def load_lynxlink_mha(self, loop_count=1):
        if self.xdma_ids_mha is None:
            print("No MHA ids provided; nothing to initialize")
            return
        if isinstance(self.xdma_ids_mha, str):
            mha_ids = [id.strip() for id in self.xdma_ids_mha.split(",")]
        elif isinstance(self.xdma_ids_mha, (int, list, tuple)):
            mha_ids = [str(self.xdma_ids_mha)] if isinstance(self.xdma_ids_mha, int) else [str(id) for id in self.xdma_ids_mha]
        else:
            mha_ids = []
        if not mha_ids:
            print("No valid MHA ids found")
            return
        self.mha_loaders = []
        for i, mha_id in enumerate(mha_ids):
            model_path = ""
            if self.model_paths_mha is not None:
                if i < len(self.model_paths_mha) and str(self.model_paths_mha[i]).strip():
                    model_path = str(self.model_paths_mha[i]).strip()
                else:
                    print(f"No valid MHA Model Path for index {i}")
                    return
            else:
                if self.model_path_mha and str(self.model_path_mha).strip():
                    model_path = str(self.model_path_mha).strip()
                else:
                    print("No valid MHA Model Path")
                    return
            args_mha = SimpleNamespace(
                model_path=model_path,
                lynxlink_ids=mha_id,
                loop_count=loop_count,
                load_version=self.load_version,
                device_config_file=self.device_config_file,
            )
            loader_mha = model_load(args_mha)
            if loader_mha == 1:
                print(f"MHA model {i} (ID: {mha_id}) load failed")
                return
            self.mha_loaders.append(loader_mha)
            print(f"MHA model {i} (ID: {mha_id}) loaded successfully")

    def run_lynxlink_mha(self, interleave_size=512, reduce_max_num=0):
        if not hasattr(self, "mha_loaders") or not self.mha_loaders:
            print("No MHA models loaded; nothing to execute")
            return
        print(
            f"[apu] download_execbd_task interleave_size={int(interleave_size)} reduce_max_num={int(reduce_max_num)}",
            file=sys.stderr,
            flush=True,
        )
        for i, loader_mha in enumerate(self.mha_loaders):
            ret = loader_mha.download_execbd_task(interleave_size, reduce_max_num)
            if ret != 0:
                print(f"MHA execbd task {i} failed (ret={ret})")
                return
            print(f"MHA execbd task {i} executed successfully")


APU_LOADER = None


def _build_input_text(tokenizer, prompt, system, use_chat_template):
    if not use_chat_template:
        return prompt
    prompt_template = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt_template.format(system=system, input_text=prompt)


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


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position, base):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len, device, dtype):
        if self._cached_cos is not None and self._cached_len >= seq_len:
            if self._cached_cos.device == device and self._cached_cos.dtype == dtype:
                return
        if self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_len = seq_len

    def apply_rotary(self, x, positions):
        assert x.shape[-1] == self.head_dim
        device = x.device
        dtype = x.dtype
        seq_len = int(positions.max().item()) + 1
        self._build_cache(seq_len=seq_len, device=device, dtype=dtype)
        cos = self._cached_cos.index_select(0, positions)
        sin = self._cached_sin.index_select(0, positions)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return (x * cos) + (_rotate_half(x) * sin)


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor


class Qwen2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        assert self.head_dim * self.num_heads == self.hidden_size
        assert self.num_heads % self.num_kv_heads == 0
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position=cfg.max_position_embeddings,
            base=cfg.rope_theta,
        )

    def forward(
        self,
        x,
        *,
        positions,
        kv_cache,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rotary.apply_rotary(q, positions)
        k = self.rotary.apply_rotary(k, positions)

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache.k.shape[2]
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)
        new_cache = KVCache(k=k, v=v)

        if self.num_groups != 1:
            s = k.shape[2]
            k = (
                k[:, :, None, :, :]
                .expand(b, self.num_kv_heads, self.num_groups, s, self.head_dim)
                .reshape(b, self.num_heads, s, self.head_dim)
            )
            v = (
                v[:, :, None, :, :]
                .expand(b, self.num_kv_heads, self.num_groups, s, self.head_dim)
                .reshape(b, self.num_heads, s, self.head_dim)
            )

        s = k.shape[2]

        def _sdpa(q_, k_, v_, *, attn_mask, is_causal):
            out32 = F.scaled_dot_product_attention(
                q_.float(),
                k_.float(),
                v_.float(),
                attn_mask=(attn_mask.float() if attn_mask is not None else None),
                is_causal=is_causal,
            )
            return out32.to(dtype=q_.dtype)

        if kv_cache is None:
            out = _sdpa(q, k, v, attn_mask=None, is_causal=True)
        else:
            if t == 1:
                out = _sdpa(q, k, v, attn_mask=None, is_causal=False)
            else:
                q_pos = torch.arange(past_len, past_len + t, device=x.device, dtype=torch.long)
                k_pos = torch.arange(s, device=x.device, dtype=torch.long)
                causal = (k_pos[None, :] <= q_pos[:, None])
                attn_mask = torch.where(
                    causal,
                    torch.tensor(0.0, device=x.device, dtype=q.dtype),
                    torch.tensor(float("-inf"), device=x.device, dtype=q.dtype),
                )
                out = _sdpa(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out, new_cache


class Qwen2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        if x.dtype == torch.float16:
            g = self.gate_proj(x).float()
            u = self.up_proj(x).float()
            h = F.silu(g) * u
            out = self.down_proj(h.to(dtype=torch.float16)).float()
            return out.to(dtype=torch.float16)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    _next_layer_id = 0
    _fused = None
    _hidden = 0

    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen2MLP(cfg)
        self.layer_id = Qwen2DecoderLayer._next_layer_id
        Qwen2DecoderLayer._next_layer_id += 1

        if Qwen2DecoderLayer._fused is None:
            if _FCA is not None and hasattr(_FCA, "FusedImpl"):
                hidden = int(cfg.hidden_size)
                Qwen2DecoderLayer._fused = _FCA.FusedImpl(
                    FCA_LINK_IDS,
                    hidden,
                    hidden,
                    1,
                    1,
                    int(cfg.num_hidden_layers),
                )
                Qwen2DecoderLayer._hidden = hidden

    def forward(
        self,
        x,
        *,
        positions,
        kv_cache,
        use_cache=True,
    ):
        residual = x
        presents = None

        if self.layer_id < replace_layer:
            fca_in = residual.to(torch.float16).contiguous()
            mha_out = torch.empty_like(fca_in)
            Qwen2DecoderLayer._fused.forward(
                hidden_states=fca_in,
                output=mha_out,
                layer_id=int(self.layer_id),
            )
            w16 = self.self_attn.o_proj.weight.to(torch.float16)
            b = self.self_attn.o_proj.bias
            attention_output = F.linear(
                mha_out,
                w16,
                b.to(torch.float16) if b is not None else None,
            ).to(dtype=residual.dtype)
        else:
            hidden_states = self.input_layernorm(x)
            attention_output, presents = self.self_attn(
                hidden_states,
                positions=positions,
                kv_cache=(kv_cache if use_cache else None),
            )

        hidden_states = residual + attention_output
        residual2 = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual2 + hidden_states
        if use_cache:
            return hidden_states, presents
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nl = int(cfg.num_hidden_layers)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg) for _ in tqdm(range(nl), desc="load model")]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        input_ids,
        *,
        kv_caches=None,
        start_pos=0,
    ):
        b, t = input_ids.shape
        x = self.embed_tokens(input_ids)
        #import pdb; pdb.set_trace()
        pos = torch.arange(start_pos, start_pos + t, device=input_ids.device, dtype=torch.long)
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        new_caches = []
        for i in range(len(self.layers)):
            x, c = self.layers[i](x, positions=pos, kv_cache=kv_caches[i], use_cache=True)  # type: ignore[misc]
            new_caches.append(c)
        x = self.norm(x)
        return x, new_caches


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        *,
        kv_caches=None,
        start_pos=0,
    ):
        h, caches = self.model(input_ids, kv_caches=kv_caches, start_pos=start_pos)
        #import pdb; pdb.set_trace()
        logits = self.lm_head(h)
        return logits, caches


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
):
    from safetensors.torch import load_file  # type: ignore

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
            ok = _set_module_tensor_as_param(model, k, t)
            if ok:
                missing.discard(k)
            else:
                unexpected.append(k)

    if "lm_head.weight" in missing and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.lm_head.weight = model.model.embed_tokens.weight   
        missing.remove("lm_head.weight")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--apu-model-path", type=str, default="")
    ap.add_argument("--apu-model-path-0", type=str, default="")
    ap.add_argument("--apu-model-path-1", type=str, default="")
    ap.add_argument("--apu-model-path-2", type=str, default="")
    ap.add_argument("--apu-model-path-3", type=str, default="")
    ap.add_argument("--apu-xlink-ids", type=str, default="")
    ap.add_argument("--apu-device-config-file", type=str, default="")
    ap.add_argument("--apu-load-version", type=int, default=1)
    ap.add_argument("--replace-layer", type=int, default=0)
    ap.add_argument("--apu-loop-count", type=int, default=0xFFFFFFFF)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--system", type=str, default="You are a helpful assistant.")
    ap.add_argument("--no-template", "--no-chat-template", dest="no_template", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--prefill-chunk-size", type=int, default=0)
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--input-ids", type=str, default="")
    ap.add_argument("--eos-id", type=int, default=-1)
    ap.add_argument("--validate-dataset", action="store_true", default=False)
    args = ap.parse_args()

    global FCA_LINK_IDS, replace_layer
    FCA_LINK_IDS = _parse_xlink_ids(args.apu_xlink_ids) if args.apu_xlink_ids.strip() else [0]
    replace_layer = max(0, int(args.replace_layer))
    fca_mha_fuse_interleave = 0
    if hasattr(_FCA, "load_config"):
        fca_mha_fuse_interleave = int(_FCA.load_config())

    global APU_LOADER
    per_paths = [
        args.apu_model_path_0.strip(),
        args.apu_model_path_1.strip(),
        args.apu_model_path_2.strip(),
        args.apu_model_path_3.strip(),
    ]
    have_per_paths = any(p for p in per_paths)
    if args.apu_xlink_ids.strip() and (args.apu_model_path.strip() or have_per_paths):
        APU_LOADER = ApuLoader(
            xdma_ids_mha=args.apu_xlink_ids.strip(),
            model_path_mha=(args.apu_model_path.strip() if args.apu_model_path.strip() else None),
            model_paths_mha=(per_paths if have_per_paths else None),
            device_config_file=args.apu_device_config_file,
            load_version=int(args.apu_load_version),
        )
    prompt = args.prompt
    if not prompt.strip():
        if not bool(args.validate_dataset):
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

    Qwen2DecoderLayer._next_layer_id = 0
    Qwen2DecoderLayer._fused = None
    Qwen2DecoderLayer._hidden = 0
    model = Qwen2ForCausalLM(cfg)
    load_weights_safetensors_direct_to_gpu(
        model,
        args.model_path,
        device=device,
        dtype=dtype,
    )
    model.eval()

    if APU_LOADER is not None:
        APU_LOADER.load_lynxlink_mha(loop_count=int(args.apu_loop_count))
        if fca_mha_fuse_interleave == 0 and hasattr(_FCA, "load_config"):
            fca_mha_fuse_interleave = int(_FCA.load_config())
        APU_LOADER.run_lynxlink_mha(
            interleave_size=int(fca_mha_fuse_interleave) * 512,
            reduce_max_num=0,
        )

    InitializeLink(cfg)

    if args.no_tokenizer:
        ids = [int(x.strip()) for x in args.input_ids.split(",") if x.strip()]
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        eos_id = args.eos_id if args.eos_id >= 0 else None
        tokenizer = None
    else:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
        input_text = _build_input_text(
            tokenizer=tokenizer,
            prompt=prompt,
            system=args.system,
            use_chat_template=(not args.no_template),
        )
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
        if args.eos_id >= 0:
            eos_id = args.eos_id

    if bool(args.validate_dataset):
        if tokenizer is None:
            print("[dataset] 需要 tokenizer（请不要加 --no-tokenizer）", file=sys.stderr, flush=True)
            return 2
        try:
            run_humaneval_and_print(model, tokenizer, cfg, device, args, dataset_dir="./openai_humaneval/")
        except Exception as e:
            print(f"[dataset] 运行失败: {e}", file=sys.stderr, flush=True)
            return 2
        return 0

    kv = [None] * cfg.num_hidden_layers
    start_pos = 0

    t_prefill0 = time.time()
    with torch.inference_mode():
        logits = None
        total = int(input_ids.shape[1])
        for off in range(total):
            step_ids = input_ids[:, off : off + 1].contiguous()
            logits, kv = model(step_ids, kv_caches=kv, start_pos=start_pos)
            start_pos += 1
        assert logits is not None

    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.time() - t_prefill0

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

    t_decode0 = time.time()
    if eos_ids is None or int(next_id) not in eos_ids:
        last_token = int(next_id)
        for step in range(1, args.max_new_tokens):
            if int(start_pos) >= int(cfg.max_position_embeddings):
                break
            step_ids = torch.tensor([[last_token]], device=device, dtype=torch.long)
            with torch.inference_mode():
                step_logits, kv = model(step_ids, kv_caches=kv, start_pos=start_pos)
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
    print(
        "[benchmark] "
        f"prefill: {prompt_tokens} tokens, time {prefill_s:.3f} s, "
        f"throughput {tps_prefill:.2f} tps; "
        f"generation: {gen_tokens} tokens, time {decode_s:.3f} s, "
        f"throughput {tps_decode:.2f} tps",
        file=sys.stderr,
        flush=True,
    )

    if tokenizer is None:
        from transformers import AutoTokenizer  # type: ignore
        tok2 = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
        print(tok2.decode(generated, skip_special_tokens=True))
        return 0

    print(tokenizer.decode(generated, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    main()
