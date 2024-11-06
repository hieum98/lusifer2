from typing import Optional
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend
import einops
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


class FFWithAddedTokens(nn.Module):
    def __init__(
            self, 
            in_dim: int,
            out_dim: int, 
            num_added_tokens: int=1, 
            model_dtype=torch.bfloat16
            ):
        super().__init__()
        self.dtype = model_dtype
        self.ff = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )
        self.num_added_tokens = num_added_tokens
        if num_added_tokens > 0:
            print(f'Adding {num_added_tokens} trainable tokens to FFWithAddedTokens')
            self.added_tokens = nn.Parameter(torch.randn(num_added_tokens, out_dim))
    
    def forward(self, x, **kwargs):
        # Cast to correct device type
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            x = self.ff(x)
            if self.num_added_tokens > 0:
                added_tokens = einops.repeat(self.added_tokens, 'n d -> b n d', b=x.size(0))
                x = torch.cat([x, added_tokens], dim=1) # (b, n + n_a, d)
            x = x.to(dtype=self.dtype)
        return x


class ProbalisticTokenEmbedding(nn.Embedding):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(tokens)
        return torch.matmul(tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        torch.nn.init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class EmbeddingTable(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int, 
            vocab_size: Optional[int],
            padding_idx: Optional[int]=None,
            llm_embedding: Optional[nn.Embedding]=None,
            model_dtype=torch.bfloat16
            ) -> None:
        super().__init__()
        self.dtype = model_dtype
        self.ff = nn.Sequential(
            nn.Linear(in_dim, vocab_size),
            nn.Softmax(dim=-1),
        )
        print(f'Initializing EmbeddingTable with vocab_size={vocab_size}, out_dim={out_dim}, padding_idx={padding_idx}')
        self.embedding = ProbalisticTokenEmbedding(vocab_size, out_dim, padding_idx=padding_idx)
        if llm_embedding is not None:
            # Initialize from LLM embedding
            assert llm_embedding.weight.size() == self.embedding.weight.size(), "embedding sizes must match!"
            self.embedding.weight.data = llm_embedding.weight.data

    def forward(self, x, **kwargs):
        # Cast to correct device type
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            x = self.ff(x) # (b, n, vocab_size)
            x = self.embedding(x)
            x = x.to(dtype=self.dtype) # (b, n, d)
        return x


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


class LatentAttention(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_latents: int,
            latent_dim: int,
            cross_heads: int=8,
            model_dtype=torch.bfloat16
            ) -> None:
        super().__init__()
        self.cross_attn = PreNorm(
            latent_dim,
            Attention(latent_dim, hidden_dim, heads=cross_heads, dim_head=hidden_dim),
            context_dim=hidden_dim
        )
        self.cross_ff = PreNorm(latent_dim, FeedForward(latent_dim))
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.dtype = model_dtype

    def forward(self, hiddens):
        # Cast to correct device type
        with torch.autocast(device_type=hiddens.device.type, dtype=torch.float32):
            b, *_, device = *hiddens.shape, hiddens.device
            x = einops.repeat(self.latents, 'n d -> b n d', b=b)
            hiddens = self.cross_attn(x, context=hiddens, mask=None)
            hiddens = self.cross_ff(hiddens) + hiddens # (b, n, d)
        hiddens = hiddens.to(dtype=self.dtype)
        return hiddens
    