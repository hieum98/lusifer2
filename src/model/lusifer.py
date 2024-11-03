from contextlib import nullcontext
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import date
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import mteb

from src.model.bidirectional_modelings.modeling_bidirectional_mistral import BidirectionalMistralForCausalLM
from src.model.bidirectional_modelings.modeling_bidirectional_llama import BidirectionalLlamaForCausalLM
from src.model.bidirectional_modelings.modeling_bidirectional_phi3 import BidirectionalPhi3ForCausalLM
from src.model.bidirectional_modelings.modeling_bidirectional_phi import BidirectionalPhiForCausalLM
from src.model.bidirectional_modelings.modeling_bidirectional_qwen2 import BidirectionalQwen2ForCausalLM
from src.model.bidirectional_modelings.modeling_bidirectional_gemma2 import BidirectionalGemma2ForCausalLM
from src.model.bidirectional_modelings.modeling_nv_embed import LatentAttentionModel
from src.model.bidirectional_modelings.config_nvembed import LatentAttentionConfig
from src.model.connection_modules import FFWithAddedTokens, EmbeddingTable
from src.model.utils import find_all_linear_names
from src.special_tokens import SPECIAL_TOKENS


class Lusifer(nn.Module):
    def __init__(
            self,
            universal_learner_name_or_path: str,
            encoder_name_or_path: str,
            universal_learner_backbone_type: str = 't5',
            encoder_backbone_type: str = 'mistral',
            is_freeze_universal_learner: bool = True,
            is_freeze_encoder: bool = False,
            connection_type: str = 'ff',
            num_added_tokens: int = 0,
            pooling_method: str='mean',
            encoder_lora_name: str = 'encoder_lora',
            universal_learner_lora_name: str = 'universal_learner_lora',
            encoder_lora_target_modules: Union[str, List[str]] = "all",
            universal_learner_lora_target_modules: Union[str, List[str]] = "all",
            loar_r: int = 16,
            lora_alpha: int = 32,
            dropout: float = 0.1,
            attn_implementation: str = 'flash_attention_2',
            model_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        
        super().__init__()
        self.hprams = {
            'universal_learner_name_or_path': universal_learner_name_or_path,
            'encoder_name_or_path': encoder_name_or_path,
            'universal_learner_backbone_type': universal_learner_backbone_type,
            'encoder_backbone_type': encoder_backbone_type,
            'is_freeze_universal_learner': is_freeze_universal_learner,
            'is_freeze_encoder': is_freeze_encoder,
            'connection_type': connection_type,
            'num_added_tokens': num_added_tokens,
            'pooling_method': pooling_method,
            'encoder_lora_name': encoder_lora_name,
            'universal_learner_lora_name': universal_learner_lora_name,
            'encoder_lora_target_modules': encoder_lora_target_modules,
            'universal_learner_lora_target_modules': universal_learner_lora_target_modules,
            'loar_r': loar_r,
            'lora_alpha': lora_alpha,
            'dropout': dropout,
            'attn_implementation': attn_implementation,
            'model_dtype': model_dtype,
        }
        if attn_implementation == "flash_attention_2":
            model_dtype = torch.bfloat16
            self.model_dtype = model_dtype
        self.pooling_method = pooling_method
        self.mteb_model_meta = mteb.ModelMeta(
            name='Lusifer',
            revision='dev',
            release_date=date.today().strftime("%Y-%m-%d"),
            languages=None,
        )

        # Universal Learner
        self.universal_learner_tokenizer = self.create_tokenizer(universal_learner_name_or_path, universal_learner_backbone_type)
        self.universal_learner = self.create_transformer(
            model_name_or_path=universal_learner_name_or_path,
            use_lora=True if universal_learner_lora_name else False,
            lora_r=loar_r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            adapter_name=universal_learner_lora_name,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
            target_modules=universal_learner_lora_target_modules,
        )
        if is_freeze_universal_learner and universal_learner_lora_name != None:
            print("Warning: You are freezing the univeral learner but the model has an adapter. Set is_freeze_universal_learner=False to train the adapter.")
            is_freeze_universal_learner = False
        if is_freeze_universal_learner:
            self.universal_learner.requires_grad_(False)
        elif universal_learner_lora_name == None:
            # Always cast the model to the float32 when it is fully trainable to avoid the error in the mixed precision training and can compartible with grad_clip
            self.universal_learner = self.universal_learner.to(dtype=torch.float32)
        self.universal_learner_dim = self.universal_learner.config.hidden_size

        # Encoder
        self.encoder_tokenizer = self.create_tokenizer(encoder_name_or_path, encoder_backbone_type)
        self.encoder = self.create_transformer(
            model_name_or_path=encoder_name_or_path,
            is_llm_bidirectional=True,
            backbone_type=encoder_backbone_type,
            use_lora=True if encoder_lora_name else False,
            lora_r=loar_r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            adapter_name=encoder_lora_name,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
            target_modules=encoder_lora_target_modules,
        )
        if encoder_backbone_type == 'nvidia/NV-Embed-v2':
            print("Loading latent attention model of NV-Embed-v2")
            self.laten_attention_model, loading_info = LatentAttentionModel.from_pretrained('Hieuman/nvembed-v2-latent-attention', output_loading_info=True)
            print(f"Latent attention model loading info: {loading_info}")
            self.laten_attention_model.requires_grad_(False)
        self.encoder_dim = self.encoder.config.hidden_size
        if is_freeze_encoder and encoder_lora_name != None:
            print("Warning: You are freezing the encoder but the model has an adapter. Set is_freeze_encoder=False to train the adapter.")
            is_freeze_encoder = False
        if is_freeze_encoder:
            self.encoder.requires_grad_(False)
        self.encoder_backbone_type = encoder_backbone_type

        # Connector
        self.num_added_tokens = num_added_tokens
        if self.num_added_tokens == 0 and connection_type == 'attn':
            print("Warning: You are using attention connection but num_added_tokens is 0. Setting the connection type to ff.")
            connection_type = 'ff'
        self.connection_type = connection_type
        if connection_type == 'ff':
            self.connection_module = FFWithAddedTokens(
                in_dim=self.universal_learner_dim,
                out_dim=self.encoder_dim,
                num_added_tokens=self.num_added_tokens,
                model_dtype=model_dtype,
            )
        elif connection_type == 'embedding_table':
            self.connection_module = EmbeddingTable(
                in_dim=self.universal_learner_dim,
                out_dim=self.encoder_dim,
                vocab_size=self.encoder.config.vocab_size,
                padding_idx=self.encoder.config.pad_token_id,
                llm_embedding=self.encoder.get_input_embeddings(),
                model_dtype=model_dtype,
            )
            self.num_added_tokens = 0
        else:
            raise NotImplementedError(f"Connection type {connection_type} not implemented")
        
        # Projector
        self.output_projection = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_dim, self.encoder_dim),
        )
    
    def create_tokenizer(self, model_name_or_path: str, backbone_type: str):
        # Load tokenizer
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right", # Has to be right so masking of instruction tokens works correctly
            trust_remote_code=True,
        )
        pad_token = SPECIAL_TOKENS.get(backbone_type, {}).get("pad", tokenizer.eos_token)
        mask_token = SPECIAL_TOKENS.get(backbone_type, {}).get("mask", tokenizer.unk_token)
        if tokenizer.pad_token_id is None:
            print(f"Tokenizer does not have a pad token. We will use {pad_token} as the pad token.")
            tokenizer.pad_token = pad_token
            assert tokenizer.pad_token_id is not None, "Tokenizer does not have a pad token id"
        if tokenizer.mask_token_id is None:
            print(f"Tokenizer does not have a mask token. We will use {mask_token} as the mask token.")
            tokenizer.mask_token = mask_token
            assert tokenizer.mask_token_id is not None, "Tokenizer does not have a mask token id"
        return tokenizer

    def create_transformer(
            self,
            model_name_or_path: str,
            backbone_type: str = 'mistral',
            is_llm_bidirectional: bool = False,
            use_lora: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            target_modules: Union[str, List[str]] = "all",
            adapter_name: str = 'default',
            quantization: bool = False,
            attn_implementation: str = None,
            model_dtype: torch.dtype = torch.bfloat16,
    ):  
        print(f"Loading model from {model_name_or_path}")
        if use_lora:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
                # See https://github.com/huggingface/transformers/pull/24906
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False
            )

        if quantization:
            # Prompt warning if quantization is enabled 
            print("Quantization is enabled. This may affect the performance of the model. And currently, quantization is only supported for inference or multi-gpu training WITH DPP.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        kwargs = {
            'pretrained_model_name_or_path': model_name_or_path,
            'config': config,
            'quantization_config': bnb_config,
            'torch_dtype': torch.bfloat16 if attn_implementation == "flash_attention_2" else model_dtype,
            'attn_implementation': attn_implementation,
            'trust_remote_code': True,
            'output_loading_info': True,
        }
        model_class = AutoModel
        if not is_llm_bidirectional:
            if 'mt5' in model_name_or_path:
                model_class = MT5EncoderModel
                kwargs = {
                    'pretrained_model_name_or_path': model_name_or_path, 
                    'config': config,
                    'torch_dtype': torch.bfloat16 if attn_implementation == "flash_attention_2" else model_dtype,
                    'output_loading_info': True,
                    }
            elif 'xlm' in model_name_or_path:
                kwargs.pop('attn_implementation')
        else:
            if backbone_type in ["mistral", "nvidia/NV-Embed-v2"]:
                model_class = BidirectionalMistralForCausalLM
            elif backbone_type == "llama":
                model_class = BidirectionalLlamaForCausalLM
            elif backbone_type == "phi3":
                model_class = BidirectionalPhi3ForCausalLM
            elif backbone_type == "phi":
                model_class = BidirectionalPhiForCausalLM
            elif backbone_type == "qwen2":
                model_class = BidirectionalQwen2ForCausalLM
            elif backbone_type == 'gemma2':
                model_class = BidirectionalGemma2ForCausalLM
            else:
                model_class = AutoModel
        
        print(f"Using model class: {model_class}")
        transformer, loading_info = model_class.from_pretrained(**kwargs)
        print(f"Model loading info: {loading_info}")

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(transformer, quantization)
            assert isinstance(target_modules, list) or target_modules == 'all-linear', "target_modules must be a list or 'all-linear'"
            task_type = TaskType.FEATURE_EXTRACTION
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=task_type,
                target_modules=target_modules,
            )
            if adapter_name is None:
                adapter_name = 'default'
            transformer: PeftModel = get_peft_model(transformer, lora_config, adapter_name=adapter_name)
        
        return transformer 

    def construct_input_attn_mask(self, attention_mask: torch.Tensor):
        if self.connection_type in ['ff', 'embedding_table']:	
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((attention_mask.size(0), self.num_added_tokens), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)
        else:
            raise NotImplementedError(f"Connection type {self.connection_type} not implemented")
        return attention_mask
    
    def forward(
            self,
            input_ids: torch.Tensor, # (batch_size, seq_len)
            attention_mask: torch.Tensor, # (batch_size, seq_len)
            llm_input_ids: Optional[torch.Tensor] = None, # (batch_size, seq_len)
            llm_attention_mask: Optional[torch.Tensor] = None, # (batch_size, seq_len)
            lm_labels: Optional[torch.Tensor] = None, # (batch_size, seq_len)
            ):
        universal_representation = self.universal_learner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states[-1] # (batch_size, in_len, hidden_size)
        universal_representation = self.connection_module(universal_representation, attention_mask=attention_mask)
        attention_mask = self.construct_input_attn_mask(attention_mask)

        if llm_input_ids is not None:
            assert lm_labels is not None, "lm_labels must be provided when llm_input_ids is provided"
            assert lm_labels.size(0) == input_ids.size(0), "The batch size of lm_labels and input_ids must be the same"

            target_representation = self.encoder.model.get_input_embeddings()(llm_input_ids) # (batch_size, out_len, hidden_size)
            embeddings = torch.cat([universal_representation, target_representation], dim=1) # (batch_size, in_len + out_len, hidden_size)
            target_attention_mask = torch.cat([attention_mask, llm_attention_mask], dim=1)
            input_labels = torch.zeros((universal_representation.size(0), universal_representation.size(1)), device=universal_representation.device, dtype=input_ids.dtype) + -100
            labels = torch.cat([input_labels, lm_labels], dim=1)
            is_causal = True # Must be True for causal language modeling to make sure the model only uses the past tokens
        else:
            embeddings = universal_representation
            target_attention_mask = attention_mask
            labels = None
            is_causal = False # Must be False for feature extraction to make sure the model uses all tokens

        outputs = self.encoder(
                input_ids=None,
                attention_mask=target_attention_mask,
                labels=labels,
                inputs_embeds=embeddings,
                return_dict=True,
                is_causal=is_causal,
                output_hidden_states=True
            )
        if labels is not None:
            ce_loss = outputs.loss
        else:
            ce_loss = None
        input_len = universal_representation.size(1)
        input_reps = outputs.hidden_states[-1][:, :input_len]

        if self.encoder_backbone_type == 'nvidia/NV-Embed-v2':
            pool_mask = attention_mask.clone()
            with torch.autocast(device_type=input_reps.device.type, dtype=self.model_dtype):
                sentence_representation = self.laten_attention_model(input_reps, pool_mask)
            projected_representation = sentence_representation
        else:
            if self.connection_type in ['ff', 'embedding_table']:
                sentence_representation = self.pooling(
                    hidden_state=input_reps,
                    attention_mask=attention_mask,
                    prompt_length=None, # Always None with luifer because the prompt embedding might containt some information dueto soft-prompting in the universal learner
                ) # (batch_size, hidden_size)
            else:
                raise NotImplementedError(f"Connection type {self.connection_type} not implemented")
            with torch.autocast(device_type=sentence_representation.device.type, dtype=self.model_dtype):
                projected_representation = self.output_projection(sentence_representation) # (batch_size, hidden_size)

        return {
            'reps': sentence_representation,
            'projected_reps': projected_representation,
            'ce_loss': ce_loss,
        }
    
    def tokenize_example(
            self, 
            example: Tuple[str, str],
            is_query: bool = True,
            max_length: int = 512,
    ) -> BatchEncoding:
        query_format = "{instruction}\n{example}"
        candidate_format = "{instruction}\nCandidate:\n{example}"
        if is_query:
            emb_example = query_format.format(instruction=example[0], example=example[1])
        else:
            emb_example = candidate_format.format(instruction=example[0], example=example[1])
        model_inputs = self.encoder_tokenizer(
            text=emb_example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return model_inputs
    
    def encode(
        self,
        sentences: Union[List[str], str],
        is_query: bool = True,
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        **kwargs,
    ):  
        is_single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_sentence = True
        
        sentences = [(instruction, s) for s in sentences]
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            batch = sentences[start_index:start_index+batch_size]
            inputs = [self.tokenize_example(example, is_query=is_query, max_length=max_length) for example in batch]
            inputs = self.universal_learner_tokenizer.pad(inputs, return_tensors='pt', pad_to_multiple_of=8)
            inputs = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device),
            }
            with torch.no_grad():
                reps = self(**inputs)['reps']
            all_embeddings.append(reps.cpu().to(torch.float32).numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if is_single_sentence:
            return all_embeddings[0]
        return all_embeddings

    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, is_query=True, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, is_query=False, **kwargs)

    def set_model_revision(self, revision: str):
        self.mteb_model_meta.revision = revision


class WrappedLusifer(nn.Module):
    def __init__(
            self,
            universal_learner_name_or_path: str,
            encoder_name_or_path: str,
            universal_learner_backbone_type: str = 't5',
            encoder_backbone_type: str = 'mistral',
            is_freeze_universal_learner: bool = True,
            is_freeze_encoder: bool = False,
            connection_type: str = 'ff',
            num_added_tokens: int = 0,
            pooling_method: str='mean',
            encoder_lora_name: str = 'encoder_lora',
            universal_learner_lora_name: str = 'universal_learner_lora',
            encoder_lora_target_modules: Union[str, List[str]] = "all",
            universal_learner_lora_target_modules: Union[str, List[str]] = "all",
            loar_r: int = 16,
            lora_alpha: int = 32,
            dropout: float = 0.1,
            attn_implementation: str = 'flash_attention_2',
            model_dtype: torch.dtype = torch.bfloat16,
            model_revision: str = 'dev',
            model_checkpoint: Optional[str] = None,
            num_gpus: int = 8,
    ) -> None:
        super().__init__()

        self.mteb_model_meta = mteb.ModelMeta(
            name='Lusifer',
            revision=model_revision,
            release_date=date.today().strftime("%Y-%m-%d"),
            languages=None,
        )

        self.model = Lusifer(
            universal_learner_name_or_path=universal_learner_name_or_path,
            encoder_name_or_path=encoder_name_or_path,
            universal_learner_backbone_type=universal_learner_backbone_type,
            encoder_backbone_type=encoder_backbone_type,
            is_freeze_universal_learner=is_freeze_universal_learner,
            is_freeze_encoder=is_freeze_encoder,
            connection_type=connection_type,
            num_added_tokens=num_added_tokens,
            pooling_method=pooling_method,
            encoder_lora_name=encoder_lora_name,
            universal_learner_lora_name=universal_learner_lora_name,
            encoder_lora_target_modules=encoder_lora_target_modules,
            universal_learner_lora_target_modules=universal_learner_lora_target_modules,
            loar_r=loar_r,
            lora_alpha=lora_alpha,
            dropout=dropout,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
        )

        if model_checkpoint is not None and os.path.exists(model_checkpoint):
            print(f"Loading model from checkpoint: {model_checkpoint}")
            state_dict = torch.load(model_checkpoint, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict['model'], strict=False)

        self.tokenizer = self.model.encoder_tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_gpus = min(torch.cuda.device_count(), num_gpus)
        print(f"Using {self.num_gpus} GPUs")
        self.model.to(self.device)
        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    def tokenize_example(
            self, 
            example: Tuple[str, str],
            is_query: bool = True,
            max_length: int = 512,
    ) -> BatchEncoding:
        query_format = "{instruction}\n{example}"
        candidate_format = "{instruction}\nCandidate:\n{example}"
        if is_query:
            emb_example = query_format.format(instruction=example[0], example=example[1])
        else:
            emb_example = candidate_format.format(instruction=example[0], example=example[1])
        model_inputs = self.tokenizer(
            text=emb_example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return model_inputs
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        is_query: bool = True,
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        **kwargs,
    ):  
        is_single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_sentence = True
        
        sentences = [(instruction, s) for s in sentences]
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            batch = sentences[start_index:start_index+batch_size]
            inputs = [self.tokenize_example(example, is_query=is_query, max_length=max_length) for example in batch]
            inputs = self.tokenizer.pad(inputs, return_tensors='pt', pad_to_multiple_of=8)
            inputs = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device),
            }
            reps = self.model(**inputs)['reps']
            all_embeddings.append(reps.cpu().to(torch.float32).numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if is_single_sentence:
            return all_embeddings[0]
        return all_embeddings

    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, is_query=True, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, is_query=False, **kwargs)



