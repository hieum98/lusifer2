import bisect
from collections import defaultdict
import math
import os
import random
from typing import Dict, Tuple
import einops
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
import datasets

from src.data_modules.constants import DATA, PRETRAINING_RECONSTRUCT, PRETRAINING_PASSAGE2QUERY, PRETRAINING_QUERY2PASSAGE


class LusiferDataset(Dataset):
    def __init__(
            self, 
            data_name: str,
            number_training_samples: int=1_000_000,
            neg_per_sample: int=1,
            pos_per_sample: int=1,
            seed: int=777,
            ):
        super().__init__()
        self.data_name = data_name
        self.data_path = DATA[data_name]['data_path']
        self.instruction = DATA[data_name]['instruction']
        self.enable_cross_batch_negative_sampling = DATA[data_name].get('enable_cross_batch_negative_sampling', True)
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.seed = seed
        print(f"Seed: {self.seed}")
        self.rng = random.Random(self.seed)

        self.data, self.cluster = self.get_data()

    def get_data(self):
        print(f"Loading data from {self.data_name}")
        number_data = self.number_training_samples
        dataset = datasets.load_dataset(self.data_name, split='train')
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            print("Failed to get number of CPU cores, using default value 1")

        if len(dataset) > number_data:
            cluster = set(dataset['cluster'])
            example_per_cluster = math.ceil(number_data / len(cluster))
            cluster_with_id = dataset.map(lambda example, idx: {'id': idx, 'cluster': example['cluster']}, with_indices=True, num_proc=max_num_worker_suggest, remove_columns=dataset.column_names, load_from_cache_file=False)
            cluster_with_id = cluster_with_id.to_pandas()
            # group by cluster
            cluster_with_id = cluster_with_id.groupby('cluster')['id'].apply(list).reset_index()
            cluster_with_id = cluster_with_id.to_dict(orient='records')
            # sort by the number of examples in the cluster
            cluster_with_id.sort(key=lambda x: len(x['id']))

            # get the examples
            selected_index = []
            for clus in cluster_with_id:
                in_cluster_index = clus['id']
                in_cluster_index.sort()
                in_cluster_index = self.rng.sample(in_cluster_index, min(len(in_cluster_index), example_per_cluster))
                selected_index.extend(in_cluster_index)

            if len(selected_index) < number_data:
                all_data_index = list(range(len(dataset)))
                self.rng.shuffle(all_data_index)
                for idx in all_data_index:
                    if idx not in selected_index:
                        selected_index.append(idx)
                    if len(selected_index) >= number_data:
                        break
            selected_index.sort()
            dataset = dataset.select(selected_index)

        print(f"Assigning cluster to each example for the dataset {self.data_name} of size {len(dataset)}...")
        cluster = dataset.map(lambda example, idx: {'cluster': example['cluster'], 'id': idx}, with_indices=True, 
                                      num_proc=max_num_worker_suggest, remove_columns=dataset.column_names, load_from_cache_file=False)
        # group by cluster
        cluster = cluster.to_pandas()
        cluster = cluster.groupby('cluster')['id'].apply(list).reset_index()
        cluster = cluster.to_dict(orient='records')
        cluster.sort(key=lambda x: x['cluster'])
        cluster = {clus['cluster']: sorted(clus['id']) for clus in cluster}
            
        return dataset, cluster

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.data[index]
        pos = self.rng.sample(example['positive'], min(len(example['positive']), self.pos_per_sample))
        neg = self.rng.sample(example['negative'], min(len(example['negative']), self.neg_per_sample))
        assert len(pos) > 0, "At least one positive example per sample, got {} in idx {}. Please check the data {}".format(example, index, self.data_name)
        assert len(neg) > 0, "At least one negative example per sample, got {} in idx {}. Please check the data {}".format(example, index, self.data_name)

        if self.data_name in PRETRAINING_PASSAGE2QUERY:
            alignment_instruction = "Please write a query that can be used to retrieve above passage."
            is_passage2query = True
        elif self.data_name in PRETRAINING_QUERY2PASSAGE:
            alignment_instruction = "Please write a passage that can be used to answer above query."
            is_passage2query = False
        else:
            alignment_instruction = "Please write a query that can be used to retrieve above passage."
            is_passage2query = True

        return {
            'query_label': index,
            'query': example['query'], # str
            'positive': pos, # list of str
            'negative': neg, # list of str
            'instruction': self.instruction,
            'alignment_instruction': alignment_instruction,
            'enable_cross_batch_negative_sampling': self.enable_cross_batch_negative_sampling,
            'is_passage2query': is_passage2query,
        }


class ConcatLusiferDataset(ConcatDataset):
    """
    An extension of ConcatDataset that guarantees that each example has a unique query_label.
    """
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        example =  self.datasets[dataset_idx][sample_idx]

        # Update the query_label to be unique across all datasets
        example['query_label'] = idx
        return example


class LusiferCollator:
    def __init__(
            self, 
            max_length: int, 
            input_tokenizer: PreTrainedTokenizer, 
            output_tokenizer: PreTrainedTokenizer,
            mask_probability: float=0.0,
            constrastive_training_only: bool=False,
            ):
        self.constrastive_training_only = constrastive_training_only
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.formater = "{instruction}\n{example}"
        self.candidate_formater = "{instruction}\nCandidate:\n{example}"

    def tokenize_example(
            self,
            example: str,
            tokenizer: PreTrainedTokenizer,
            is_query: bool=False,
            is_alignment: bool=False,
            instruction: str="",
            ) -> BatchEncoding:
        if len(example) == 0:
            print('example:', example)
        if is_query or is_alignment:
            example = self.formater.format(instruction=instruction, example=example)
        else:
            example = self.formater.format(instruction=instruction, example=example)
        
        model_inputs = tokenizer(
            example,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return model_inputs

    def __call__(self, batch):
        batch_size = len(batch)

        query_labels = [example['query_label'] for example in batch]
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        enable_cross_batch_negative_sampling = all([example['enable_cross_batch_negative_sampling'] for example in batch])

        min_pos_per_sample = min([len(example['positive']) for example in batch])
        min_neg_per_sample = min([len(example['negative']) for example in batch])
        assert min_pos_per_sample > 0, "At least one positive example per sample"
        assert min_neg_per_sample > 0, "At least one negative example per sample"

        batch_query_in = []
        batch_query_out = []
        batch_pos_in = []
        batch_pos_out = []
        batch_neg_in = []
        batch_neg_out = []
        for example in batch:
            q = example['query']
            pos = example['positive']
            neg = example['negative']
            instruction = example['instruction']
            is_passage2query = example['is_passage2query']
            alignment_instruction = example['alignment_instruction']
            reconstruct_instruction = "Please reconstruct the following text given the original text above."

            neg = random.sample(neg, min_neg_per_sample)
            for example in neg:
                # Reconstruct the negative example
                n_in = self.tokenize_example(example, tokenizer=self.input_tokenizer, is_query=False, instruction=instruction)
                if self.constrastive_training_only:
                    n_out = self.tokenize_example(example, tokenizer=self.output_tokenizer, is_query=False, instruction=instruction)
                else:
                    n_out = self.tokenize_example(example, tokenizer=self.output_tokenizer, is_alignment=True, instruction=reconstruct_instruction)
                batch_neg_in.append(n_in)
                batch_neg_out.append(n_out)
            
            if is_passage2query:
                # Reconstruct the query
                q_in = self.tokenize_example(q, is_query=True, instruction=instruction, tokenizer=self.input_tokenizer)
                if self.constrastive_training_only:
                    q_out = self.tokenize_example(q, is_query=True, instruction=instruction, tokenizer=self.output_tokenizer)
                else:
                    q_out = self.tokenize_example(q, is_alignment=True, instruction=reconstruct_instruction, tokenizer=self.output_tokenizer)
                batch_query_in.append(q_in)
                batch_query_out.append(q_out)
                pos = random.sample(pos, min_pos_per_sample)
                for example in pos:
                    # Align the positive example to the query
                    p_in = self.tokenize_example(example, is_query=False, instruction=instruction, tokenizer=self.input_tokenizer)
                    if self.constrastive_training_only:
                        p_out = self.tokenize_example(example, is_query=False, instruction=instruction, tokenizer=self.output_tokenizer)
                    else:
                        p_out = self.tokenize_example(q, is_alignment=True, instruction=alignment_instruction, tokenizer=self.output_tokenizer)
                    batch_pos_in.append(p_in)
                    batch_pos_out.append(p_out)
            else:
                # Align the query to the positive example
                q_in = self.tokenize_example(q, is_query=True, instruction=instruction, tokenizer=self.input_tokenizer)
                if self.constrastive_training_only:
                    q_out = self.tokenize_example(q, is_query=True, instruction=instruction, tokenizer=self.output_tokenizer)
                else:
                    random_pos = random.choice(pos)
                    q_out = self.tokenize_example(random_pos, is_alignment=True, instruction=alignment_instruction, tokenizer=self.output_tokenizer)
                batch_query_in.append(q_in)
                batch_query_out.append(q_out)
                pos = random.sample(pos, min_pos_per_sample)
                for example in pos:
                    # Reconstruct the positive example
                    p_in = self.tokenize_example(example, is_query=False, instruction=instruction, tokenizer=self.input_tokenizer)
                    if self.constrastive_training_only:
                        p_out = self.tokenize_example(example, is_query=False, instruction=instruction, tokenizer=self.output_tokenizer)
                    else:
                        p_out = self.tokenize_example(example, is_alignment=True, instruction=reconstruct_instruction, tokenizer=self.output_tokenizer)
                    batch_pos_in.append(p_in)
                    batch_pos_out.append(p_out)
        
        batch_in = batch_query_in + batch_pos_in + batch_neg_in
        in_batch = self.input_tokenizer.pad(batch_in, return_tensors='pt')
        in_ids = einops.rearrange(in_batch['input_ids'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, in_seq_len)
        in_attn_mask = einops.rearrange(in_batch['attention_mask'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, in_seq_len)

        # q_ids = batch['input_ids'][:len_q] # (batch_size, q_len)
        # q_attn_mask = batch['attention_mask'][:len_q]
        # p_ids = batch['input_ids'][len_q:len_q+len_p]
        # p_ids = einops.rearrange(p_ids, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        # p_attn_mask = batch['attention_mask'][len_q:len_q+len_p]
        # p_attn_mask = einops.rearrange(p_attn_mask, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        # n_ids = batch['input_ids'][len_q+len_p:]
        # n_ids = einops.rearrange(n_ids, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
        # n_attn_mask = batch['attention_mask'][len_q+len_p:]
        # n_attn_mask = einops.rearrange(n_attn_mask, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)

        if self.constrastive_training_only:
            target_ids = None
            target_attention_mask = None
            target_labels = None
        else:
            target_batch = batch_query_out + batch_pos_out + batch_neg_out
            target_batch = self.output_tokenizer.pad(target_batch, return_tensors='pt') # (bs*(1 + #p + #n), target_seq_len)
            target_batch['labels'] = target_batch['input_ids'].clone()
            padding_indices = target_batch['attention_mask'] == 0
            target_batch['labels'][padding_indices] = -100
            if self.mask_probability > 0.0:
                # # Random attention masking
                # masked_indices_for_attention = torch.rand(target_batch['attention_mask'].shape) < self.mask_probability
                # target_batch['attention_mask'][masked_indices_for_attention] = 0
                # Random token masking
                masked_indices_for_input = torch.rand(target_batch['input_ids'].shape) < self.mask_probability
                masked_indices_for_input[padding_indices] = False
                target_batch['input_ids'][masked_indices_for_input] = self.output_tokenizer.mask_token_id
            
            target_ids = einops.rearrange(target_batch['input_ids'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, target_seq_len)
            target_attention_mask = einops.rearrange(target_batch['attention_mask'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, target_seq_len)
            target_labels = einops.rearrange(target_batch['labels'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, target_seq_len)
        
            # target_q_ids = target_batch['input_ids'][:len_q] # (batch_size, q_len)
            # target_q_attn_mask = target_batch['attention_mask'][:len_q]
            # target_q_labels = target_batch['labels'][:len_q]
            # target_p_ids = target_batch['input_ids'][len_q:len_q+len_p]
            # target_p_ids = einops.rearrange(target_p_ids, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
            # target_p_attn_mask = target_batch['attention_mask'][len_q:len_q+len_p]
            # target_p_attn_mask = einops.rearrange(target_p_attn_mask, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
            # target_p_labels = target_batch['labels'][len_q:len_q+len_p]
            # target_p_labels = einops.rearrange(target_p_labels, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
            # target_n_ids = target_batch['input_ids'][len_q+len_p:]
            # target_n_ids = einops.rearrange(target_n_ids, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
            # target_n_attn_mask = target_batch['attention_mask'][len_q+len_p:]
            # target_n_attn_mask = einops.rearrange(target_n_attn_mask, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
            # target_n_labels = target_batch['labels'][len_q+len_p:]
            # target_n_labels = einops.rearrange(target_n_labels, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)

        return {
            'enable_cross_batch_negative_sampling': enable_cross_batch_negative_sampling,
            'query_labels': query_labels, # (batch_size,)
            'min_pos_per_sample': min_pos_per_sample,
            'in_ids': in_ids, # (batch_size, 1 + #p + #n, in_seq_len)
            'in_attention_mask': in_attn_mask,
            'target_ids': target_ids, # (batch_size, 1 + #p + #n, target_seq_len)
            'target_attention_mask': target_attention_mask,
            'target_labels': target_labels,
            
            # 'query_ids': q_ids, # (batch_size, q_len)
            # 'query_attention_mask': q_attn_mask,
            # 'positive_ids': p_ids, # (batch_size, #p, p_len)
            # 'positive_attention_mask': p_attn_mask,
            # 'negative_ids': n_ids, # (batch_size, #n, n_len)
            # 'negative_attention_mask': n_attn_mask,

            # 'target_query_ids': target_q_ids, # (batch_size, q_len)
            # 'target_query_attention_mask': target_q_attn_mask, 
            # 'target_query_labels': target_q_labels,
            # 'target_positive_ids': target_p_ids, # (batch_size, #p, p_len)
            # 'target_positive_attention_mask': target_p_attn_mask,
            # 'target_positive_labels': target_p_labels,
            # 'target_negative_ids': target_n_ids, # (batch_size, #n, n_len)
            # 'target_negative_attention_mask': target_n_attn_mask,
            # 'target_negative_labels': target_n_labels,
        }
            
