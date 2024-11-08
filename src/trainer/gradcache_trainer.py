import shutil
from collections import UserDict
import pathlib
from contextlib import nullcontext
import time
from typing import Any, Callable, Dict, List, Optional
import typing
import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import get_device_states, set_device_states
import lightning as L

from src.model.lusifer import Lusifer, WrappedLusifer
from src.trainer.loss import ContrastiveLoss
from src.trainer.utils import clear_unused_gpu_mem, split_input
from src.eval.eval import eval_multilingual, eval_mteb


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class GradCacheTrainer:
    def __init__(
            self,
            fabric: L.Fabric,
            use_ce: bool = True,
            use_con: bool = True,
            loss_type: str = 'NTXentLoss',
            temperature: float = 0.05,
            is_distance: bool = True,
            use_miner: bool = False,
            chunk_size: Optional[int] = 1,
            ) -> None:
        
        self.fabric = fabric
        self.chunk_size = chunk_size
        assert use_ce or use_con, 'At least one of CE or Con loss should be enabled'
        self.use_ce = use_ce
        self.use_con = use_con
        self.loss_fn = ContrastiveLoss(
            loss_type=loss_type,
            temperature=temperature,
            is_distance=is_distance,
            use_miner=use_miner,
        )

        self.best_overall_metric = 0.0
        self.best_en = 0.0
        self.best_multi = 0.0

    def get_input_tensors(self, model_input) -> List[torch.Tensor]:
        if isinstance(model_input, torch.Tensor):
            return [model_input]
        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])
        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])
        else:
            return []
    
    def forward_no_grad(
            self, 
            model: Lusifer, 
            model_inputs: Dict[str, torch.Tensor],
            ):
        with torch.no_grad():
            rnd_state = RandContext(*self.get_input_tensors(model_inputs))

            P = model_inputs['min_pos_per_sample']
            B = model_inputs['in_ids'].size(0)
            in_ids = model_inputs['in_ids'] # (batch_size, 1 + #p + #n, in_seq_len)
            in_attention_mask = model_inputs['in_attention_mask']
            target_ids = model_inputs['target_ids'] # (bs, 1 + #p + #n, out_seq_len)
            target_attention_mask = model_inputs['target_attention_mask']
            target_labels = model_inputs['target_labels']

            input_ids = einops.rearrange(in_ids, 'b n s -> (b n) s')
            attention_mask = einops.rearrange(in_attention_mask, 'b n s -> (b n) s')
            if target_ids is not None:
                llm_input_ids = einops.rearrange(target_ids, 'b n s -> (b n) s')
                llm_attention_mask = einops.rearrange(target_attention_mask, 'b n s -> (b n) s')
                llm_labels = einops.rearrange(target_labels, 'b n s -> (b n) s')
            else:
                llm_input_ids = None
                llm_attention_mask = None
                llm_labels = None
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_input_ids=llm_input_ids,
                llm_attention_mask=llm_attention_mask,
                lm_labels=llm_labels,
            )
            projected_reps = outputs['projected_reps'] # (batch_size * (1 + #p + #n), embed_dim)
            projected_reps = einops.rearrange(projected_reps, '(b n) d -> b n d', b=B)
            projected_query = projected_reps[:, 0]
            projected_pos = projected_reps[:, 1:1+P]
            projected_neg = projected_reps[:, 1+P:]
        
        return projected_query, projected_pos, projected_neg, rnd_state

    def compute_cons_loss_from_reps(
            self,
            query_projections: torch.Tensor, # (batch_size, embed_dim)
            pos_projections: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_projections: torch.Tensor, # (batch_size, num_neg, embed_dim)
            query_labels: torch.Tensor, # (batch_size,)
            cross_batch_loss: bool = True,
            ) -> torch.Tensor:
        con_loss = self.loss_fn(
            q_embeds=query_projections,
            q_labels=query_labels,
            pos_embeds=pos_projections,
            neg_embeds=neg_projections,
            cross_batch_loss=cross_batch_loss,
        )
        return con_loss
    
    @typing.no_type_check
    def build_cache(
            self,
            query_projections: torch.Tensor, # (batch_size, embed_dim)
            pos_projections: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_projections: torch.Tensor, # (batch_size, num_neg, embed_dim)
            query_labels: torch.Tensor, # (batch_size,)
            cross_batch_loss: bool = True,
            ):
        B, P, _ = pos_projections.size()
        B, N, _ = neg_projections.size()
        projections = torch.cat([
            query_projections.unsqueeze(1), # (batch_size, 1, embed_dim)
            pos_projections, # (batch_size, num_pos, embed_dim)
            neg_projections, # (batch_size, num_neg, embed_dim)
        ], dim=1) # (batch_size, 1 + num_pos + num_neg, embed_dim)
        projections = projections.detach().requires_grad_(True)

        query_projections = projections[:, 0] # (batch_size, embed_dim)
        pos_projections = projections[:, 1:1+P] # (batch_size, num_pos, embed_dim)
        neg_projections = projections[:, 1+P:] # (batch_size, num_neg, embed_dim)

        with nullcontext():
            with self.fabric.autocast():
                con_loss = self.compute_cons_loss_from_reps(
                    query_projections=query_projections,
                    pos_projections=pos_projections,
                    neg_projections=neg_projections,
                    query_labels=query_labels,
                    cross_batch_loss=cross_batch_loss,
                )
        self.fabric.backward(con_loss)
        cache = projections.grad # (batch_size, 1 + num_pos + num_neg, embed_dim)
        
        if cross_batch_loss:
            con_loss = con_loss.detach() / self.fabric.world_size
        else:
            con_loss = con_loss.detach()

        return cache, con_loss

    def forward_backward(
            self,
            model: Lusifer,
            model_inputs: Dict[str, torch.Tensor],
            state: RandContext, 
            cache: torch.Tensor, # (batch_size, 1 + num_pos + num_neg, embed_dim)
            num_chunks: int,
            ):
        with state:
            P = model_inputs['min_pos_per_sample']
            B = model_inputs['in_ids'].size(0)
            in_ids = model_inputs['in_ids'] # (batch_size, 1 + #p + #n, in_seq_len)
            in_attention_mask = model_inputs['in_attention_mask']
            target_ids = model_inputs['target_ids'] # (bs, 1 + #p + #n, out_seq_len)
            target_attention_mask = model_inputs['target_attention_mask']
            target_labels = model_inputs['target_labels']

            input_ids = einops.rearrange(in_ids, 'b n s -> (b n) s')
            attention_mask = einops.rearrange(in_attention_mask, 'b n s -> (b n) s')
            llm_input_ids = einops.rearrange(target_ids, 'b n s -> (b n) s')
            llm_attention_mask = einops.rearrange(target_attention_mask, 'b n s -> (b n) s')
            llm_labels = einops.rearrange(target_labels, 'b n s -> (b n) s') if self.use_ce else None
            
            # GC mini-batch forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_input_ids=llm_input_ids,
                llm_attention_mask=llm_attention_mask,
                lm_labels=llm_labels,
            )
            if self.use_ce:
                ce_loss = outputs['ce_loss']
                loss = ce_loss / num_chunks # num_chunks acts as num_accumulated_steps
            else:
                ce_loss = None
            if self.use_con:
                cache = einops.rearrange(cache, 'b n d -> (b n) d', b=B) # (batch_size * (1 + num_pos + num_neg), embed_dim)
                projections = outputs['projected_reps']
                surrougate = torch.dot(projections.flatten(), cache.flatten())
                loss = surrougate
            else:
                surrougate = None
            
            if self.use_con and self.use_ce:
                loss = ce_loss + surrougate
            # GC mini-batch backward pass
            self.fabric.backward(loss)
        return ce_loss, loss
    
    def train_step(
            self,
            model: Lusifer,
            batch: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
        
        # Split input into chunks
        assert 'min_pos_per_sample' in batch, 'min_pos_per_sample is required for negative sampling'
        P = batch.pop('min_pos_per_sample')
        enable_cross_batch_negative_sampling = batch.pop('enable_cross_batch_negative_sampling', True)
        splitted_inputs = split_input(batch, self.chunk_size)
        num_chunks = len(splitted_inputs)

        # Forward pass for each chunk
        if self.use_con:
            rnd_states = []
            all_query_projections = []
            all_pos_projections = []
            all_neg_projections = []
            for chunk in splitted_inputs:
                chunk['min_pos_per_sample'] = P
                query_projections, pos_projections, neg_projections, rnd_state = self.forward_no_grad(model, chunk)
                all_query_projections.append(query_projections)
                all_pos_projections.append(pos_projections)
                all_neg_projections.append(neg_projections)
                rnd_states.append(rnd_state)
            all_query_projections = torch.cat(all_query_projections, dim=0)
            all_pos_projections = torch.cat(all_pos_projections, dim=0)
            all_neg_projections = torch.cat(all_neg_projections, dim=0)

            # Build cache for representations from all chunks
            labels = batch['query_labels']
            cache, con_loss = self.build_cache(
                query_projections=all_query_projections,
                pos_projections=all_pos_projections,
                neg_projections=all_neg_projections,
                query_labels=labels,
                cross_batch_loss=enable_cross_batch_negative_sampling,
            )
            cache = cache.split(self.chunk_size, dim=0)
        else:
            con_loss = None
            rnd_states = [nullcontext() for _ in range(len(splitted_inputs))]
            cache = [None for _ in range(len(splitted_inputs))]

        # Forward and backward pass for each chunk
        accumulated_flags = [True for _ in range(len(splitted_inputs)-1)] + [False]
        all_ce_loss = []
        all_loss = []
        for chunk, c, state, flag in zip(splitted_inputs, cache, rnd_states, accumulated_flags):
            chunk['min_pos_per_sample'] = P
            with self.fabric.no_backward_sync(model, enabled=flag):
                ce_loss, loss = self.forward_backward(
                    model=model,
                    model_inputs=chunk,
                    state=state,
                    cache=c,
                    num_chunks=num_chunks,
                )
            all_ce_loss.append(ce_loss)
            all_loss.append(loss)
        
        if None in all_ce_loss:
            ce_loss = None
        else:
            ce_loss = torch.stack(all_ce_loss).mean()
        loss = torch.stack(all_loss).mean()
        return con_loss, ce_loss, loss

    def fit_epoch(
            self,
            model: Lusifer,
            train_loader: DataLoader,
            state: Dict[str, Any],
            lr_max_steps: int = 1000,
            grad_norm_clip: float = None,
            log_interval: int = 1,
            checkpoint_iterval: Optional[int] = 10000,
            checkpoint_dir: Optional[str] = './checkpoints/',
            checkpoint_filter: Optional[Callable] = None,
            model_revision: Optional[str] = 'v0.1',
            eval_batch_size: Optional[int] = 32,
            ):
        """
        Fit epoch for gradient cache training.
        """
        optimizer: torch.optim.Optimizer = state["optimizer"]
        scheduler : torch.optim.lr_scheduler.LambdaLR = state.get("scheduler", None)
        current_step = state.get("current_step", 0) # checkpoint iteration number
        epoch_num = state.get("epoch_num", 0) # checkpoint epoch number
        self.fabric.print(f"Starting epoch {epoch_num} with {len(train_loader)} iterations")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if current_step > len(train_loader)*epoch_num + batch_idx:
                continue
            if current_step > lr_max_steps:
                break

            current_step = current_step + 1
            if current_step == 1:
                size_info = {k: v.size() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.fabric.print("First batch data: {}".format(size_info))

            iter_t0 = time.perf_counter()  
            con_loss, ce_loss, loss = self.train_step(model=model, batch=batch)

            if grad_norm_clip is not None:
                self.fabric.clip_gradients(model, optimizer, max_norm=grad_norm_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            if current_step % log_interval == 0:
                t1 = time.perf_counter()

                metrics = {
                    'con_loss': con_loss.item() if con_loss is not None else None,
                    'ce_loss': ce_loss.item() if ce_loss is not None else None,
                    'iter_time': t1 - iter_t0,
                    'epoch': epoch_num,
                    # 'iter_num': batch_idx,
                    'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                }
                self.fabric.log_dict(metrics, step=current_step)
                self.fabric.print(
                    f"Epoch {epoch_num} | Iter {batch_idx} |"
                    f" ConLoss: {metrics['con_loss']} |"
                    f" CELoss: {metrics['ce_loss']} |"
                    f" LR: {metrics['lr']} |"
                    f" Iter time: {metrics['iter_time']}s |"
                )
            
            # Save checkpoint and evaluate each checkpoint interval steps or at the end of the epoch or at the end of training
            if current_step % checkpoint_iterval == 0 or current_step == lr_max_steps or batch_idx + 1 == len(train_loader):
                checkpoint_path = pathlib.Path(checkpoint_dir) / "lastest.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "current_step": current_step,
                    "epoch_num": epoch_num if batch_idx < len(train_loader)-1 else epoch_num + 1,
                }
                if checkpoint_filter is not None:
                    self.fabric.save(checkpoint_path, state, filter={'model': checkpoint_filter})
                else:
                    self.fabric.save(checkpoint_path, state)
                self.fabric.print(f"Checkpoint saved at {checkpoint_path}")
                clear_unused_gpu_mem()
                self.fabric.load(checkpoint_path, state, strict=False)
                model = state.pop("model")
                optimizer = state.pop("optimizer")
                scheduler = state.pop("scheduler")
                self.fabric.barrier()
                model_hprams = model.hprams

                # Evaluate model
                if self.fabric.global_rank == 0:
                    self.fabric.print("Evaluating model")
                    _model_revision = f"{model_revision}_step-{current_step}_epoch-{epoch_num}"
                    eval_model = WrappedLusifer(
                        model_revision=_model_revision, 
                        model_checkpoint=checkpoint_path,
                        **model_hprams
                        )
                    mteb_results = eval_mteb(
                        model=eval_model,
                        output_folder=checkpoint_dir,
                        batch_size=eval_batch_size,
                        is_quick_run=True,
                    )
                    multilingual_results = eval_multilingual(
                        model=eval_model,
                        langs=['ru', 'vi', 'fa', 'hi', 'bn', 'yo'],
                        output_folder=checkpoint_dir,
                        batch_size=eval_batch_size,
                        is_quick_run=True,
                    )
                    results = {
                        'Avg/mteb_quick_avg': mteb_results['avg']['all_tasks'],
                        'Avg/multilingual_quick_avg': multilingual_results['avg']['all_tasks'],
                    }
                    self.fabric.log_dict(results, step=current_step)
                    # Eval logic here
                    self.fabric.print("Model evaluation finished")
                    del eval_model
                    clear_unused_gpu_mem()

                    # Save best checkpoint based on evaluation
                    if results['Avg/mteb_quick_avg'] > self.best_en:
                        self.best_en = results['Avg/mteb_quick_avg']
                        best_checkpoint_path = pathlib.Path(checkpoint_dir) / "best_en.ckpt"
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        self.fabric.print(f"Best en checkpoint saved at {best_checkpoint_path}")
                    if results['Avg/multilingual_quick_avg'] > self.best_multi:
                        self.best_multi = results['Avg/multilingual_quick_avg']
                        best_checkpoint_path = pathlib.Path(checkpoint_dir) / "best_multi.ckpt"
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        self.fabric.print(f"Best multi checkpoint saved at {best_checkpoint_path}")
                self.fabric.barrier()
        return checkpoint_path

