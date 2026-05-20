import gc
import os
import queue
import shelve
import time
from collections import OrderedDict
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM


def get_contact_vector(cmap):
    cmap = cmap.cpu()
    L = cmap.shape[0]
    vals = []
    for k in range(3, 400):
        if k >= L:
            vals.append(0.)
        else:
            diag = cmap.diagonal(offset=k)
            vals.append(sum([diag[i] for i in range(diag.shape[0])]))
    return torch.Tensor(vals)


def dedupe_preserve_order(items):
    seen = OrderedDict()
    for item in items:
        seen.setdefault(item, None)
    return list(seen.keys())

# ---------------------------------------------------------------------------
# CUDA utils
# ---------------------------------------------------------------------------

def maybe_cleanup_cuda(device=None):
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
    model = EsmForMaskedLM.from_pretrained(f'facebook/{model_name}', dtype=torch.bfloat16)
    model.set_attn_implementation('eager')
    model = model.to(device).eval()
    return tokenizer, model

# ---------------------------------------------------------------------------
# Batching — count-based, only controls work distribution
# ---------------------------------------------------------------------------

def make_count_batches(items, batch_size):
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches

# ---------------------------------------------------------------------------
# Unified compute with GPU/CPU pipelining, token caching, and intra-task
# result caching.
# ---------------------------------------------------------------------------

def process_unified_batch(model, tokenizer, items):
    """Process a batch of items with GPU/CPU pipelining.
    While GPU computes the next forward pass, CPU computes the previous
    sequence's contact vector in a background thread.

    Caches tokenized inputs and computed results within the batch so that
    repeated sequences (e.g., shared wt in -sub mode) are only computed once."""
    device = next(model.parameters()).device
    mask_id = tokenizer._token_to_id['<mask>']
    score_results = {}
    full_results = {}

    # Intra-task caches: avoid recomputing scores/features for sequences
    # that appear multiple times within this worker batch
    local_score_cache = {}   # (true_seq, masked_seq) -> score
    local_full_cache = {}    # true_seq -> {'emb': ..., 'cvec': ...}

    # Tokenization cache: avoid re-tokenizing the same string
    tok_cache = {}

    def tokenize_cached(seq):
        if seq not in tok_cache:
            tok_cache[seq] = tokenizer([seq], return_tensors='pt', padding=True)
        # Always .to(device) since tensors may have been created on CPU
        return tok_cache[seq].to(device)

    from concurrent.futures import ThreadPoolExecutor

    def compute_cvec_in_background(cmap_tensor):
        return get_contact_vector(cmap_tensor)

    pending_cvec = None  # (future, full_key, true_seq)

    def collect_pending():
        """Collect the pending background cvec computation and store result."""
        nonlocal pending_cvec
        if pending_cvec is not None:
            fut, fk, seq = pending_cvec
            cvec_result = fut.result().numpy()
            local_full_cache[seq]['cvec'] = cvec_result
            pending_cvec = None

    with ThreadPoolExecutor(max_workers=1) as bg_executor:
        for score_key, full_key, true_seq, masked_seq in items:

            # --- Check if score already computed in this batch ---
            score_cache_key = (true_seq, masked_seq)
            if score_cache_key not in local_score_cache:
                true_tok = tokenize_cached(true_seq)
                masked_tok = tokenize_cached(masked_seq)

                out_logits = model(**masked_tok).logits.transpose(1, 2)
                tgt = true_tok['input_ids'][:, :out_logits.shape[-1]]
                loss = torch.exp(cross_entropy(out_logits, tgt, reduction='none'))
                loss[~(masked_tok['input_ids'] == mask_id)] = 0
                local_score_cache[score_cache_key] = float(
                    loss.sum().float().detach().cpu().item())

                del out_logits, tgt, loss

            score_results[score_key] = local_score_cache[score_cache_key]

            # --- Check if full features already computed in this batch ---
            if true_seq not in local_full_cache:
                # Must collect any pending cvec before starting a new full pass,
                # since we need the GPU free and the previous cmap consumed
                collect_pending()

                true_tok = tokenize_cached(true_seq)

                out = model(**true_tok, output_attentions=True,
                            output_hidden_states=True)
                attention_mask = true_tok['attention_mask']
                tokens = true_tok['input_ids']
                attns = torch.stack(out['attentions'], dim=1)
                attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)

                emb = out.hidden_states[-1].mean(dim=-2)
                cmap = model.esm.contact_head(
                    tokens.to(torch.bfloat16), attns.to(torch.bfloat16))[0]
                emb_np = emb[0].float().detach().cpu().numpy()

                del out, attention_mask, tokens, attns, emb, true_tok

                # Store embedding immediately, cvec will be filled by background
                local_full_cache[true_seq] = {'emb': emb_np, 'cvec': None}

                # Submit cvec computation to background
                fut = bg_executor.submit(compute_cvec_in_background, cmap)
                pending_cvec = (fut, full_key, true_seq)

            full_results[full_key] = None  # placeholder, filled below

        # Collect final pending cvec
        collect_pending()

    # Fill full_results from local cache
    for score_key, full_key, true_seq, masked_seq in items:
        full_results[full_key] = local_full_cache[true_seq]

    # Clear tokenization cache (may hold GPU tensors)
    tok_cache.clear()

    gc.collect()
    torch.cuda.empty_cache()
    return score_results, full_results


# ---------------------------------------------------------------------------
# Worker — autocast enabled to match gpu1.py
# ---------------------------------------------------------------------------

def worker_loop(gpu_id, model_name, task_queue, result_queue):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    result_queue.put({'type': 'ready', 'gpu': gpu_id})
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            batch_id = task['batch_id']
            items = task['items']
            try:
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    score_cache, full_cache = process_unified_batch(
                        model, tokenizer, items)
                result_queue.put({
                    'type': 'done', 'gpu': gpu_id,
                    'batch_id': batch_id, 'n_items': len(items),
                    'score_cache': score_cache, 'full_cache': full_cache})
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and len(items) > 1:
                    maybe_cleanup_cuda(device=gpu_id)
                    result_queue.put({
                        'type': 'split', 'gpu': gpu_id,
                        'batch_id': batch_id, 'items': items,
                        'message': f'GPU {gpu_id} OOM on {len(items)} items; splitting.'})
                else:
                    result_queue.put({
                        'type': 'error', 'gpu': gpu_id,
                        'batch_id': batch_id, 'error': repr(e)})
                    break
    finally:
        del model
        maybe_cleanup_cuda(device=gpu_id)
        result_queue.put({'type': 'stopped', 'gpu': gpu_id})

# ---------------------------------------------------------------------------
# Work-queue dispatcher — unified single phase
# ---------------------------------------------------------------------------

def _key_to_str(key):
    return str(key)

def run_unified_with_queue(items, model_name, gpu_ids,
                           worker_batch_size=128, checkpoint_path=None):
    desc = 'unified compute'

    score_cache = {}
    full_cache = {}
    if checkpoint_path:
        try:
            with shelve.open(checkpoint_path) as db:
                for k, v in db.items():
                    if k.startswith('s:'):
                        score_cache[k[2:]] = v
                    elif k.startswith('f:'):
                        full_cache[k[2:]] = v
            if score_cache or full_cache:
                print(f'[checkpoint] Resumed {len(score_cache)} scores, '
                      f'{len(full_cache)} features', flush=True)
        except Exception as ex:
            print(f'[checkpoint] Could not load checkpoint: {ex}', flush=True)

    # filter out already-computed items
    cached_score_keys = set(score_cache.keys())
    cached_full_keys = set(full_cache.keys())
    remaining = [x for x in items
                 if (_key_to_str(x[0]) not in cached_score_keys or
                     _key_to_str(x[1]) not in cached_full_keys)]

    if not remaining:
        print(f'All {len(items)} items cached, skipping.', flush=True)
        return score_cache, full_cache

    print(f'{len(items) - len(remaining)} cached, {len(remaining)} to compute.', flush=True)

    batches = make_count_batches(remaining, worker_batch_size)

    ctx = get_context('spawn')
    queue_cap = max(len(gpu_ids) * 4, 8)
    task_queue = ctx.Queue(maxsize=queue_cap)
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=worker_loop,
                           args=(gid, model_name, task_queue, result_queue))
               for gid in gpu_ids]
    for w in workers:
        w.start()

    startup_bar = tqdm(total=len(workers), desc=f'{desc} startup', unit='gpu',
                       dynamic_ncols=True, leave=True)
    startup_bar.refresh()
    ready = 0
    while ready < len(workers):
        msg = result_queue.get()
        if msg.get('type') == 'ready':
            ready += 1; startup_bar.update(1)
        elif msg.get('type') == 'error':
            startup_bar.close(); raise RuntimeError(msg['error'])
    startup_bar.close()

    pbar = tqdm(total=len(remaining), desc=desc, unit='seq',
                dynamic_ncols=True, leave=True, mininterval=1.0)
    pbar.refresh()

    next_batch_id = 0
    next_batch_idx = 0
    inflight = 0
    completed_batches = 0
    last_event_time = time.time()
    split_events = 0
    last_split_gpu = None
    last_split_size = None
    split_log_every = int(os.environ.get('ESM_SPLIT_LOG_EVERY', '25'))
    checkpoint_every = 5000
    items_since_ckpt = 0

    def submit_some(max_new=None):
        nonlocal next_batch_idx, next_batch_id, inflight
        target = queue_cap if max_new is None else max_new
        submitted = 0
        while next_batch_idx < len(batches) and inflight < queue_cap and submitted < target:
            task_queue.put({'batch_id': next_batch_id,
                            'items': batches[next_batch_idx]})
            next_batch_id += 1; next_batch_idx += 1; inflight += 1; submitted += 1

    submit_some()

    try:
        while inflight > 0 or next_batch_idx < len(batches):
            try:
                msg = result_queue.get(timeout=1)
            except queue.Empty:
                elapsed = int(time.time() - last_event_time)
                pbar.set_postfix(pending=inflight,
                                 queued=len(batches) - next_batch_idx,
                                 done=completed_batches,
                                 splits=split_events,
                                 idle_s=f'{elapsed}')
                pbar.refresh(); continue

            mtype = msg.get('type')
            last_event_time = time.time()

            if mtype == 'done':
                for k, v in msg['score_cache'].items():
                    score_cache[_key_to_str(k)] = v
                for k, v in msg['full_cache'].items():
                    full_cache[_key_to_str(k)] = v
                n = msg['n_items']
                pbar.update(n)
                completed_batches += 1; inflight -= 1; items_since_ckpt += n
                submit_some(max_new=1)
                pbar.set_postfix(pending=inflight,
                                 queued=len(batches) - next_batch_idx,
                                 done=completed_batches,
                                 gpu=msg.get('gpu'),
                                 splits=split_events)
                pbar.refresh()
                if checkpoint_path and items_since_ckpt >= checkpoint_every:
                    with shelve.open(checkpoint_path) as db:
                        for k, v in msg['score_cache'].items():
                            db['s:' + _key_to_str(k)] = v
                        for k, v in msg['full_cache'].items():
                            db['f:' + _key_to_str(k)] = v
                    items_since_ckpt = 0

            elif mtype == 'split':
                old_items = msg['items']
                inflight -= 1; completed_batches += 1; split_events += 1
                last_split_gpu = msg.get('gpu')
                last_split_size = len(old_items)
                mid = len(old_items) // 2
                new_batches = [x for x in (old_items[:mid], old_items[mid:]) if x]
                batches[next_batch_idx:next_batch_idx] = new_batches
                submit_some(max_new=len(new_batches))
                if split_log_every > 0 and (split_events == 1 or split_events % split_log_every == 0):
                    tqdm.write(f'{desc}: {split_events} OOM splits so far '
                               f'(last: gpu={last_split_gpu}, batch_size={last_split_size})')
                pbar.set_postfix(pending=inflight,
                                 queued=len(batches) - next_batch_idx,
                                 done=completed_batches,
                                 splits=split_events)
                pbar.refresh()

            elif mtype == 'error':
                raise RuntimeError(msg['error'])

        pbar.close()
    finally:
        for _ in workers:
            task_queue.put(None)
        for w in workers:
            w.join()
        if checkpoint_path:
            with shelve.open(checkpoint_path) as db:
                for k, v in score_cache.items():
                    db['s:' + _key_to_str(k)] = v
                for k, v in full_cache.items():
                    db['f:' + _key_to_str(k)] = v

    return score_cache, full_cache

# ---------------------------------------------------------------------------
# Main extraction — builds unified work items with dedup and sorting
# ---------------------------------------------------------------------------

def extract_features_workqueue(model_name, ids, wt_seqs, mu_seqs, masked_wt, masked_mu,
                                gpu_ids=None, checkpoint_dir=None,
                                worker_batch_size=128):
    rows = list(zip(ids, wt_seqs, mu_seqs, masked_wt, masked_mu))

    # Build unique work items and sort: wt items first so they get computed
    # and cached early, then mu items benefit from warm caches within each
    # worker batch.
    wt_seen = OrderedDict()
    mu_seen = OrderedDict()
    for _, wt, mu, mwt, mmu in rows:
        sk_wt = (wt, mwt)
        fk_wt = wt
        wt_seen.setdefault((sk_wt, fk_wt), (sk_wt, fk_wt, wt, mwt))
        sk_mu = (mu, mmu)
        fk_mu = mu
        mu_seen.setdefault((sk_mu, fk_mu), (sk_mu, fk_mu, mu, mmu))

    # Remove mu items that are identical to a wt item (already covered)
    for key in list(mu_seen.keys()):
        if key in wt_seen:
            del mu_seen[key]

    # Wt items first, then mu items
    unique_items = list(wt_seen.values()) + list(mu_seen.values())

    n_wt = len(wt_seen)
    n_mu = len(mu_seen)
    print(f'[dedup] {len(unique_items)} unique work items '
          f'({n_wt} wt + {n_mu} mu, from {2*len(rows)} total)', flush=True)
    print(f'Using worker_batch_size={worker_batch_size}, gpu_ids={gpu_ids}', flush=True)

    ckpt = os.path.join(checkpoint_dir, 'ckpt_unified') if checkpoint_dir else None

    score_cache, full_cache = run_unified_with_queue(
        unique_items, model_name, gpu_ids,
        worker_batch_size=worker_batch_size, checkpoint_path=ckpt)

    out_rows = []
    for uid, wt, mu, mwt, mmu in tqdm(rows, desc='assemble features', unit='seq'):
        wt_score = score_cache[_key_to_str((wt, mwt))]
        mu_score = score_cache[_key_to_str((mu, mmu))]
        emb_wt = full_cache[_key_to_str(wt)]['emb']
        emb_mu = full_cache[_key_to_str(mu)]['emb']
        cvec_wt = full_cache[_key_to_str(wt)]['cvec']
        cvec_mu = full_cache[_key_to_str(mu)]['cvec']
        embedding_distance = float(np.square(emb_wt - emb_mu).sum())
        denom = np.sum(cvec_wt + cvec_mu)
        structural_distance = float(np.sum(np.abs(cvec_wt - cvec_mu)) / denom) if denom != 0 else 0.0
        out_rows.append({'ids': uid, 'wt_score': wt_score, 'mu_score': mu_score,
                         'embedding_distance': embedding_distance,
                         'structural_distance': structural_distance})
    return pd.DataFrame(out_rows)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_gpu_ids():
    env = os.environ.get('ESM_GPU_IDS')
    if env:
        return [int(x.strip()) for x in env.split(',') if x.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []

def main():
    sample_name = os.environ['SAMPLE_NAME']
    in_file = f'output/{sample_name}/masked_sequences.csv'
    out_file = f'output/{sample_name}/esm_features.csv'
    ckpt_dir = os.environ.get('ESM_CHECKPOINT_DIR',
                              f'output/{sample_name}/checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    X = pd.read_csv(in_file)
    X = X.loc[X[['wt', 'mu']].map(len).max(axis=1) <= 2000]

    ids = list(X['id'])
    wt = list(X['wt'])
    mu = list(X['mu'])
    masked_wt = list(X['masked_wt'])
    masked_mu = list(X['masked_mu'])

    gpu_ids = parse_gpu_ids()
    if not gpu_ids:
        raise RuntimeError('No CUDA devices available.')

    model_name = os.environ.get('ESM_MODEL_NAME', 'esm2_t33_650M_UR50D')
    worker_batch_size = int(os.environ.get('ESM_WORKER_BATCH_SIZE', '128'))

    res = extract_features_workqueue(
        model_name=model_name,
        ids=ids, wt_seqs=wt, mu_seqs=mu,
        masked_wt=masked_wt, masked_mu=masked_mu,
        gpu_ids=gpu_ids,
        checkpoint_dir=ckpt_dir,
        worker_batch_size=worker_batch_size,
    )

    res.to_csv(out_file, index=False)
    print(f'wrote {out_file}')

if __name__ == '__main__':
    main()
