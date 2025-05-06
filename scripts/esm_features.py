from transformers import AutoTokenizer, EsmForMaskedLM
import sys, os, gc
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm

def get_contact_vector(cmap):
    return torch.Tensor([0. if k >= cmap.shape[0] else   # protein shorter than 400 aa
                         # count of the contact at distance k
                         sum([cmap[i, i+k] for i in range(cmap.shape[0]-k)])
                         for k in range(3, 400)
                         ])

def esm_features(model, tokenizer, ids, wt_seqs, mu_seqs, masked_wt, masked_mu):
    mask_id = tokenizer._token_to_id['<mask>']
    res = {'ids': ids}

    # tokenize all sequences
    wt_tok = tokenizer(wt_seqs, return_tensors='pt', padding=True).to('cuda')
    mu_tok = tokenizer(mu_seqs, return_tensors='pt', padding=True).to('cuda')
    masked_wt_tok = tokenizer(masked_wt, return_tensors='pt', padding=True).to('cuda')
    masked_mu_tok = tokenizer(masked_mu, return_tensors='pt', padding=True).to('cuda')

    ######################################################
    # get score of wt sequence
    out_logits = model(**masked_wt_tok).logits.transpose(1, 2)
    tgt = wt_tok['input_ids'][:, :out_logits.shape[-1]]
    loss = torch.exp(cross_entropy(out_logits, tgt, reduction='none'))
    loss[~(masked_wt_tok['input_ids'] == mask_id)] = 0
    res['wt_score'] = loss.sum(axis=-1).type(torch.float).detach().cpu().numpy()

    # CLEAN MEMORY
    del masked_wt_tok, tgt, out_logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    ######################################################
    # get score of mu sequence
    out_logits = model(**masked_mu_tok).logits.transpose(1, 2)
    tgt = mu_tok['input_ids'][:, :out_logits.shape[-1]]
    loss = torch.exp(cross_entropy(out_logits, tgt, reduction='none'))
    loss[~(masked_mu_tok['input_ids'] == mask_id)] = 0
    res['mu_score'] = loss.sum(axis=-1).type(torch.float).detach().cpu().numpy()

    # CLEAN MEMORY
    del masked_mu_tok, tgt, out_logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    #####################################################

    tokens_wt = wt_tok["input_ids"]
    out_wt = model(**wt_tok, output_attentions=True,
                   output_hidden_states=True)
    attention_mask_wt = wt_tok["attention_mask"]
    attns_wt = torch.stack(out_wt['attentions'],1)
    attns_wt *= attention_mask_wt.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    attns_wt *= attention_mask_wt.unsqueeze(1).unsqueeze(2).unsqueeze(4)


    cmap_wt = model.esm.contact_head(tokens_wt.to(torch.bfloat16),
                   attns_wt.to(torch.bfloat16)) # compute wt contact map

    # convert contact map to contact vector
    cvecs_wt = torch.stack([get_contact_vector(cmap_wt[i])
                            for i in range(cmap_wt.shape[0])])

    emb_wt = out_wt.hidden_states[-1].mean(axis=-2) # get embedding of wt sequence


    # CLEAN MEMORY
    del cmap_wt, wt_tok, out_wt, attention_mask_wt, tokens_wt, attns_wt
    gc.collect()
    torch.cuda.empty_cache()

    ########################################################


    tokens_mu = mu_tok["input_ids"]
    out_mu = model(**mu_tok, output_attentions=True,
                   output_hidden_states=True)
    attention_mask_mu = mu_tok["attention_mask"]
    attns_mu = torch.stack(out_mu['attentions'],1)
    attns_mu *= attention_mask_mu.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    attns_mu *= attention_mask_mu.unsqueeze(1).unsqueeze(2).unsqueeze(4)


    cmap_mu = model.esm.contact_head(tokens_mu.to(torch.bfloat16),
                                     attns_mu.to(torch.bfloat16)) # compute mu contact map

    # convert contact map to contact vector
    cvecs_mu = torch.stack([get_contact_vector(cmap_mu[i])
                            for i in range(cmap_mu.shape[0])])

    emb_mu = out_mu.hidden_states[-1].mean(axis=-2) # get embedding of mu sequence

    # CLEAN MEMORY
    del cmap_mu, mu_tok, out_mu, attention_mask_mu, tokens_mu, attns_mu
    gc.collect()
    torch.cuda.empty_cache()

    ##########################################################

    # compute embedding distance
    res['embedding_distance'] = ((emb_wt - emb_mu) ** 2).sum(axis=-1).type(torch.float).detach().cpu().numpy()
    # compute contact vector distances
    res['structural_distance'] = ( torch.sum(torch.abs(cvecs_wt-cvecs_mu), axis=1) / torch.sum(cvecs_wt+cvecs_mu, axis=1)).detach().cpu().numpy()
    return pd.DataFrame.from_dict(res)


def batch_extract_features(model, tokenizer, ids, wt_seqs, mu_seqs, masked_wt, masked_mu, wmax=None):
    # call esm_features on batches of data
    batches = make_batches(wt_seqs, mu_seqs, masked_wt, masked_mu, ids, wmax)

    return pd.concat([esm_features(model, tokenizer, id_b, wt_b, mu_b, m_wt_b, m_mu_b)
                      for (wt_b, mu_b, m_wt_b, m_mu_b, id_b) in tqdm(batches)])


def make_batches(wt, mu, m_wt, m_mu, ids, wmax=None):
    seq_len = [len(s) for s in wt]  # get the array of protein lengths
    sort = np.argsort(seq_len)[::-1]  # sort the length in descending order

    # sort the data according to protein length
    sorted_wt = [wt[i] for i in sort]
    sorted_m_wt = [m_wt[i] for i in sort]
    sorted_m_mu = [m_mu[i] for i in sort]
    sorted_mu = [mu[i] for i in sort]
    sorted_ids = [ids[i] for i in sort]
    sorted_lens = [seq_len[i] for i in sort]

    # set the maximum weight as the len of the biggest sequence
    if wmax is None:
        wmax = max(seq_len)
    assert wmax >= max(seq_len)  # if there is a sequence with len > wmax assertion will rise

    i = 0
    count = 0
    res = []
    while i < len(wt):  # loop untill all sequences are batched
        batch_wt = []
        batch_mu = []
        batch_ids = []
        batch_m_wt = []
        batch_m_mu = []
        wtot = 0
        while i < len(wt) and wtot + sorted_lens[i] <= wmax:  # untill there is free space
            wtot += sorted_lens[i]  # update weight

            # prepare batch
            batch_wt.append(sorted_wt[i])
            batch_mu.append(sorted_mu[i])
            batch_ids.append(sorted_ids[i])
            batch_m_wt.append(sorted_m_wt[i])
            batch_m_mu.append(sorted_m_mu[i])
            i += 1
        # append batch
        res.append((batch_wt, batch_mu, batch_m_wt, batch_m_mu, batch_ids))
    return res



#%%



# get input and output file path
sample_name = os.environ['SAMPLE_NAME']
in_file = f'output/{sample_name}/masked_sequences.csv'
ou_file = f'output/{sample_name}/esm_features.csv'

# read input file and filter out too long sequences
X = pd.read_csv(in_file)
X = X.loc[[max((len(s1), len(s2))) <= 2000 for _, (s1, s2) in X[['wt', 'mu']].iterrows()], :]

# prepare input for esm feature extraction
ids = list(X['id'])
wt = list(X.wt)
mu = list(X.mu)
masked_wt = list(X.masked_wt)
masked_mu = list(X.masked_mu)

# load model
model_name = 'esm2_t33_650M_UR50D'
tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}', device_map='auto', torch_dtype=torch.bfloat16)
model = EsmForMaskedLM.from_pretrained(f'facebook/{model_name}', device_map='auto', torch_dtype=torch.bfloat16)

# run feature extraction
model.eval()
with torch.no_grad(), torch.amp.autocast('cuda'):
    res = batch_extract_features(model, tokenizer, ids, wt, mu, masked_wt, masked_mu, wmax=1000)

res.to_csv(ou_file)  # save result

