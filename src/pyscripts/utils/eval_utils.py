import numpy as np
import pandas as pd
import math
from scipy.stats import kendalltau


def kendalls_tau_scipy(rank1, rank2):
    '''
    Wrapper function to discard 2nd returned argument 
    of SciPy's Kendall's Tau implementation
    '''
    kt, p = kendalltau(rank1, rank2)
    return kt

    def calculate_kendall_tau_distance_simple(permutation1, permutation2) -> int:
        '''
        Ke Yang function for simple computation of normalized Kendall's Tau
        '''
        item_to_rank = {item: rank for rank, item in enumerate(permutation1)}
        m = len(permutation1)
        dist = 0
        for i, e_i in enumerate(permutation2[:-1]):
            rank_e_i_in_permutation1 = item_to_rank[e_i]
            for e_j in permutation2[i + 1:]:
                if rank_e_i_in_permutation1 > item_to_rank[e_j]:
                    dist += 1
        return dist /(m * (m-1)/2)

def calculate_kendall_tau_distance_quick(x, y):
    """
    Ke Yang function for fast computation of normalized Kendall's Tau

    Code from Scipy. See details in
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/stats.py#L2827

    """

    x = np.asarray(x)
    y = np.asarray(y)

    m = x.size
    reference = np.arange(m)
    
    if (not np.equal(x, reference).all()) and (not np.equal(y, reference).all()):
        x_to_rank = {item: rank for rank, item in enumerate(x)}
        x = reference  # For unknown reason, either x or y must be counter ranking.
        y = np.asarray([x_to_rank[item] for item in y])

    n = np.int64(len(x))
    temp = list(range(n))  # support structure used by mergesort

    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support
    # returns the number of swaps required by an equivalent bubble sort

    def mergesort(offs, length):
        exchcnt = 0
        if length == 1:
            return 0
        if length == 2:
            if y[perm[offs]] <= y[perm[offs + 1]]:
                return 0
            t = perm[offs]
            perm[offs] = perm[offs + 1]
            perm[offs + 1] = t
            return 1
        length0 = length // 2
        length1 = length - length0
        middle = offs + length0
        exchcnt += mergesort(offs, length0)
        exchcnt += mergesort(middle, length1)
        if y[perm[middle - 1]] < y[perm[middle]]:
            return exchcnt
        # merging
        i = j = k = 0
        while j < length0 or k < length1:
            if k >= length1 or (j < length0 and y[perm[offs + j]] <=
                                y[perm[middle + k]]):
                temp[i] = perm[offs + j]
                d = i - j
                j += 1
            else:
                temp[i] = perm[middle + k]
                d = (offs + i) - (middle + k)
                k += 1
            if d > 0:
                exchcnt += d
            i += 1
        perm[offs:offs + length] = temp[0:length]
        return exchcnt

    # initial sort on values of x and, if tied, on values of y
    perm = np.lexsort((y, x))

    # count exchanges
    exchanges = mergesort(0, n)

    return exchanges/(m * (m-1)/2)

def num_retained_at_top_k(rank1, rank2, k=None):

    '''
    Return size of intersection at top k
    Default k is 20% of m (number of items)
    '''
    
    if not k:
        k=int(0.2*len(rank1))

    top_k_1= [i for i,x in enumerate(rank1) if x<=k]
    top_k_2= [i for i,x in enumerate(rank2) if x<=k]
    num_retained = len(set(top_k_1) & set(top_k_2))
    
    return num_retained/len(rank1)


def compute_k_recall(true_list, pred_list, batch_size = 10):
    if len(true_list):
        if batch_size == len(true_list):
            return len(set(true_list).intersection(pred_list)) / len(true_list)
        else:
            res = 0
            for i in range(batch_size, len(true_list)+1, batch_size):
                res += len(set(true_list[0:i]).intersection(pred_list[0:i])) / i
            return round(res / len(true_list), 3)
    else:
        return 0

def compute_ap(true_list, pred_list, k, batch_size=1):
    # if len(pred_list) != k:
    #     print("AP input wrong, not equal ", k)
    #     exit()
    if len(pred_list) > 0:
        res_ap = 0
        for i in range(batch_size, len(pred_list) + 1, batch_size):
            if pred_list[i - 1] in true_list:
                res_ap += (len(set(true_list[0:i]).intersection(pred_list[0:i])) / i)
        return round(res_ap / k, 3)
    else:
        if len(true_list) > 0:
            return 0
        else:
            return -1

def compute_score_util(top_k_IDS, _orig_df, _orig_sort_col, opt_u=None):
    if not opt_u:
        if len(top_k_IDS) > 0:
            opt_u = sum(_orig_df.head(len(top_k_IDS))[_orig_sort_col])
    if len(top_k_IDS) > 0:
        cur_u = sum([_orig_df[_orig_df["UID"] == x][_orig_sort_col].values[0] for x in top_k_IDS])
        return round(1-(cur_u/opt_u), 3)
    else:
        return -1

def KL_divergence(p_list, q_list, log_base=2):
    res = 0
    for pi, qi in zip(p_list, q_list):
        res += pi*math.log(pi/qi,log_base)
    return res


def compute_rKL(_top_df, _orig_df, sort_col=None, group_col="GR", cut_off=10, log_base=2):
    if sort_col: # random permutation within ties in _top_df
        print("!!!Ties in the ranking!!!")
        rand_top_df = pd.DataFrame(columns=list(_top_df.columns))
        for yi in _top_df[sort_col].unique():
            yi_df = _top_df[_top_df[sort_col]==yi].copy().sample(frac=1).reset_index(drop=True)
            rand_top_df = pd.concat([rand_top_df, yi_df])
    else:
        rand_top_df = _top_df.copy()
    base_quotas = dict(_orig_df[group_col].value_counts(normalize=True))
    res = 0
    for ci in range(cut_off, rand_top_df.shape[0], cut_off):
        ci_quotas = dict(rand_top_df.head(ci)[group_col].value_counts(normalize=True))
        ci_p_list = []
        ci_q_list = []
        for gi, gi_v in base_quotas.items():
            if gi in ci_quotas:
                ci_p_list.append(ci_quotas[gi])
            else:
                ci_p_list.append(0.001) # to compute the KL-diverfence for value 0
            if gi_v == 0:
                ci_q_list.append(0.001)
            else:
                ci_q_list.append(base_quotas[gi])
        res += KL_divergence(ci_p_list, ci_q_list, log_base=log_base) / math.log(ci + 1, log_base)
    return res



def compute_igf_ratio(top_k_IDS, _orig_df, _orig_sort_col):
    # assume _orig_df is sorted according to the _orig_sort_col
    cur_res = min(_orig_df[_orig_df["UID"].isin(top_k_IDS)][_orig_sort_col]) / max(_orig_df[~_orig_df["UID"].isin(top_k_IDS)][_orig_sort_col])
    if cur_res > 1:
        return 1
    else:
        return cur_res