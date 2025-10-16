from typing import Dict 
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor, is_attn: bool = False):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    max_order = cache_dic['max_order'] if is_attn else 0
    print(f"max_order: {max_order}")
    for i in range(max_order):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def del_cache(cache_dic: Dict, current: Dict):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    updated_taylor_factors = {}
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def saving_sparse_info(cache_dic: Dict, current: Dict, sparse_q_ratio, sparse_kv_ratio):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    
    cache_dic['cache'][-1][current['stream']][current['layer']]['sparse_ratio'] = [sparse_q_ratio, sparse_kv_ratio]


def get_sparse_info(cache_dic: Dict, current: Dict):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    return cache_dic['cache'][-1][current['stream']][current['layer']]['sparse_ratio']




def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    #x = current['t'] - current['activated_times'][-1]
    output = 0

    # return cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0]
    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    
    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}