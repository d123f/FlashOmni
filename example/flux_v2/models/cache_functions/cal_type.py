from .force_scheduler import force_scheduler
from ..taylorseer_utils import get_sparse_info

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if (cache_dic['fresh_ratio'] == 0.0) and (not cache_dic['taylor_cache']):
        # FORA:Uniform
        first_step = (current['step'] == 0)
    else:
        # ToCa: First enhanced
        first_step = (current['step'] < cache_dic['first_enhance'])
        #first_step = (current['step'] <= 3)

    last_step = (current['step'] == current['num_steps'] - 1)

    current['warmup'] = first_step
    if not first_step:
        fresh_interval = cache_dic['cal_threshold']
    else:
        fresh_interval = cache_dic['fresh_threshold']

    if first_step:
        if (current['step'] == cache_dic['first_enhance'] - 1):
            current['sparseqkv'] = True
        else:
            current['sparseqkv'] = False

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1 ) or last_step:
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        
        #current['activated_times'].append(current['t'])
        force_scheduler(cache_dic, current)
    
    elif (cache_dic['taylor_cache']):
        cache_dic['cache_counter'] += 1
        current['type'] = 'Sparse'


def cal_type_sparse(cache_dic, current):
    if current['type'] == 'Sparse':
        sparse_q_ratio, sparse_kv_ratio = get_sparse_info(cache_dic, current)
        if cache_dic['cache_index']['taylor_start'][current['stream']][current['layer']]:
            current['sparse_type'] = "taylor_cache"
        else:
            current['sparse_type'] = 'sparseqkv'
            
        print(current['step'], sparse_q_ratio, sparse_kv_ratio, current['layer'], current['type'], current['sparse_type'], current['stream'])
