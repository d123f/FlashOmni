from diffusers.models import FluxTransformer2DModel

def cache_init(self: FluxTransformer2DModel):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index['taylor_start'] = {}

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_index['taylor_start']['double_stream'] = {}
    cache_index['taylor_start']['single_stream'] = {}
    cache_dic['cache_counter'] = 0

    #for j in range(19):
    for j in range(self.config.num_layers):
        cache[-1]['double_stream'][j] = {}
        cache_index['taylor_start']['double_stream'][j] = False

    #for j in range(38):
    for j in range(self.config.num_single_layers):
        cache[-1]['single_stream'][j] = {}
        cache_index['taylor_start']['single_stream'][j] = False

    cache_dic['taylor_cache'] = False

    mode = 'SparseQKV'

    if mode == 'SparseQKV':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 5 # <-- N 
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = 2  # <-- O 
        cache_dic['first_enhance'] = 8
        # cache_dic['threshold_q'] = 0.5
        # cache_dic['threshold_kv'] = 0.70
        cache_dic['threshold_q'] = 0.50
        cache_dic['threshold_kv'] = 0.15
        cache_dic['saving_threshold_q_for_taylor'] = 0.3
        cache_dic['max_sequence_length'] = 512
        cache_dic['sparseqkv_wrapper'] = self.sparseqkv_wrapper
        
    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = self.num_steps
    print(f"cache_dic settings: {cache_dic}")
    print(f"Total num_steps: {current['num_steps']}")
    
    return cache_dic, current