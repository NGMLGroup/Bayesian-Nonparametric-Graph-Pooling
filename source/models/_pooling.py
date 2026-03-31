from tgp.poolers import get_pooler


POOLERS_WITH_K = {'bnpool', 'diff', 'dmon', 'mincut', 'jb', 'hosc', 'eigen'}


def build_pooler(pooler, in_channels, pooled_nodes, pool_kwargs):
    pool_kwargs = dict(pool_kwargs or {})
    if pooler in POOLERS_WITH_K:
        pool_kwargs.setdefault('k', pooled_nodes)
    return get_pooler(pooler, in_channels=in_channels, **pool_kwargs)
