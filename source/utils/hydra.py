from typing import Any, List
from omegaconf import OmegaConf

def ensure_list(value: Any) -> List:
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)
    else:
        return [value]

def cat_resolver(*objs) -> List:
    ll= [elem for obj in objs for elem in ensure_list(obj)]
    return ll

def prod_resolver(*x):
    p = x[0]
    for elem in x[1:]:
        p *= elem
    return p

def listmult_resolver(list_value: List[Any], times: int) -> List[Any]:
    assert isinstance(list_value, list), f"Expected a list, got {type(list_value)}"
    return list_value * times

def register_resolvers():
    # ${neg:-4} -> 4
    OmegaConf.register_new_resolver(name='neg', resolver=lambda x: -x)
    # ${in:2,[1,2,3]} -> True
    OmegaConf.register_new_resolver(name='in', resolver=lambda x, a: x in a)
    # ${not:True} -> False
    OmegaConf.register_new_resolver(name='not', resolver=lambda x: not x)
    # ${sum:1,2,3,4} -> 10
    OmegaConf.register_new_resolver(name='sum', resolver=lambda *x: sum(x))
    # ${prod:1,2,3,4} -> 24
    OmegaConf.register_new_resolver(name='prod', resolver=prod_resolver)
    # ${div:1,4} -> 0.25
    OmegaConf.register_new_resolver(name='div', resolver=lambda x, d: x / d)
    # ${exp:3,2} -> 9
    OmegaConf.register_new_resolver(name='exp', resolver=lambda x, e: x**e)
    # ${cat:1,2,[3,[4]]} -> [1,2,3,[4]]
    OmegaConf.register_new_resolver(name='cat', resolver=cat_resolver)
    # ${merge:[1,2,3],[4,5]} -> [1,2,3,4,5]
    OmegaConf.register_new_resolver("merge", lambda x, y : x + y)
    # ${listprod: [2], 3} -> [2, 2, 2]
    OmegaConf.register_new_resolver("listmult", listmult_resolver)
    # ${round:1.3} -> 1
    OmegaConf.register_new_resolver(name='round', resolver=lambda x: round(x))