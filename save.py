import json
from collections import namedtuple
from typing import List

Rule = namedtuple("Rule", "arity param1 param2 param3 param4")


def export_population(file: str, population: List[List[Rule]]):
    with open(file, "w") as f:
        json.dump(population, f, sort_keys=True)


def load_population(src_file: str) -> List[List[Rule]]:
    with open(src_file, "r") as f:
        tmp_pop = json.load(f)
        print(tmp_pop)
    return [[Rule(g[0], g[1], g[2], g[3], g[4]) for g in i if len(i)!=0] for i in tmp_pop]
