import json

from genetic import *


def export_population(file: str, population: List[List[Rule]]):
    with open(file, "w") as f:
        json.dump(population, f, sort_keys=True)


def load_population(src_file: str) -> List[List[Rule]]:
    with open(src_file, "r") as f:
        tmp_pop = json.load(f)
    return [[Rule(g[0], g[1], g[2], g[3], g[4]) for g in i] for i in tmp_pop]