import random
from collections import namedtuple
from typing import Tuple, List

POPULATION_SIZE = 20
RANDOM_RULE_CHANCE = 10
MUTATE_ARITY_CHANCE = 30
MUTATE_PARAM_CHANCE = 50
MUTATE_DOUBLE_CHANCE = 20


Rule = namedtuple("Rule", "arity param1 param2 param3 param4")


def mutate_rule(rule: Rule):
    mutation = random.randint(0,100)
    arity = getattr(rule,'arity')
    param1 = getattr(rule,'param1')
    param2 = getattr(rule,'param2')
    param3 = getattr(rule,'param3')
    param4 = getattr(rule,'param4')
    if mutation < MUTATE_ARITY_CHANCE:
        if arity == 'U':
            arity = 'B'
        if arity == 'B':
            arity = '*'
        if arity == '*':
            arity = 'U'
    if mutation >= MUTATE_ARITY_CHANCE & mutation < MUTATE_PARAM_CHANCE:
            param1+=random.randint(-1,1)
            param2+=random.randint(-1,1)
    if mutation >= MUTATE_ARITY_CHANCE+MUTATE_PARAM_CHANCE & mutation <MUTATE_DOUBLE_CHANCE:
            param3 = param1
            param4 = 0
    return Rule(arity,param1,param2,param3,param4)


def add_random_rule(individual: List[Rule]):
    individual+=(create_rule())
    pass


def mutate(individual: List[Rule]) -> None:
    for rule in individual:
        mutate_rule(rule)
    if random.randint(0, 100) < RANDOM_RULE_CHANCE:
        add_random_rule(individual)


def cross_individuals(i1: list, i2: list) -> Tuple[List, List]:
    total = i1 + i2
    random.shuffle(total)
    cut = random.randint(1, len(total))
    return total[:cut], total[cut:]


def create_rule() -> Rule:
    arity = 'U' if random.randint(0, 5) < 3 else 'B' if random.randint(0,5) < 3 else '*'
    param1 = random.randint(-2, 2)
    param2 = 0
    return Rule(arity, param1, param2, None, None)


def create_individual() -> List[Rule]:
    size = random.randint(1, 6)
    return [create_rule() for x in range(size)]


def initialize_population() -> List[List[Rule]]:
    return [create_individual() for x in range(POPULATION_SIZE)]


if __name__ == '__main__':
    pop = initialize_population()
    print(pop)

    crossed_pop = []
    while len(crossed_pop) < POPULATION_SIZE:
        random.shuffle(pop)
        for i1, i2 in zip(pop[:2], pop[2::4]):
            a, b = cross_individuals(i1, i2)
            crossed_pop.append(a)
            crossed_pop.append(b)
    print(crossed_pop)
    print(len(crossed_pop))
