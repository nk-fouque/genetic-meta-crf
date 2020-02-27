#!/usr/bin/python3
import subprocess

from genetic import *

Rule = namedtuple("Rule", "arity param1 param2 param3 param4")

# https://stackoverflow.com/questions/13332268/how-to-use-subprocess-command-with-pipes

# Currently the model file must already exist
# TODO generate model file from individual


WAPITI_LOCATION = "wapiti-1.5.0/wapiti"
TEMPLATE_FILE = "generated_files/template"
DATASET = "atis.train"
MODEL_FILE = "generated_files/modele"
LABEL_FILE = "generated_files/labels"
EVAL_FILE = "generated_files/reseval"

Evaluation = namedtuple("Evaluation", "accuracy precision recall f1")


def write_rule(rule: Rule, name: str):
    arity = getattr(rule, 'arity')
    param1 = getattr(rule, 'param1')
    param2 = getattr(rule, 'param2')
    param3 = getattr(rule, 'param3')
    param4 = getattr(rule, 'param4')
    res = arity + name + '%x[' + str(param1) + ',' + str(param2) + ']'
    if param3 is not None:
        res += '/%x[' + str(param3) + ',' + str(param4) + ']'
    return res


def generate_template(individual: List[Rule], id: str):
    f = open(TEMPLATE_FILE + id, "w+")
    for i, rule in enumerate(individual):
        f.write(write_rule(rule, str(i)) + "\n")
    f.close()


def fitness_population(population: List[List[Rule]]) -> List[Evaluation]:
    return [fitness_individual(individual, str(id)) for id, individual in enumerate(population)]


def fitness_individual(individual: List[Rule], id: str) -> Evaluation:
    generate_template(individual, str(id))
    try:
        subprocess.check_call(
            WAPITI_LOCATION + ' train -p ' + TEMPLATE_FILE + id + ' -t 8 ' + DATASET + ' ' + MODEL_FILE + id,
            shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output)

    try:
        subprocess.check_call(
            WAPITI_LOCATION + ' label -m ' + MODEL_FILE + id + ' <' + DATASET + ' > ' + LABEL_FILE + id, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output)

    try:
        subprocess.check_call('cat ' + LABEL_FILE + id + ' | perl evaluation.pl > ' + EVAL_FILE + id, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output)

    with open(EVAL_FILE + str(id), "r") as f:
        for i, line in enumerate(f):
            if i == 1:
                l = line.split()
                accuracy = l[1]
                precision = l[3]
                recall = l[5]
                f1 = l[7]
                break
    return Evaluation(accuracy, precision, recall, f1)


if __name__ == '__main__':
    ind = create_individual()
    print(fitness_individual(ind, '0'))
