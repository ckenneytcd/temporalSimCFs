import json


def load_fact(fact_file):
    with open(fact_file, 'r') as f:
        content = json.load(f)

    target = content['target']

    return content, target