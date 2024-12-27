import json


def load_json(datadir: str):
    with open(datadir, 'r') as f:
        return json.load(f)
    
def load_jsonl(datadir: str):
    output = []
    with open(datadir) as f:
        for line in f.readlines():
            output.append(json.loads(line))
    return output

def load_txt(datadir: str):
    with open(datadir, 'r') as f:
        return f.read()