import json


def read_manifest(manifest_file):
    with open(manifest_file, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(manifest_file, data, append=False):
    mode = 'a' if append else 'w'
    with open(manifest_file, mode) as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')