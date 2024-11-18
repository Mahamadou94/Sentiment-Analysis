def extract_annotations(annotation):
    ann = set()
    for elem in annotation:
        result = elem["result"]
        if len(result) > 0:
            label = set(result[0]["value"]["choices"])
        else:
            label = set()
        ann.update(label)
    return list(ann)
