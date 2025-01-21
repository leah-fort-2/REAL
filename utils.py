def do_unfold_dict(d, prefix=''):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from do_unfold_dict(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}: {value}"

def unfold(d):
    for item in do_unfold_dict(d):
        print(item)
