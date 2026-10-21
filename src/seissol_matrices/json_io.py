import numpy as np
import json
import os
import re
import itertools


def read_json(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    file_txt = " ".join([line for line in lines])
    return json.loads(file_txt)


def write_json(content, filename):
    file_content = json.dumps(content, indent="  ")
    file_content = re.sub(r"\n  \s+", " ", file_content)
    with open(filename, "w+") as f:
        f.write(file_content)


def create_empty_json_file(filename):
    write_json([], filename)


def _entries(tensor):
    """The entries above 1e-14 of a tensor, as 1-based indices and the value."""
    entries = []
    for index in itertools.product(*[range(dim) for dim in tensor.shape]):
        if np.abs(tensor[tuple(index)]) > 1e-14:
            entries.append([i + 1 for i in index] + [str(tensor[tuple(index)])])
    return entries


def write_tensor(tensor, tensorname, filename):
    if not os.path.exists(filename):
        create_empty_json_file(filename)

    tensor_list = read_json(filename)
    assert isinstance(tensor_list, list)

    m = {"name": tensorname, "shape": tensor.shape, "entries": _entries(tensor)}
    tensor_list.append(m)
    write_json(tensor_list, filename)


def write_tensors(tensors, filename):
    """Writes the tensors {name: array} into one file at once, appending them to
    the tensors the file has."""
    tensor_list = read_json(filename) if os.path.exists(filename) else []
    assert isinstance(tensor_list, list)
    for tensorname, tensor in tensors.items():
        tensor = np.asarray(tensor)
        tensor_list.append(
            {"name": tensorname, "shape": tensor.shape, "entries": _entries(tensor)}
        )
    write_json(tensor_list, filename)


def read_tensor(tensorname, filename):
    tensor_list = read_json(filename)
    for raw_tensor in tensor_list:
        if raw_tensor["name"] == tensorname:
            # we still support the old format here
            if "rows" in raw_tensor:
                shape = [int(raw_tensor["rows"])]
                if "columns" in raw_tensor:
                    shape += [int(raw_tensor["columns"])]
            else:
                shape = [int(dim) for dim in raw_tensor["shape"]]

            dims = len(shape)
            tensor = np.zeros(shape)
            for entry in raw_tensor["entries"]:
                index = tuple(entry[i] - 1 for i in range(dims))
                value = entry[-1]
                tensor[index] = value
            return tensor
    raise ValueError(f"Tensor with name {tensorname} not found in {filename}.")


# legacy function names


def write_matrix(tensor, tensorname, filename):
    write_tensor(tensor, tensorname, filename)


def read_matrix(tensorname, filename):
    return read_tensor(tensorname, filename)
