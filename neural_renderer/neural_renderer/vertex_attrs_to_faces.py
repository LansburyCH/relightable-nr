import torch


def vertex_attrs_to_faces(vertex_attrs, faces):
    """
    :param vertex_attrs: [batch size, number of vertices, num of attrs]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, num of attrs]
    """
    assert (vertex_attrs.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertex_attrs.shape[0] == faces.shape[0])
    assert (faces.shape[2] == 3)

    bs, nv, na = vertex_attrs.shape
    bs, nf = faces.shape[:2]
    device = vertex_attrs.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertex_attrs = vertex_attrs.reshape((bs * nv, na))
    # pytorch only supports long and byte tensors for indexing
    return vertex_attrs[faces.long()]
