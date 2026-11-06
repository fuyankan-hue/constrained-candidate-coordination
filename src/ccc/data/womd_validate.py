"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
import numpy as np

def tensor(sample, key, shape, dtype=None):
    t = sample[key]
    assert isinstance(t, torch.Tensor), key
    assert tuple(t.shape) == tuple(shape), (key, tuple(t.shape), shape)
    if dtype is not None:
        assert t.dtype == dtype, (key, t.dtype)
    assert torch.isfinite(t).all().item(), key
    return t


def validate_m2(b, sid):
    assert str(b['scene']) == sid, (sid, b['scene'])
    c = len(b['context_ids'])
    ids = tensor(b, 'context_ids', (c,), torch.int64)
    assert c > 0 and len(set(ids.tolist())) == c and (ids >= 0).all()
    t = len(b['target_ids'])
    targets = tensor(b, 'target_ids', (t,), torch.int64)
    assert len(set(targets.tolist())) == t and set(targets.tolist()) <= set(ids.tolist())
    for k in ('near', 'far', 'refine'):
        tensor(b, k, (c, 6, 128), torch.float32)
    tensor(b, 'world', (c, 6, 80, 2), torch.float32)
    tensor(b, 'current', (c, 2), torch.float32)
    tensor(b, 'pi', (c, 6), torch.float32)
    target_mask = tensor(b, 'is_target', (c,), torch.bool)
    assert torch.equal(target_mask, torch.isin(ids, targets))


def validate_m1(a, sid, b=None):
    assert str(a['scene']) == sid, (sid, a['scene'])
    n = len(a['original_agent_id'])
    ids = tensor(a, 'original_agent_id', (n,), torch.int64)
    assert len(set(ids.tolist())) == n and (ids >= 0).all()
    for k, shape in {'unary': (n,6,9), 'pair': (n,6,8), 'pi': (n,6), 'fde': (n,6),
                     'world': (n,6,80,2), 'gt': (n,80,2)}.items():
        tensor(a, k, shape, torch.float32)
    valid = tensor(a, 'valid', (n,80), torch.bool)
    assert torch.equal(tensor(a, 'valid8', (n,), torch.bool), valid[:,79])
    err = torch.linalg.vector_norm(a['world'][:,:,79] - a['gt'][:,None,79], dim=-1)
    assert torch.allclose(err, a['fde'], atol=1e-5, rtol=1e-5)
    assert (a['fde'] >= 0).all()
    if b is not None:
        validate_m2(b, sid)
        assert torch.equal(ids, b['target_ids']), (sid, 'target id mismatch')
        mapping = {int(v): i for i,v in enumerate(b['context_ids'].tolist())}
        assert all(int(v) in mapping for v in ids.tolist())
        rows = torch.tensor([mapping[int(v)] for v in ids.tolist()], dtype=torch.long)
        for k in ('world', 'pi'):
            assert torch.allclose(a[k], b[k][rows], atol=1e-3, rtol=1e-4), (sid, k, 'M1/M2 predictions differ', float((a[k]-b[k][rows]).abs().max()))
    return n

