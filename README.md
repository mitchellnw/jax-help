# Jax Help

1. `base.py` is a simple MNIST training example. For NN `f(x,w)` we train `w`.

2. `lines.py` is MNIST for training a range of params. We are trying to train `w_a` and `w_b` such
that `f(x, (1-t) * w_a + t * w_b)` gives high accuracy for any `t`. This works. For each
batch just ranodmly choose a `t`.

3. `ftoc_lines.py` is like `lines.py`, except our logits are supposed to be given
by `df/dt`. In `lines.py` the logits are given by `f`. For some reason, `ftoc_lines`
gets extremely slow after the first four batches, probably due to some `jit` issues.

Any help would be extremely appreciated.