"""Semantically-keyed random number source.

The central problem this solves: two guidance implementations that are
mathematically identical can still diverge numerically if they *consume* the
global RNG in a different order or a different number of times.  That happens
constantly here -- ``N_iter=0`` skips the mean-guidance loop, ``gamma_bar=0``
may short-circuit the smoothing draws, ``N_recur=1`` skips re-noising -- and
every such branch shifts all downstream noise.

A ``NoiseTape`` removes order from the picture entirely.  Noise is requested by
a *semantic key* such as ``("delta", t, j)``.  The same key always yields the
same tensor, regardless of when (or whether) the other engine asked for it.

Keys are hashed with blake2b rather than :func:`hash`, because Python salts
string hashing per process (``PYTHONHASHSEED``) and the tape must be
reproducible across runs.
"""

import hashlib

import torch

# Keys are tuples of these types only, so that repr() is a stable serialisation.
_ALLOWED_KEY_ATOMS = (str, int, bool, type(None))


def _canonical(key):
    """Normalise a key to a tuple and validate that it serialises stably."""
    if not isinstance(key, tuple):
        key = (key,)
    for atom in key:
        if not isinstance(atom, _ALLOWED_KEY_ATOMS):
            raise TypeError(
                f"NoiseTape keys may only contain {_ALLOWED_KEY_ATOMS}, "
                f"got {type(atom)!r} in key {key!r}. Floats and tensors are "
                "rejected because their repr is not a stable identity."
            )
    return key


class NoiseTape:
    """Reproducible, order-independent source of standard normal noise.

    Parameters
    ----------
    seed:
        Master seed. Two tapes with the same seed serve identical tensors for
        identical keys.
    device, dtype:
        Defaults for materialised tensors. Noise is always *generated* on CPU
        in float64 and then cast, so that a CPU run and a GPU run of the same
        seed agree bit-for-bit in the generated values.
    """

    def __init__(self, seed, device="cpu", dtype=torch.float64):
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self._cache = {}
        self._meta = {}
        self.access_log = []

    # -- internals ---------------------------------------------------------

    def _key_seed(self, key):
        payload = repr((self.seed,) + key).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        # torch.Generator.manual_seed accepts a signed 64-bit value; keep it
        # comfortably inside the positive range.
        return int.from_bytes(digest, "big") % (2**63 - 1)

    # -- public API --------------------------------------------------------

    def randn(self, key, shape, device=None, dtype=None):
        """Return standard normal noise for ``key``, drawing it on first use.

        Repeated requests for the same key return the *same* tensor. A repeat
        request with a different shape is an error rather than a silent
        re-draw: it almost always means the two engines disagree about what the
        key denotes.
        """
        key = _canonical(key)
        shape = tuple(int(s) for s in shape)
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype

        self.access_log.append(key)

        if key in self._cache:
            prev_shape = self._meta[key]
            if prev_shape != shape:
                raise ValueError(
                    f"NoiseTape key {key!r} was first requested with shape "
                    f"{prev_shape} and is now requested with shape {shape}. "
                    "The two callers disagree about this key's meaning."
                )
            master = self._cache[key]
        else:
            generator = torch.Generator(device="cpu").manual_seed(self._key_seed(key))
            master = torch.randn(shape, generator=generator, dtype=torch.float64)
            self._cache[key] = master
            self._meta[key] = shape

        # The cache always holds the float64 master draw and casts on the way
        # out.  Caching the *first requested* dtype instead would make the
        # served values depend on request order across dtypes -- exactly the
        # order dependence this class exists to eliminate.
        return master.to(device=device, dtype=dtype)

    def keys(self):
        """Set of keys that have been materialised."""
        return set(self._cache)

    def requested_keys(self):
        """Set of keys that have been *requested*, in any order."""
        return set(self.access_log)

    def reset_log(self):
        self.access_log = []

    def fork(self):
        """A tape with the same seed but an empty log, sharing no cache."""
        return NoiseTape(self.seed, device=self.device, dtype=self.dtype)


def compare_access(tape_a, tape_b):
    """Diff two tapes' requested keys.

    Returns ``(only_in_a, only_in_b)`` as sorted lists. Order is deliberately
    *not* compared -- order-independence is the whole point of the tape -- but
    a key requested by one engine and not the other is a real discrepancy and
    is what this surfaces.
    """
    a, b = tape_a.requested_keys(), tape_b.requested_keys()
    return sorted(a - b, key=repr), sorted(b - a, key=repr)
