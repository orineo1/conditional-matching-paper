"""Real-model twisted-SMC for discrete-x CDM.

Replaces the sanity task's two toy components with real models, behind the
same SMC skeleton and the same two x0_hat estimators (mode / sampled):

  Prior p_theta(x_0 | x_t): a masked LM (BERT family) used as the denoiser of
    absorbing discrete diffusion over a fixed-length token canvas following a
    fixed prefix ("a photo of ..."). Masked-LM training IS the masked-diffusion
    denoising objective (MDLM/DiffusionBERT observation), so per-position
    logits at [MASK] slots are the factorized posterior marginals.

  Loss L(x_0), consumed strictly as a scalar (no gradients w.r.t. x):
    --loss text  : 1 - cosine(CLIP text emb of decoded prompt, CLIP text emb
                   of the hidden ground-truth prompt). Cheap; runs locally on
                   MPS/CPU. Integration test for the real-LM prior.
    --loss image : biased MMD^2 in CLIP image-embedding space between n_cond
                   SDXL-Turbo images of the decoded prompt and a fixed target
                   set generated from the ground-truth prompt. Common random
                   numbers: the same initial latents are reused for every
                   prompt, so L is deterministic and memoizable. Cluster/GPU.

Usage (local text-mode comparison, both estimators):
    conda run -n grassy_dit python discrete_x/real_smc.py --loss text \
        --beta 60 --beta_anneal --outdir output/discrete_x_real_text

Cluster image mode: see scripts/submit_discrete_smc.sh
"""

import argparse
import json
import os
import time

import numpy as np
import torch

MASK = -1


def logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def mmd2_biased(X, Y, gamma, kyy_mean=None):
    def sqdist(A, B):
        return (
            np.sum(A**2, axis=1)[:, None]
            + np.sum(B**2, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )

    kxx = np.exp(-gamma * sqdist(X, X)).mean()
    kxy = np.exp(-gamma * sqdist(X, Y)).mean()
    if kyy_mean is None:
        kyy_mean = np.exp(-gamma * sqdist(Y, Y)).mean()
    return float(kxx + kyy_mean - 2.0 * kxy)


# ---------------------------------------------------------------- prior
class BertMaskedPrior:
    """Masked LM as the exact-interface replacement for the toy denoiser:
    batch_marginals(X) -> (B, canvas_len, A) probs over an allowed sub-vocab
    (whole-word lowercase alphabetic tokens, so decodes are readable prompts).
    Canvas entries index into self.allowed; MASK = -1."""

    def __init__(self, model_name, device, prefix_text, canvas_len,
                 clip_tok=None, vocab_words=None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        )
        self.device = device
        self.canvas_len = canvas_len
        self.prefix_text = prefix_text
        self.prefix_ids = self.tok(prefix_text, add_special_tokens=False)[
            "input_ids"
        ]
        vocab = self.tok.get_vocab()
        if vocab_words is not None:
            # controlled vocabulary: the allowed alphabet is exactly this word
            # list (used for the V-ladder, where V is the only free variable)
            missing = [w for w in vocab_words if w not in vocab]
            assert not missing, f"not single tokens of {model_name}: {missing}"
            candidates = sorted((w, vocab[w]) for w in set(vocab_words))
        else:
            candidates = sorted(
                (s, i)
                for s, i in vocab.items()
                if s.isalpha() and s.islower() and len(s) > 1
            )
        if clip_tok is not None:
            # grad-tilt needs a one-hot-aligned map into CLIP's embedding
            # matrix, so keep only words that are single CLIP-BPE tokens too
            kept, clip_ids = [], []
            for s, i in candidates:
                cids = clip_tok(s, add_special_tokens=False)["input_ids"]
                if len(cids) == 1:
                    kept.append((s, i))
                    clip_ids.append(cids[0])
            candidates = kept
            self.clip_ids = np.array(clip_ids)
        else:
            self.clip_ids = None
        self.allowed = np.array([i for _, i in candidates])
        self._allowed_t = torch.tensor(self.allowed, device=device)
        self._id_to_idx = {int(i): k for k, i in enumerate(self.allowed)}
        self.A = len(self.allowed)

    def encode_canvas(self, words):
        """Words -> canvas indices; asserts each word is a single allowed token."""
        idx = []
        for w in words:
            ids = self.tok(w, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1 and ids[0] in self._id_to_idx, (
                f"'{w}' is not a single allowed token"
            )
            idx.append(self._id_to_idx[ids[0]])
        assert len(idx) == self.canvas_len
        return np.array(idx)

    def decode(self, x):
        words = [
            "[MASK]" if v == MASK else self.tok.convert_ids_to_tokens(
                int(self.allowed[v])
            )
            for v in x
        ]
        return f"{self.prefix_text} " + " ".join(words)

    @torch.no_grad()
    def batch_marginals(self, X):
        B = X.shape[0]
        mask_id = self.tok.mask_token_id
        rows = []
        for b in range(B):
            rows.append(
                [self.tok.cls_token_id]
                + self.prefix_ids
                + [
                    mask_id if v == MASK else int(self.allowed[v])
                    for v in X[b]
                ]
                + [self.tok.sep_token_id]
            )
        ids = torch.tensor(rows, device=self.device)
        logits = self.model(input_ids=ids).logits
        off = 1 + len(self.prefix_ids)
        sub = logits[:, off : off + self.canvas_len, :].index_select(
            -1, self._allowed_t
        )
        probs = torch.softmax(sub.float(), dim=-1).cpu().numpy().astype(np.float64)
        return probs / probs.sum(axis=-1, keepdims=True)


class LLaDAPrior:
    """LLaDA-8B-Base (trained masked-diffusion LM) as the denoiser, same
    interface as BertMaskedPrior. Requires 3 loader shims for its 4.x-era
    remote code under transformers 5.x (missing all_tied_weights_keys,
    tie_weights signature, config.use_cache). BPE tokenizer: allowed words
    are leading-space single tokens (' rain'), optionally intersected with
    single-CLIP-token words for grad-tilt."""

    MASK_ID = 126336

    def __init__(self, device, prefix_text, canvas_len, clip_tok=None,
                 name="GSAI-ML/LLaDA-8B-Base", chunk=32):
        import inspect

        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import (
            get_class_from_dynamic_module,
        )

        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        auto_map = cfg.auto_map if isinstance(cfg.auto_map, dict) else {}
        cls_ref = auto_map.get("AutoModel") or auto_map.get(
            "AutoModelForCausalLM")
        cls = get_class_from_dynamic_module(cls_ref, name)
        if not hasattr(cls, "all_tied_weights_keys"):
            cls.all_tied_weights_keys = property(lambda self: {})
        _orig_tie = cls.tie_weights
        if "missing_keys" not in inspect.signature(_orig_tie).parameters:
            cls.tie_weights = lambda s, *a, **k: _orig_tie(s)
        self.model = cls.from_pretrained(
            name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        for attr, val in [("use_cache", False), ("output_attentions", False),
                          ("output_hidden_states", False)]:
            if not hasattr(self.model.config, attr):
                setattr(self.model.config, attr, val)

        self.device = device
        self.canvas_len = canvas_len
        self.prefix_text = prefix_text
        self.chunk = chunk
        self.prefix_ids = self.tok(prefix_text, add_special_tokens=False)[
            "input_ids"]

        vocab = self.tok.get_vocab()
        candidates = []
        for s, i in sorted(vocab.items()):
            if s.startswith("Ġ") and s[1:].isalpha() and s[1:].islower() \
                    and len(s) > 2:
                candidates.append((s[1:], i))
        if clip_tok is not None:
            kept, clip_ids = [], []
            for w, i in candidates:
                cids = clip_tok(w, add_special_tokens=False)["input_ids"]
                if len(cids) == 1:
                    kept.append((w, i))
                    clip_ids.append(cids[0])
            candidates = kept
            self.clip_ids = np.array(clip_ids)
        else:
            self.clip_ids = None
        self.words = [w for w, _ in candidates]
        self.allowed = np.array([i for _, i in candidates])
        self._id_to_idx = {int(i): k for k, i in enumerate(self.allowed)}
        self.A = len(self.allowed)

    def encode_canvas(self, words):
        idx = []
        for w in words:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1 and ids[0] in self._id_to_idx, (
                f"'{w}' is not a single allowed token"
            )
            idx.append(self._id_to_idx[ids[0]])
        assert len(idx) == self.canvas_len
        return np.array(idx)

    def decode(self, x):
        words = ["[MASK]" if v == MASK else self.words[v] for v in x]
        return f"{self.prefix_text} " + " ".join(words)

    @torch.no_grad()
    def batch_marginals(self, X):
        B = X.shape[0]
        rows = []
        for b in range(B):
            rows.append(
                self.prefix_ids
                + [self.MASK_ID if v == MASK else int(self.allowed[v])
                   for v in X[b]]
            )
        ids = torch.tensor(rows, device=self.device)
        allowed_t = torch.tensor(self.allowed, device=self.device)
        off = len(self.prefix_ids)
        outs = []
        for c in range(0, B, self.chunk):
            logits = self.model(ids[c:c + self.chunk]).logits
            sub = logits[:, off:off + self.canvas_len, :].index_select(
                -1, allowed_t)
            outs.append(torch.softmax(sub.float(), dim=-1).cpu().numpy())
        probs = np.concatenate(outs, axis=0).astype(np.float64)
        return probs / probs.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- losses
class ClipTextCosineLoss:
    """L(prompt) = 1 - cosine(CLIP text emb, target prompt's CLIP text emb)."""

    def __init__(self, target_text, device,
                 clip_name="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPTokenizer

        self.tok = CLIPTokenizer.from_pretrained(clip_name)
        self.model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        self.device = device
        self.target = self._embed(target_text)
        self.cache = {}
        self.n_evals = 0

    @torch.no_grad()
    def _embed(self, text):
        ids = self.tok(text, return_tensors="pt", padding=True,
                       truncation=True).to(self.device)
        out = self.model.get_text_features(**ids)
        # transformers 5.x returns an output object whose pooler_output is the
        # projected embedding; 4.x returns the tensor directly
        e = (out.pooler_output if hasattr(out, "pooler_output") else out)[0]
        return (e / e.norm()).cpu().numpy()

    def __call__(self, text):
        if text not in self.cache:
            self.cache[text] = float(1.0 - self._embed(text) @ self.target)
            self.n_evals += 1
        return self.cache[text]


def save_image_grid(images, path, cols=8, thumb=256):
    from PIL import Image

    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for k, im in enumerate(images):
        grid.paste(im.resize((thumb, thumb)),
                   ((k % cols) * thumb, (k // cols) * thumb))
    grid.save(path)


class TurboImageMMDLoss:
    """L(prompt) = biased MMD^2 in CLIP image space between n_cond SDXL-Turbo
    generations of the prompt (common fixed latents -> deterministic, memoized)
    and a target set generated once from the ground-truth prompt."""

    def __init__(self, target_text, device, n_cond=8, n_target=64,
                 turbo_name="stabilityai/sdxl-turbo",
                 clip_name="openai/clip-vit-large-patch14", task_seed=0,
                 save_dir=None, n_steps=1, cfg=0.0):
        self.n_steps = n_steps
        self.cfg = cfg
        # StableDiffusionXLPipeline directly, NOT AutoPipelineForText2Image:
        # auto_pipeline imports HunyuanDiT, which crashes on transformers 5.x
        # (removed MT5Tokenizer) with diffusers 0.36
        from diffusers import StableDiffusionXLPipeline
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            turbo_name, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
        self.clip_proc = CLIPImageProcessor.from_pretrained(clip_name)
        self.clip_tok = CLIPTokenizer.from_pretrained(clip_name)
        self.device = device
        self.n_cond = n_cond

        # target_text: one prompt, or a list of prompts pooled with equal
        # weights (mixture target: G with no single generating prompt)
        target_texts = ([target_text] if isinstance(target_text, str)
                        else list(target_text))
        lat_shape = (4, 64, 64)  # 512x512
        g = torch.Generator("cpu").manual_seed(task_seed + 888)
        self.latents_cond = torch.randn((n_cond, *lat_shape), generator=g,
                                        dtype=torch.float16)
        g = torch.Generator("cpu").manual_seed(task_seed + 777)
        n_per = n_target // len(target_texts)
        latents_tgt = torch.randn((len(target_texts) * n_per, *lat_shape),
                                  generator=g, dtype=torch.float16)
        target_images = []
        for k, txt in enumerate(target_texts):
            target_images.extend(
                self._generate(txt, latents_tgt[k * n_per:(k + 1) * n_per])
            )
        n_target = len(target_images)
        if save_dir:
            save_image_grid(target_images,
                            os.path.join(save_dir, "target_distribution.png"))
        S_G = self._embed_images(target_images)
        self.S_G = S_G
        d2 = (
            np.sum(S_G**2, 1)[:, None] + np.sum(S_G**2, 1)[None, :]
            - 2.0 * S_G @ S_G.T
        )
        self.gamma = 1.0 / np.median(d2[np.triu_indices(n_target, k=1)])
        self._kyy_mean = float(np.exp(-self.gamma * d2).mean())
        self.cache = {}
        self.n_evals = 0

    @torch.no_grad()
    def _generate(self, text, latents, batch=8):
        imgs = []
        for i in range(0, len(latents), batch):
            l = latents[i : i + batch].to(self.device)
            out = self.pipe(
                prompt=[text] * len(l), latents=l,
                num_inference_steps=self.n_steps,
                guidance_scale=self.cfg, height=512, width=512,
            ).images
            imgs.extend(out)
        return imgs

    @torch.no_grad()
    def _embed_images(self, images):
        px = self.clip_proc(images=images, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        out = self.clip.get_image_features(pixel_values=px)
        e = out.pooler_output if hasattr(out, "pooler_output") else out
        e = e / e.norm(dim=-1, keepdim=True)
        return e.float().cpu().numpy()

    def __call__(self, text):
        if text not in self.cache:
            E = self._embed_images(self._generate(text, self.latents_cond))
            self.cache[text] = mmd2_biased(E, self.S_G, self.gamma,
                                           self._kyy_mean)
            self.n_evals += 1
        return self.cache[text]

    @torch.no_grad()
    def _zero_shot(self, embs, prompts):
        """P(prompts[0]) per image via CLIP zero-shot over `prompts`."""
        enc = self.clip_tok(prompts, return_tensors="pt", padding=True).to(
            self.device)
        out = self.clip.get_text_features(**enc)
        te = out.pooler_output if hasattr(out, "pooler_output") else out
        te = (te / te.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        logits = 100.0 * (embs @ te.T)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        return float(p[:, 0].mean())

    def demographics(self, embs):
        """Mean P(man) and P(old) over a set of CLIP image embeddings."""
        return {
            "p_male": self._zero_shot(
                embs, ["a photo of a man", "a photo of a woman"]),
            "p_old": self._zero_shot(
                embs, ["a photo of an old person",
                       "a photo of a young person"]),
        }

    def eval_demographics(self, text, n=64, seed=555):
        """Generate n fresh images for `text` and report demographics."""
        g = torch.Generator("cpu").manual_seed(seed)
        lat = torch.randn((n, 4, 64, 64), generator=g, dtype=torch.float16)
        return self.demographics(self._embed_images(self._generate(text, lat)))


# ---------------------------------------------------------------- grad tilt
class GradTilt:
    """First-order (D-CBG / Gibbs-with-Gradients style) proposal tilt.

    One batched CLIP-text forward+backward prices ALL allowed candidate
    tokens at every canvas slot: represent each slot as a one-hot q_i over
    the allowed vocab, embed as q_i @ E (E = CLIP token embeddings of the
    allowed words, injected via a forward hook on the token-embedding layer),
    and differentiate the surrogate loss
        L_tilt(x) = 1 - cos( CLIP_text(x), target_vec )
    w.r.t. q. The gradient row g[n, i, :] scores every candidate fill at
    slot i of particle n. Used ONLY to shape proposals; the FK weights carry
    the exact correction log p_theta(v) - log q(v), so any surrogate bias is
    corrected and the sampler stays valid."""

    def __init__(self, clip_model, clip_tok, prior, target_vec, device):
        self.clip = clip_model
        self.tok = clip_tok
        self.prior = prior
        self.device = device
        self.target = torch.tensor(target_vec, device=device,
                                   dtype=torch.float32)
        emb_layer = self.clip.text_model.embeddings.token_embedding
        self.E = emb_layer.weight[
            torch.tensor(prior.clip_ids, device=device)
        ].detach()  # (A, d)
        self.clip.requires_grad_(False)  # grads only w.r.t. q, not weights
        self.slot_off = 1 + len(
            clip_tok(prior.prefix_text, add_special_tokens=False)["input_ids"]
        )  # BOS + prefix tokens

    def scores(self, X_dec):
        """X_dec: (N, canvas_len) fully decoded allowed-vocab indices.
        Returns dL_tilt/dq of shape (N, canvas_len, A)."""
        N, S = X_dec.shape
        texts = [self.prior.decode(x) for x in X_dec]
        enc = self.tok(texts, return_tensors="pt", padding="max_length",
                       truncation=True).to(self.device)
        q = torch.zeros((N, S, self.E.shape[0]), device=self.device,
                        requires_grad=True)
        with torch.no_grad():
            onehot = torch.zeros_like(q)
            onehot[torch.arange(N)[:, None], torch.arange(S)[None, :],
                   torch.tensor(X_dec, device=self.device)] = 1.0
        soft = (q + onehot) @ self.E  # (N, S, d); grad flows through q

        emb_layer = self.clip.text_model.embeddings.token_embedding

        def hook(_mod, _inp, out):
            out = out.clone()
            out[:, self.slot_off:self.slot_off + S, :] = soft.to(out.dtype)
            return out

        h = emb_layer.register_forward_hook(hook)
        try:
            out = self.clip.get_text_features(**enc)
            e = out.pooler_output if hasattr(out, "pooler_output") else out
            e = e / e.norm(dim=-1, keepdim=True)
            loss = (1.0 - e @ self.target).sum()
            loss.backward()
        finally:
            h.remove()
        return q.grad.detach().cpu().numpy()

    @torch.no_grad()
    def exact_losses(self, x_dec, slot, cand):
        """Exact surrogate loss for each candidate fill: L_tilt of the full
        prompt with x_dec[slot] replaced by each v in cand. One batched CLIP
        text forward — no Taylor approximation."""
        texts = []
        for v in cand:
            x = x_dec.copy()
            x[slot] = int(v)
            texts.append(self.prior.decode(x))
        enc = self.tok(texts, return_tensors="pt", padding="max_length",
                       truncation=True).to(self.device)
        out = self.clip.get_text_features(**enc)
        e = out.pooler_output if hasattr(out, "pooler_output") else out
        e = e / e.norm(dim=-1, keepdim=True)
        return (1.0 - e @ self.target).float().cpu().numpy()


# ---------------------------------------------------------------- SMC core
def sample_token(marg_row, rng, top_k=0):
    """Sample a token from a marginal, optionally truncated to its top-k
    (renormalized). Truncation modifies the denoiser's sampling distribution
    (standard top-k decoding), not the FK weighting."""
    if top_k and top_k < len(marg_row):
        idx = np.argpartition(marg_row, -top_k)[-top_k:]
        p = marg_row[idx]
        return int(idx[rng.choice(top_k, p=p / p.sum())])
    return int(rng.choice(len(marg_row), p=marg_row))


def estimate_twist(prior, loss_fn, x, marg, estimator, n_dec, beta, rng,
                   top_k=0):
    """(log h(x), L_hat, decode) given precomputed marginals for x."""
    masked = np.where(x == MASK)[0]
    if len(masked) == 0:
        l = loss_fn(prior.decode(x))
        return -beta * l, l, prior.decode(x)
    if estimator == "mode":
        x0 = x.copy()
        x0[masked] = marg[masked].argmax(axis=-1)
        text = prior.decode(x0)
        l = loss_fn(text)
        return -beta * l, l, text
    elif estimator == "sampled":
        losses = np.empty(n_dec)
        first = None
        for j in range(n_dec):
            x0 = x.copy()
            for i in masked:
                x0[i] = sample_token(marg[i], rng, top_k)
            text = prior.decode(x0)
            losses[j] = loss_fn(text)
            if j == 0:
                first = text
        return logsumexp(-beta * losses) - np.log(n_dec), float(losses.mean()), first
    raise ValueError(estimator)


def systematic_resample(W, rng):
    N = len(W)
    positions = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(W), positions), N - 1)


def run_smc(prior, loss_fn, estimator, N, T, beta, n_dec, seed,
            ess_frac=0.5, beta_anneal=False, top_k=0, log_path=None,
            verbose=True, x_init=None, t0=None, corrupt_frac=0.0,
            grad_tilt=None, tilt_scale=1.0, remask_sigma=0.0):
    """x_init/t0: SDEdit-style prompt editing. Instead of starting fully
    masked at t=T, each particle starts from the clean source canvas x_init
    with every position independently masked w.p. t0/T — an exact sample of
    the forward (absorbing) process at time t0 — and the reverse chain runs
    from t0. Initial FK weights are the particles' twist values h_{t0}(x)
    (particles now differ at init; for the identical fully-masked start this
    reduces to uniform weights, so from-scratch behavior is unchanged)."""
    rng_prop = np.random.default_rng(10_000 + seed)
    rng_est = np.random.default_rng(20_000 + seed)
    rng_res = np.random.default_rng(30_000 + seed)

    def beta_at(s):
        return beta * (T - s) / T if beta_anneal else beta

    Lp = prior.canvas_len
    corrupted_src = None
    if x_init is not None:
        assert t0 is not None and 1 <= t0 <= T
        x_src = np.asarray(x_init, dtype=int).copy()
        if corrupt_frac > 0:
            # "truly noisy" source: substitute random tokens BEFORE re-masking.
            # Substitution is NOT part of the absorbing forward process, so a
            # corrupted token that is not re-masked can never be revised —
            # this probes that structural limitation. One corrupted sentence
            # per run (dedicated rng), shared by all particles.
            rng_cor = np.random.default_rng(40_000 + seed)
            for i in range(Lp):
                if rng_cor.random() < corrupt_frac:
                    v = int(rng_cor.integers(prior.A - 1))
                    x_src[i] = v if v < x_src[i] else v + 1  # skip original
            corrupted_src = x_src.copy()
        particles = np.tile(x_src, (N, 1))
        remask = rng_prop.random((N, Lp)) < (t0 / T)
        particles[remask] = MASK
    else:
        t0 = T
        particles = np.full((N, Lp), MASK, dtype=int)

    marg_init = prior.batch_marginals(particles)
    logh_prev = np.empty(N)
    for n in range(N):
        logh_prev[n] = estimate_twist(prior, loss_fn, particles[n],
                                      marg_init[n], estimator, n_dec,
                                      beta_at(t0), rng_est, top_k)[0]
    logw = logh_prev.copy()

    records = []
    n_resamples = 0
    t_start = time.perf_counter()
    for t in range(t0, 1 - 1, -1):
        # reverse step: unmask each masked position w.p. 1/t
        marg = prior.batch_marginals(particles)
        tilt_scores = None
        if grad_tilt is not None:
            X_dec = particles.copy()
            for n in range(N):
                mi = np.where(particles[n] == MASK)[0]
                if len(mi):
                    X_dec[n, mi] = marg[n, mi].argmax(axis=-1)
            tilt_scores = grad_tilt.scores(X_dec)  # (N, S, A) = dL_tilt/dq
        logq_corr = np.zeros(N)
        for n in range(N):
            masked = np.where(particles[n] == MASK)[0]
            p_un = 1.0 if t <= 1 else 1.0 / t
            for i in masked[rng_prop.random(len(masked)) < p_un]:
                if tilt_scores is None:
                    particles[n, i] = sample_token(marg[n, i], rng_prop,
                                                   top_k)
                else:
                    # gradient shortlists candidates; the surrogate loss is
                    # then evaluated EXACTLY on the pool (no Taylor error).
                    # proposal q(v) ∝ p_theta(v) exp(-scale·beta_t·L_tilt(v));
                    # FK weight keeps the exact log p_theta(v) - log q(v)
                    kk = top_k if top_k else 50
                    pool = np.union1d(
                        np.argpartition(marg[n, i], -kk)[-kk:],
                        np.argpartition(-tilt_scores[n, i], -kk)[-kk:],
                    )
                    l_ex = grad_tilt.exact_losses(X_dec[n], i, pool)
                    logits = (np.log(marg[n, i, pool] + 1e-12)
                              - tilt_scale * beta_at(t - 1) * l_ex)
                    p = np.exp(logits - logits.max())
                    p /= p.sum()
                    j = int(rng_prop.choice(len(pool), p=p))
                    v = int(pool[j])
                    particles[n, i] = v
                    logq_corr[n] += (np.log(marg[n, i, v] + 1e-12)
                                     - np.log(p[j]))
        # remasking corrector (ReMDM-style): let the chain REVISE unmasked
        # tokens — each unmasked position re-masks w.p. sigma_t, decaying to
        # 0 so the chain still terminates fully decoded. The corrector is
        # part of the proposal kernel, so the h-ratio weights stay valid.
        if remask_sigma > 0 and t > 1:
            sig = remask_sigma * (t - 1) / T
            for n in range(N):
                unmasked = np.where(particles[n] != MASK)[0]
                for i in unmasked[rng_prop.random(len(unmasked)) < sig]:
                    particles[n, i] = MASK
        # twist at the new state x_{t-1}
        b_s = beta_at(t - 1)
        marg_new = prior.batch_marginals(particles)
        logh_new = np.empty(N)
        l_hat = np.empty(N)
        p0_text = None
        for n in range(N):
            lh, l, dec = estimate_twist(prior, loss_fn, particles[n],
                                        marg_new[n], estimator, n_dec, b_s,
                                        rng_est, top_k)
            logh_new[n] = lh
            l_hat[n] = l
            if n == 0:
                p0_text = dec
        logw += logh_new - logh_prev + logq_corr
        logh_prev = logh_new
        W = np.exp(logw - logsumexp(logw))
        ess = 1.0 / np.sum(W**2)
        resampled = False
        if ess < ess_frac * N:
            idx = systematic_resample(W, rng_res)
            particles = particles[idx]
            logh_prev = logh_prev[idx]
            logw = np.zeros(N)
            n_resamples += 1
            resampled = True
        rec = {
            "t": t - 1, "ess": float(ess), "w_var": float(np.var(W)),
            "L_min": float(l_hat.min()), "L_max": float(l_hat.max()),
            "resampled": resampled, "p0_x0_decode": p0_text,
        }
        records.append(rec)
        if verbose:
            print(f"  t={t-1:3d} ESS={ess:7.1f} L=[{rec['L_min']:.4f},"
                  f"{rec['L_max']:.4f}] resample={resampled} "
                  f"p0='{p0_text}'", flush=True)

    wall = time.perf_counter() - t_start
    final_losses = np.array(
        [loss_fn(prior.decode(particles[n])) for n in range(N)]
    )
    best = int(np.argmin(final_losses))
    result = {
        "estimator": estimator, "seed": seed, "beta": beta,
        "beta_anneal": beta_anneal, "t0": t0,
        "x_init": None if x_init is None else list(map(int, x_init)),
        "corrupt_frac": corrupt_frac,
        "corrupted_source_decode": (
            None if corrupted_src is None else prior.decode(corrupted_src)
        ),
        "L_corrupted_source": (
            None if corrupted_src is None
            else float(loss_fn(prior.decode(corrupted_src)))
        ),
        "n_dec": n_dec if estimator == "sampled" else 1,
        "x_best": particles[best].tolist(),
        "x_best_decode": prior.decode(particles[best]),
        "L_best": float(final_losses[best]),
        "final_L_mean": float(final_losses.mean()),
        "n_resamples": n_resamples, "wall_clock_s": wall,
        "loss_evals_cum": getattr(loss_fn, "n_evals", None),
        "ess_traj": [r["ess"] for r in records],
    }
    if log_path:
        with open(log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return result, records


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["text", "image"], default="text")
    ap.add_argument("--estimators", nargs="+", default=["mode", "sampled"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_particles", type=int, default=16)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--beta", type=float, default=60.0)
    ap.add_argument("--beta_anneal", action="store_true")
    ap.add_argument("--n_dec", type=int, default=4)
    ap.add_argument("--ess_frac", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--lm", default="bert-base-uncased")
    ap.add_argument("--prior", choices=["bert", "llada"], default="bert",
                    help="llada = LLaDA-8B-Base trained masked-diffusion LM")
    ap.add_argument("--prefix", default="a photo of")
    ap.add_argument("--target_words", nargs="+",
                    default=["an", "old", "man", "with", "long", "white", "beard"])
    ap.add_argument("--target_prompts", nargs="+", default=None,
                    help="free-text target prompt(s), decoupled from the "
                         "canvas; several prompts = equal-weight mixture G "
                         "with no single generating prompt (image loss only; "
                         "requires --source_words)")
    ap.add_argument("--source_words", nargs="+", default=None,
                    help="SDEdit-style: start particles from this source "
                         "canvas (same length as target_words) with "
                         "remask_frac of positions re-masked")
    ap.add_argument("--remask_frac", type=float, default=0.5)
    ap.add_argument("--corrupt_frac", type=float, default=0.0,
                    help="before re-masking, substitute each source token "
                         "with a random one w.p. this fraction")
    ap.add_argument("--grad_tilt", action="store_true",
                    help="gradient-tilted proposals: one CLIP-text backprop "
                         "per step scores all candidate fills; FK weights "
                         "carry the exact proposal correction")
    ap.add_argument("--tilt_scale", type=float, default=1.0)
    ap.add_argument("--remask_sigma", type=float, default=0.0,
                    help="remasking-corrector strength: unmasked positions "
                         "re-mask w.p. sigma*(t-1)/T each step, letting the "
                         "chain revise early commitments")
    ap.add_argument("--n_cond", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=64)
    ap.add_argument("--sampler", choices=["turbo", "base"], default="turbo",
                    help="base = non-distilled SDXL (samples ambiguity "
                         "instead of collapsing it; ~12x slower forward, "
                         "no memory penalty since SMC never backprops f_phi)")
    ap.add_argument("--sampler_steps", type=int, default=None)
    ap.add_argument("--sampler_cfg", type=float, default=None)
    ap.add_argument("--outdir", default="output/discrete_x_real")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)

    clip_tok = None
    if args.grad_tilt:
        from transformers import CLIPTokenizer

        clip_tok = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
    if args.target_prompts:
        assert args.loss == "image" and args.source_words, \
            "--target_prompts requires --loss image and --source_words"
        canvas_len = len(args.source_words)
    else:
        canvas_len = len(args.target_words)
    if args.prior == "llada":
        prior = LLaDAPrior(device, args.prefix, canvas_len=canvas_len,
                           clip_tok=clip_tok)
    else:
        prior = BertMaskedPrior(args.lm, device, args.prefix,
                                canvas_len=canvas_len, clip_tok=clip_tok)
    print(f"allowed vocab size: {prior.A}", flush=True)

    if args.sampler == "base":
        sampler_kw = dict(
            turbo_name="stabilityai/stable-diffusion-xl-base-1.0",
            n_steps=args.sampler_steps or 12,
            cfg=args.sampler_cfg if args.sampler_cfg is not None else 3.0)
    else:
        sampler_kw = dict(n_steps=args.sampler_steps or 1,
                          cfg=args.sampler_cfg or 0.0)

    if args.target_prompts:
        x_true = None
        target_text = " | ".join(args.target_prompts)
        print(f"target prompts (mixture G): {args.target_prompts}",
              flush=True)
        loss_fn = TurboImageMMDLoss(args.target_prompts, device,
                                    n_cond=args.n_cond,
                                    n_target=args.n_target,
                                    save_dir=args.outdir, **sampler_kw)
        for p in args.target_prompts:
            print(f"  L('{p}') = {loss_fn(p):.5f}", flush=True)
    else:
        x_true = prior.encode_canvas(args.target_words)
        target_text = prior.decode(x_true)
        print(f"ground-truth prompt: '{target_text}'", flush=True)
        if args.loss == "text":
            loss_fn = ClipTextCosineLoss(target_text, device)
        else:
            loss_fn = TurboImageMMDLoss(target_text, device,
                                        n_cond=args.n_cond,
                                        n_target=args.n_target,
                                        save_dir=args.outdir, **sampler_kw)
        print(f"L(x_true) = {loss_fn(target_text):.5f}", flush=True)

    target_demo = None
    if args.loss == "image":
        target_demo = loss_fn.demographics(loss_fn.S_G)
        print(f"target demographics: {target_demo}", flush=True)

    grad_tilt = None
    if args.grad_tilt:
        if args.loss == "text":
            clip_model, target_vec = loss_fn.model, loss_fn.target
        else:
            clip_model = loss_fn.clip
            m = loss_fn.S_G.mean(axis=0)
            target_vec = m / np.linalg.norm(m)
        grad_tilt = GradTilt(clip_model, clip_tok, prior, target_vec, device)

    x_init, t0 = None, None
    if args.source_words is not None:
        assert len(args.source_words) == prior.canvas_len, \
            "source canvas must match canvas length"
        x_init = prior.encode_canvas(args.source_words)
        t0 = max(1, round(args.remask_frac * args.T))
        source_text = prior.decode(x_init)
        print(f"source prompt: '{source_text}' "
              f"(L={loss_fn(source_text):.5f}, remask_frac="
              f"{args.remask_frac} -> t0={t0}/{args.T})", flush=True)

    all_results = []
    for estimator in args.estimators:
        for seed in args.seeds:
            print(f"\n--- {estimator} seed {seed} ---", flush=True)
            log_path = os.path.join(args.outdir,
                                    f"log_{estimator}_seed{seed}.jsonl")
            res, _ = run_smc(prior, loss_fn, estimator,
                             N=args.n_particles, T=args.T, beta=args.beta,
                             n_dec=args.n_dec, seed=seed,
                             ess_frac=args.ess_frac,
                             beta_anneal=args.beta_anneal, top_k=args.top_k,
                             log_path=log_path, x_init=x_init, t0=t0,
                             corrupt_frac=args.corrupt_frac,
                             grad_tilt=grad_tilt,
                             tilt_scale=args.tilt_scale,
                             remask_sigma=args.remask_sigma)
            if res.get("corrupted_source_decode"):
                print(f"  corrupted source: "
                      f"'{res['corrupted_source_decode']}' "
                      f"(L={res['L_corrupted_source']:.5f})", flush=True)
            res["exact_recovery"] = (
                x_true is not None and res["x_best"] == x_true.tolist()
            )
            if args.loss == "image":
                res["demographics_best"] = loss_fn.eval_demographics(
                    res["x_best_decode"])
                print(f"  demographics(best): {res['demographics_best']} "
                      f"vs target {target_demo}", flush=True)
            all_results.append(res)
            if args.loss == "image":
                save_image_grid(
                    loss_fn._generate(res["x_best_decode"],
                                      loss_fn.latents_cond),
                    os.path.join(args.outdir,
                                 f"best_{estimator}_seed{seed}.png"),
                )
            print(f"[{estimator} seed {seed}] L_best={res['L_best']:.5f} "
                  f"exact={res['exact_recovery']} "
                  f"resamples={res['n_resamples']} "
                  f"time={res['wall_clock_s']:.1f}s\n"
                  f"  -> '{res['x_best_decode']}'", flush=True)

    print("\n=== Summary ===")
    for est in args.estimators:
        rs = [r for r in all_results if r["estimator"] == est]
        print(f"{est:8s} L_best={np.mean([r['L_best'] for r in rs]):.5f}"
              f"±{np.std([r['L_best'] for r in rs]):.5f} "
              f"exact={np.mean([r['exact_recovery'] for r in rs]):.0%} "
              f"time={np.mean([r['wall_clock_s'] for r in rs]):.1f}s")

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "target_text": target_text,
                   "L_x_true": (None if x_true is None
                                else loss_fn(target_text)),
                   "target_demographics": target_demo,
                   "runs": all_results},
                  f, indent=2)
    print(f"\nWrote {args.outdir}/metrics.json")


if __name__ == "__main__":
    main()
