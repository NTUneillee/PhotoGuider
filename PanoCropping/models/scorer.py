"""Self-contained loader + inference for the 60-feature composition scorer.

The scorer is a hierarchical additive model (a token-grouped Neural Additive Model with a
small set of pairwise interaction terms). It consumes the same 60 features the cropper's
selector uses, groups their per-feature contributions into four concept tokens
(S / C / A / I = subject, composition, aesthetics, integrity), sums them to a scalar composition
score, and maps that score to a calibrated 1-5 mean-opinion score (GAICD MOS scale) via isotonic regression.

Trained on GAICD crop rankings; SRCC = 0.763 on the GAICD test split (reported in the paper). The artifact
(models/composition_scorer_60f.pkl) stores the feature list, per-feature standardisation
(mu/sd), the trained weights, the interaction pairs, and the isotonic calibrator.

Usage:
    from scorer import load, build_scorer, score_60f
    art   = load()                       # dict with feats, mu, sd, state, iso, ...
    model = build_scorer(art)            # torch module on the best available device
    comp, mos = score_60f(model, art, df)  # df must contain all art["feats"] columns
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

SEED = 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOKENS = ["S", "C", "A", "I"]                    # subject / composition / aesthetics / integrity
ARTIFACT = Path(__file__).resolve().parent / "composition_scorer_60f.pkl"


# ─────────────────────────── network blocks ───────────────────────────
class NAMCore(torch.nn.Module):
    """Vectorised per-feature MLP 1->h1->h2->1 (ELU); feature_out -> (B, F)."""

    def __init__(self, n_feat, h1=32, h2=16):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = torch.nn.Parameter(torch.randn(n_feat, h1, generator=g) * 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(n_feat, h1))
        self.W2 = torch.nn.Parameter(torch.randn(n_feat, h1, h2, generator=g) / np.sqrt(h1))
        self.b2 = torch.nn.Parameter(torch.zeros(n_feat, h2))
        self.W3 = torch.nn.Parameter(torch.randn(n_feat, h2, generator=g) / np.sqrt(h2))
        self.b3 = torch.nn.Parameter(torch.zeros(n_feat))

    def feature_out(self, X):                       # X (B,F)
        h = TF.elu(X.unsqueeze(-1) * self.W1 + self.b1)          # (B,F,h1)
        h = TF.elu(torch.einsum("bfi,fio->bfo", h, self.W2) + self.b2)
        return (h * self.W3).sum(-1) + self.b3                    # (B,F)


class PairCore(torch.nn.Module):
    """Vectorised per-pair MLP 2->16->8->1 (ELU); pair_out -> (B, K)."""

    def __init__(self, K, h1=16, h2=8):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = torch.nn.Parameter(torch.randn(K, 2, h1, generator=g) * 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(K, h1))
        self.W2 = torch.nn.Parameter(torch.randn(K, h1, h2, generator=g) / np.sqrt(h1))
        self.b2 = torch.nn.Parameter(torch.zeros(K, h2))
        self.W3 = torch.nn.Parameter(torch.zeros(K, h2))     # zero-init: starts at additive baseline
        self.b3 = torch.nn.Parameter(torch.zeros(K))

    def pair_out(self, Xp2):                                  # Xp2 (B,K,2)
        h = TF.elu(torch.einsum("bki,kio->bko", Xp2, self.W1) + self.b1)
        h = TF.elu(torch.einsum("bki,kio->bko", h, self.W2) + self.b2)
        return (h * self.W3).sum(-1) + self.b3               # (B,K)


class TokenNAMX(torch.nn.Module):
    """Hierarchical additive model + optional pairwise interaction terms.

    score = sum_j w[tok(j)] f_j(x_j) + sum_p 0.5(w[tok(a_p)]+w[tok(b_p)]) g_p(x_a, x_b).
    Each feature's contribution is weighted by its concept token; a pair term's contribution
    is split 50/50 between the two features' tokens.
    """

    def __init__(self, n_feat, tok_of, pairs=None):
        super().__init__()
        self.core = NAMCore(n_feat)
        self.register_buffer("tok_of", torch.as_tensor(tok_of, dtype=torch.long))
        self.w = torch.nn.Parameter(torch.ones(len(TOKENS)))
        self.n_pairs = 0 if pairs is None or len(pairs) == 0 else len(pairs)
        if self.n_pairs:
            self.register_buffer("pair_idx", torch.as_tensor(np.asarray(pairs), dtype=torch.long))
            self.pcore = PairCore(self.n_pairs)

    def prep(self, X):
        return torch.as_tensor(X, dtype=torch.float32, device=DEV)

    def _pair_eff(self, Xp):                                  # (B,K) effective pair contribs
        po = self.pcore.pair_out(Xp[:, self.pair_idx])        # (B,K)
        pw = 0.5 * (self.w[self.tok_of[self.pair_idx[:, 0]]]
                    + self.w[self.tok_of[self.pair_idx[:, 1]]])
        return po * pw

    def feature_out(self, Xp):                                # main effects only (B,F)
        return self.core.feature_out(Xp) * self.w[self.tok_of]

    def forward(self, Xp):
        s = self.feature_out(Xp).sum(1)
        if self.n_pairs:
            s = s + self._pair_eff(Xp).sum(1)
        return s


# ─────────────────────────── public API ───────────────────────────
def load(path=ARTIFACT):
    """Load the scorer artifact dict (feats, mu, sd, tok_of, pairs, state, iso, ...)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_scorer(art):
    """Instantiate the trained scorer on the best available device."""
    model = TokenNAMX(len(art["feats"]), art["tok_of"], art["pairs"])
    model.load_state_dict(art["state"])
    model.to(DEV).eval()
    return model


def scores_in_batches(model, Xp, n, bs=65536):
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            out.append(model(Xp[i:i + bs]).float().cpu().numpy())
    return np.concatenate(out)


def score_60f(model, art, df):
    """Score a dataframe (must contain all art['feats']) -> (raw comp score, calibrated MOS)."""
    Xs = (df[art["feats"]].values.astype(np.float64) - art["mu"]) / art["sd"]
    sc = scores_in_batches(model, model.prep(Xs), len(df))
    mos = np.asarray(art["iso"].predict(sc), np.float64)
    return sc.astype(np.float64), mos
