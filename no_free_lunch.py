from __future__ import annotations
import random
from typing import Callable, List, Tuple

X = int
Y = int
Sample = List[Tuple[X, Y]]
Hypothesis = Callable[[X], Y]
Learner = Callable[[Sample, int], Hypothesis]

# Two simple learners to test against (drop-in any learner you like)
def majority_learner(S: Sample, m: int) -> Hypothesis:
    ones = sum(y for _, y in S)
    majority = 1 if ones > len(S) - ones else 0
    return lambda x: majority

def one_nn_learner(S: Sample, m: int) -> Hypothesis:
    if not S: return lambda x: 0
    S = sorted(S, key=lambda p: p[0])
    xs = [u for u,_ in S]; ys = [v for _,v in S]
    import bisect
    def h(x: X) -> Y:
        i = bisect.bisect_left(xs, x)
        if i==0: return ys[0]
        if i==len(xs): return ys[-1]
        return ys[i-1] if x-xs[i-1] <= xs[i]-x else ys[i]
    return h

# A. Ex-post flipping adversary (pointwise L(h) >= (m-n)/m)
def ex_post_adversary_risk(A: Learner, n: int, m: int, seed=0) -> float:
    rng = random.Random(seed)
    Xdom = list(range(m))
    S = rng.sample(Xdom, n)
    labels = {x: rng.randint(0,1) for x in S}
    S_lab = [(x, labels[x]) for x in S]
    h = A(S_lab, m)
    def f(x: X) -> Y:
        return labels[x] if x in labels else 1 - h(x)
    mistakes = sum(1 for x in Xdom if h(x)!=f(x))
    return mistakes / m  # >= (m-n)/m (certified)

# B. Yao adversary (pre-committed random f) — E[L] = (m-n)/(2m)
def yao_expectation(A: Learner, n: int, m: int, trials=2000, seed=1) -> float:
    rng = random.Random(seed)
    Xdom = list(range(m))
    total = 0.0
    for t in range(trials):
        # pre-commit a random target f
        ftab = {x: rng.randint(0,1) for x in Xdom}
        # draw sample S of size n (without replacement, like uniform-on-X i.i.d. approximated)
        S = rng.sample(Xdom, n)
        S_lab = [(x, ftab[x]) for x in S]
        h = A(S_lab, m)
        mistakes = sum(1 for x in Xdom if h(x)!=ftab[x])
        total += mistakes / m
    return total / trials

if __name__ == "__main__":
    n = 50
    eps = 0.05
    # choose m so that (m-n)/(2m) >= 1/2 - eps  <=>  m >= n/(2eps)
    m = max(n, int(n/(2*eps) + 0.9999))

    for name, A in [("Majority", majority_learner), ("1-NN", one_nn_learner)]:
        # Strong, ex-post adversary (pointwise)
        risk_pointwise = ex_post_adversary_risk(A, n=n, m=m, seed=42)
        # Yao adversary (pre-committed; expectation)
        risk_expected = yao_expectation(A, n=n, m=m, trials=3000, seed=7)
        bound = 0.5 - eps
        print(f"{name:8s} | m={m}, n={n} | ex-post L≈{risk_pointwise:.3f}  | Yao E[L]≈{risk_expected:.3f}  | target ≥ {bound:.3f}")
