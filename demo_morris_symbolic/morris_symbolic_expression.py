# %%
import openturns as ot
import itertools
import openturns.testing as ott
import time

# %%
def build_morris_symbolic(b0, b1_random, b2_random):

    # ---- variables
    input_vars = [f"x{i}" for i in range(20)]

    # ---- build w expressions
    w_expr = []

    for i in range(20):
        if i in [2, 4, 6]:
            w_expr.append(f"(2*(1.1*x{i}/(x{i}+0.1)-0.5))")
        else:
            w_expr.append(f"(2*(x{i}-0.5))")

    # ---- b1
    b1 = [20.0]*10 + list(b1_random)

    # ---- start expression
    expr = [f"{b0}"]

    # ---- linear terms
    for i in range(20):
        expr.append(f"{b1[i]}*{w_expr[i]}")

    # ---- quadratic terms
    idx = 0
    for i in range(20):
        for j in range(i+1, 20):
            if i < 6 and j < 6:
                coeff = -15.0
            else:
                coeff = b2_random[idx]
                idx += 1
            if coeff != 0:
                expr.append(f"{coeff}*{w_expr[i]}*{w_expr[j]}")

    # ---- cubic terms (only first 5 variables)
    for (i,j,k) in itertools.combinations(range(5), 3):
        expr.append(f"-10*{w_expr[i]}*{w_expr[j]}*{w_expr[k]}")

    # ---- quartic terms (only first 4 variables)
    for (i,j,k,l) in itertools.combinations(range(4), 4):
        expr.append(f"5*{w_expr[i]}*{w_expr[j]}*{w_expr[k]}*{w_expr[l]}")

    # ---- final expression
    full_expr = " + ".join(expr)

    return ot.SymbolicFunction(input_vars, [full_expr])

# %%
def linspace(xmin, xmax, npoints):
    """Returns a sample created from a regular grid
    from xmin to xmax with npoints points."""
    step = (xmax - xmin) / (npoints - 1)
    rg = ot.RegularGrid(xmin, step, npoints)
    vertices = rg.getVertices()
    return vertices.asPoint()

# %%
ot.RandomGenerator.SetSeed(1)

b0 = ot.DistFunc.rNormal()
b1 = ot.DistFunc.rNormal(10)
b2 = ot.DistFunc.rNormal(175)

g = build_morris_symbolic(b0, b1, b2)

# %%
# test
x = [0.5]*20
print(g(x))


# %%
# Check accuracy
x = linspace(0.0, 1.0, 20)
y = g(x)
ott.assert_almost_equal(y, [-65.75761172072895])

# %%
# Check speed
X = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * 20)
N = 1000
input_sample = X.getSample(N)
t0 = time.time()
output_sample = g(input_sample)
t1 = time.time()
elapsed_time = t1 - t0
print(f"{N / elapsed_time} eval/s")


# %%
