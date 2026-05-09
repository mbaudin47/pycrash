# %%
import openturns as ot
import openturns.testing as ott
import time


# %%
class MorrisFunction(ot.OpenTURNSPythonFunction):
    """
    Morris test function for sensitivity analysis.

    This function has input dimension 20 and output dimension 1.

    References
    ----------
    - M. D. Morris, 1991, Factorial sampling plans for preliminary
      computational experiments, Technometrics, 33, 161-174.

    Examples
    --------
    >>> import openturns as ot
    >>> ot.RandomGenerator.SetSeed(123)
    >>> b0_random = ot.DistFunc.rNormal()
    >>> b1_random = ot.DistFunc.rNormal(10)
    >>> b2_random = ot.DistFunc.rNormal(175)
    >>> morrisFunction = ot.Function(MorrisFunction(b0_random, b1_random, b2_random))
    >>> dimension = morrisFunction.getInputDimension()
    >>> distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)
    >>> input_sample = distribution.getSample(10)
    >>> output_sample = morrisFunction(input_sample)
    """

    def fmt(x):
        """Format floating point constants for ExprTk."""
        return format(float(x), ".17g")

    def build_morris_exprtk_expression(b0_random, b1_random, b2_random):
        """
        Build an ExprTk program equivalent to MorrisFunction.
        """

        if len(b1_random) != 10:
            raise ValueError(f"b1_random must have length 10, got {len(b1_random)}")

        if len(b2_random) != 175:
            raise ValueError(f"b2_random must have length 175, got {len(b2_random)}")

        b0 = MorrisFunction.fmt(b0_random)

        # b1[0:10] = 20, b1[10:20] = b1_random
        b1 = [20.0] * 10 + list(b1_random)

        b1_expr = ",".join(MorrisFunction.fmt(v) for v in b1)
        b2_expr = ",".join(MorrisFunction.fmt(v) for v in b2_random)

        expr = f"""
    var x[20] := {{
    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,
    x10,x11,x12,x13,x14,x15,x16,x17,x18,x19
    }};

    var b0 := {b0};

    var b1[20] := {{
    {b1_expr}
    }};

    var b2_random[175] := {{
    {b2_expr}
    }};

    var y := b0;

    /* --- build w --- */
    var w[20];

    for (var i := 0; i < 20; i += 1)
    {{
    w[i] := 2 * (x[i] - 0.5);
    }};

    /* nonlinear indices: Python indices 2, 4, 6 */
    w[2] := 2 * (1.1 * x[2] / (x[2] + 0.1) - 0.5);
    w[4] := 2 * (1.1 * x[4] / (x[4] + 0.1) - 0.5);
    w[6] := 2 * (1.1 * x[6] / (x[6] + 0.1) - 0.5);

    /* --- linear term --- */
    for (var i := 0; i < 20; i += 1)
    {{
    y += b1[i] * w[i];
    }};

    /* --- quadratic term --- */
    var random_index := 0;

    for (var i := 0; i < 20; i += 1)
    {{
    for (var j := i + 1; j < 20; j += 1)
    {{
        if ((i < 6) and (j < 6))
        {{
            y += -15.0 * w[i] * w[j];
        }}
        else
        {{
            y += b2_random[random_index] * w[i] * w[j];
            random_index += 1;
        }};
    }};
    }};

    /* --- cubic term: i < j < k, first 5 variables only --- */
    for (var i := 0; i < 5; i += 1)
    {{
    for (var j := i + 1; j < 5; j += 1)
    {{
        for (var k := j + 1; k < 5; k += 1)
        {{
            y += -10.0 * w[i] * w[j] * w[k];
        }};
    }};
    }};

    /* --- quartic term: i < j < k < ell, first 4 variables only --- */
    for (var i := 0; i < 4; i += 1)
    {{
    for (var j := i + 1; j < 4; j += 1)
    {{
        for (var k := j + 1; k < 4; k += 1)
        {{
            for (var ell := k + 1; ell < 4; ell += 1)
            {{
                y += 5.0 * w[i] * w[j] * w[k] * w[ell];
            }};
        }};
    }};
    }};

    y
    """
        return expr

    def __init__(self, b0_random=0.0, b1_random=ot.Point(10), b2_random=ot.Point(175)):
        """
        Create the Morris function.

        Parameters
        ----------
        b0_random : float, optional
            The constant term. Default is 0.0.
        b1_random : ot.Point(10), optional
            Random linear coefficients for dimensions 11-20. Default is zeros.
        b2_random : ot.Point(175), optional
            Random quadratic coefficients. Default is zeros.
        """
        super().__init__(20, 1)
        self.b0_random = b0_random
        self.b1_random = b1_random
        self.b2_random = b2_random
        input_vars = [
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
            "x17",
            "x18",
            "x19",
        ]

        expr = MorrisFunction.build_morris_exprtk_expression(b0_random, b1_random, b2_random)

        self.g = ot.SymbolicFunction(input_vars, [expr])

    def _exec(self, x):
        """Evaluate the Morris function at point x (vectorized)."""
        y = self.g(x)
        return y


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

g = MorrisFunction(b0, b1, b2)

# %%
# test
x = [0.5] * 20
print(g(x))


# %%
# Check accuracy
x = linspace(0.0, 1.0, 20)
y = g(x)
ott.assert_almost_equal(y, [-65.75761172072895])

# %%
# Check speed
X = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * 20)
N = 10000
input_sample = X.getSample(N)
t0 = time.time()
output_sample = g(input_sample)
t1 = time.time()
elapsed_time = t1 - t0
print(f"{N / elapsed_time} eval/s")


# %%
