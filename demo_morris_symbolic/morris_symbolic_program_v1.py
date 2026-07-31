# %%
import openturns as ot
import openturns.testing as ott
import time

# %%
input_vars = ["x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",
              "x10","x11","x12","x13","x14","x15","x16","x17","x18","x19"]

expr = """ 
var x[20] := {x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,
              x10,x11,x12,x13,x14,x15,x16,x17,x18,x19};

var y := b0;

/* --- build w --- */
var w[20];

for (var i := 0; i < 20; i += 1)
{
   w[i] := 2 * (x[i] - 0.5);
};

/* nonlinear indices */
w[2] := 2 * (1.1*x[2]/(x[2]+0.1) - 0.5);
w[4] := 2 * (1.1*x[4]/(x[4]+0.1) - 0.5);
w[6] := 2 * (1.1*x[6]/(x[6]+0.1) - 0.5);

/* --- linear term --- */
for (var i := 0; i < 20; i += 1)
{
   y += b1[i] * w[i];
};

/* --- quadratic term --- */
for (var i := 0; i < 20; i += 1)
{
   for (var j := i + 1; j < 20; j += 1)
   {
      if ((i < 6) and (j < 6))
         y += -15 * w[i] * w[j];
      else
         y += b2[i*20 + j] * w[i] * w[j];
   };
};

/* --- cubic term (first 5 vars) --- */
for (var i := 0; i < 5; i += 1)
{
   for (var j := i + 1; j < 5; j += 1)
   {
      for (var k := j + 1; k < 5; k += 1)
      {
         y += -10 * w[i] * w[j] * w[k];
      };
   };
};

/* --- quartic term (first 4 vars) --- */
for (var i := 0; i < 4; i += 1)
{
   for (var j := i + 1; j < 4; j += 1)
   {
      for (var k := j + 1; k < 4; k += 1)
      {
         for (var l := k + 1; l < 4; l += 1)
         {
            y += 5 * w[i] * w[j] * w[k] * w[l];
         };
      };
   };
};

y
"""

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

g = ot.SymbolicFunction(input_vars, [expr])

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
