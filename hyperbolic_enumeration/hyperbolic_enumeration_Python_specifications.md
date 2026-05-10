# Specifications of the Python implementation of HyperbolicEnumeration

This summary details the specifications for the Python implementation of the isotropic hyperbolic enumeration function.

## Mathematical Specifications

The goal of the enumeration function is to provide a bijective mapping between an integer index $i \in \mathbb{N}$ and a multi-index $\mathbf{\alpha} = (\alpha_0, \dots, \alpha_{d-1}) \in \mathbb{N}^d$.

### 1. The $q$-norm Criterion

The importance of a multi-index is determined by its $q$-norm (or quasi-norm when $0 < q < 1$):


$$\|\mathbf{\alpha}\|_q = \left( \sum_{j=0}^{d-1} \alpha_j^q \right)^{1/q}$$


The parameter $q$ controls the "sparsity" of the basis. Smaller values of $q$ result in a more severe penalization of high-order interactions (cross-terms where multiple $\alpha_j > 0$).

### 2. Ordering Logic

To ensure a unique and deterministic sequence, the indices are sorted according to the following hierarchy:

1. **Primary**: Increasing $q$-norm $\|\mathbf{\alpha}\|_q$.
2. **Secondary**: Increasing total degree $\sum_{j=0}^{d-1} \alpha_j$.
3. **Tertiary**: Graded reverse-lexicographic ordering (prioritizing lower indices for the same norm and degree).

---

## Algorithmic Specifications

The implementation uses a dynamic discovery approach to handle the theoretically infinite space of multi-indices.

### 1. Data Structures

* **Priority Queue (Min-Heap)**: Stores "candidate" multi-indices. It ensures that the index with the lowest $q$-norm is always extracted next.
* **Visited Set**: A hash-based set that stores all discovered multi-indices to prevent redundant processing and infinite loops.
* **Result Cache**: A list storing the ordered sequence of multi-indices to allow $O(1)$ retrieval for previously computed indices.

### 2. Execution Flow

The algorithm follows an iterative expansion process:

1. **Initialization**: Place the origin $(0, \dots, 0)$ into the priority queue and the visited set.
2. **Extraction**: Extract the element with the minimum $q$-norm from the heap.
3. **Expansion**: Generate neighbors by incrementing the degree of each dimension $j \in \{0, \dots, d-1\}$ by one:

$$\mathbf{\alpha}' = (\alpha_0, \dots, \alpha_j + 1, \dots, \alpha_{d-1})$$


4. **Validation**: If a neighbor has not been visited, calculate its $q$-norm and add it to the priority queue and visited set.
5. **Termination**: Repeat until the requested index $i$ is reached.

### 3. Complexity Analysis

| Component | Complexity |
| --- | --- |
| **Extraction** | $O(\log N)$ |
| **Insertion** | $O(\log N)$ |
| **Duplicate Check** | $O(1)$ (average) |
| **Memory** | $O(M \times d)$ |

> [!NOTE]
> $N$ represents the number of candidates in the heap, $M$ the total number of indices generated, and $d$ the dimension.

---

### Comparison of Truncation Schemes

The table below summarizes how the $q$ parameter influences the resulting set of multi-indices.

| Parameter | Type | Geometry | Interaction Penalty |
| --- | --- | --- | --- |
| $q = 1$ | Linear | Simplex (Triangle/Pyramid) | Standard |
| $q < 1$ | Hyperbolic | Concave (Hyperboloid) | High |
| $q \to \infty$ | Infinity Norm | Hypercube | None |
