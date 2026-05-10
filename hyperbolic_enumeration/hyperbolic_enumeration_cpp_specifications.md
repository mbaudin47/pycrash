This summary describes the mathematical and algorithmic specifications of the **C++ implementation** within the OpenTURNS framework.

---

## Mathematical Specifications

The C++ implementation focuses on providing a **bijective mapping** between a natural integer $i$ (the linear index) and a multi-index $\mathbf{\alpha} \in \mathbb{N}^d$ (the vector of marginal degrees).

### 1. Truncation Criterion ($q$-norm)

The importance of a multi-index is measured using the isotropic hyperbolic norm:


$$\|\mathbf{\alpha}\|_q = \left( \sum_{j=0}^{d-1} \alpha_j^q \right)^{1/q}$$

* **$q = 1$**: Represents the standard linear truncation (Total Degree).
* **$0 < q < 1$**: Represents the hyperbolic truncation, which favors main effects and low-order interactions.

### 2. Total Ordering Rule

The sequence is strictly ordered. For two indices $\mathbf{\alpha}$ and $\mathbf{\beta}$, $\mathbf{\alpha}$ precedes $\mathbf{\beta}$ if:

1. $\|\mathbf{\alpha}\|_q < \|\mathbf{\beta}\|_q$
2. If $\|\mathbf{\alpha}\|_q = \|\mathbf{\beta}\|_q$, then $\sum \alpha_j < \sum \beta_j$
3. If degrees are also equal, a **graded reverse-lexicographic order** is applied.

---

## Algorithmic Specifications

The C++ implementation (as per the `HyperbolicEnumerateFunction` class) uses a **lazy generation** strategy with a sorted candidate list.

### 1. Core Data Structures

* **`cache_` (`Collection<Indices>`)**: Stores the finalized, ordered list of multi-indices. This allows $O(1)$ access for any previously calculated index.
* **`candidates_` (`std::list<ValueType>`)**: A sorted list containing "discovered" multi-indices that haven't been finalized yet.
* **`strataCumulatedCardinal_` (`Indices`)**: Stores the indices where a "norm leap" occurs, allowing the library to group polynomials into energy levels or "strata."

### 2. Discovery and Finalization Logic

The algorithm operates as follows when the $i$-th index is requested:

1. **Extraction**: The front of the `candidates_` list (the element with the smallest norm) is moved to the `cache_`.
2. **Neighbor Generation**: For the current index $\mathbf{\alpha}$, the algorithm generates $d$ potential neighbors by incrementing each dimension:

$$\mathbf{\alpha}_{+j} = (\alpha_0, \dots, \alpha_j + 1, \dots, \alpha_{d-1})$$


3. **Sorted Insertion**: For each neighbor:
* **Boundary Check**: Ensure the degree doesn't exceed `upperBound_`.
* **Norm Calculation**: Compute the new $q$-norm.
* **Duplicate Check**: Traverse the `candidates_` list at the specific norm location to ensure the index isn't already present.
* **Insertion**: Insert the new index into the list while maintaining the $q$-norm sort order.



### 3. Strata Management

A unique feature of the C++ implementation is the detection of **Strata**. A stratum is a collection of multi-indices with the same $q$-norm.

* Whenever `current.norm > previous.norm`, a new stratum is recorded in `strataCumulatedCardinal_`.
* This facilitates operations like `getStrataCardinal`, which are essential for managing basis truncation levels in complex metamodels.

---

## Complexity and Performance

| Feature | Complexity | Description |
| --- | --- | --- |
| **Retrieval** | $O(1)$ | Immediate if the index is already in the `cache_`. |
| **Insertion** | $O(N)$ | The `std::list` requires a linear search to maintain sort order. |
| **Inverse Mapping** | $O(M)$ | The `inverse()` method performs a linear search through the cache. |
| **Memory** | $O(M \cdot d)$ | Scales with the number of generated indices and the dimension. |

> [!NOTE]
> Unlike the Python implementation which uses a priority queue ($O(\log N)$), this specific C++ implementation uses a sorted `std::list` ($O(N)$). While slightly slower for very large bases, it simplifies the management of the "norm leap" detection and duplicate checking within the same norm value.