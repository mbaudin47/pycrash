import heapq

class HyperbolicEnumerationHeap:
    def __init__(self, q, dimension):
        self.q = q
        self.dimension = dimension
        self.generated_indices = []
        self.visited = set()
        self.priority_queue = []
        
        origin = (0,) * dimension
        self.visited.add(origin)
        
        # Structure of elements in the priority queue:
        # (q_norm, sum_of_degrees, secondary_sorting_key, multi_index)
        secondary_key = tuple(-v for v in reversed(origin))
        heapq.heappush(self.priority_queue, (0.0, 0, secondary_key, origin))
        
    def _calculate_q_norm(self, multi_index):
        if self.q <= 0:
            raise ValueError("The q parameter must be strictly positive.")
        return sum(v ** self.q for v in multi_index) ** (1.0 / self.q)

    def __call__(self, i):
        # Generates multi-indices on demand until reaching index i
        while len(self.generated_indices) <= i:
            element = heapq.heappop(self.priority_queue)
            current_multi_index = element[3]
            self.generated_indices.append(current_multi_index)
            
            # Iterate through dimensions to generate neighbors
            for d in range(self.dimension):
                neighbor_list = list(current_multi_index)
                neighbor_list[d] += 1
                neighbor = tuple(neighbor_list)
                
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    q_norm = self._calculate_q_norm(neighbor)
                    total_sum = sum(neighbor)
                    # Secondary key: graded reverse lexicographical order
                    secondary_key = tuple(-v for v in reversed(neighbor))
                    
                    heapq.heappush(self.priority_queue, (q_norm, total_sum, secondary_key, neighbor))
                    
        return self.generated_indices[i]