import math
import numpy as np

STEPS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class Node(object):
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, pos=None):
        self.parent = parent
        self.pos = pos

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos

    
    def __str__(self):
        ret = "({}, {}): {}".format(self.pos[0], self.pos[1], self.f)
        return ret


class AstarPlanner(object):

    def __init__(self, grid=None):
        self.g_ = grid

        if grid is not None:
            self.width_ = grid.shape[1]
            self.height_ = grid.shape[0]

    
    def update_grid(self, grid):
        self.g_ = grid
        self.width_ = grid.shape[1]
        self.height_ = grid.shape[0]
        
    
    def search(self, src, dst, grid=None):
        if grid is not None:
            self.update_grid(grid)
        
        src_node = Node(None, src)
        dst_node = Node(None, dst)

        open_list = {}
        closed_list = {}
    
        open_list[src] = src_node
        while len(open_list) > 0:

            node = self.pop_next_(open_list)
            closed_list[node.pos] = node
            
            if node == dst_node:
                return self.tracepath_(node, closed_list)

            for step in STEPS:
                next_pos = (node.pos[0] + step[0], node.pos[1] + step[1])

                # Make sure within range
                if not self.is_valid_(next_pos[0], next_pos[1]):
                    continue

                # Make sure walkable terrain
                if not self.is_unblocked_(next_pos[0], next_pos[1]):
                    continue

                new_node = Node(node, next_pos)
                new_node.g = node.g + self.cost_(new_node, node)
                new_node.h = self.heuristic_(new_node, dst_node)
                new_node.f = new_node.g + new_node.h

                if self.if_add_node_(new_node, open_list, closed_list):
                    open_list[next_pos] = new_node
        
        # Cannot find a path
        return [], None


    def is_valid_(self, x, y):
        return x < self.width_ and x >=0 and y < self.height_ and y >= 0 

    
    def is_unblocked_(self, x, y):
        return self.g_[y, x] != 1


    def heuristic_(self, a, b):
        return math.sqrt(
            ((a.pos[0] - b.pos[0]) ** 2) 
                + ((a.pos[1] - b.pos[1]) ** 2)
        )
    

    def cost_(self, a, b, density=None):
        if a.pos[0] == b.pos[0] or a.pos[1] == b.pos[1]:
            return 1.0
        else:
            return 1.414

    
    def pop_next_(self, open_list):
        next_node = None

        for pos, node in open_list.items():
            if next_node is None:
                next_node = node
            else:
                if node.f < next_node.f:
                    next_node = node

        return open_list.pop(next_node.pos)


    def tracepath_(self, node, closed_list):
        path = []
        cost = 0.0

        while node is not None:
            path.append(node.pos)
            cost += node.g

            node = node.parent
        
        # Return reversed path
        return path[::-1], cost


    def if_add_node_(self, node, open_list, closed_list):
        # Child is on the closed list
        if node.pos in closed_list:
            return node.f < closed_list[node.pos].f
        
        # Child is already in the open list
        if node.pos in open_list:
            return node.f < open_list[node.pos].f
        
        return True


if __name__ == "__main__":
    # 1: block/wall
    # 0: free way
    grid_map = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1]
    ]
    grid_map = np.array(grid_map)

    planner = AstarPlanner(grid=grid_map)

    path, cost = planner.search((0, 0), (2, 4))
    
    print(cost)
    print(path)