import heapq
import csv
import copy 

class Node:
    def __init__(self, x, y, value, food_collected=set(), g=0, heuristic=0, parent=None) -> None:
        self.x = x
        self.y = y
        self.value = value
        self.food_collected = food_collected  # Food collected by agent
        self.g = g
        self.heuristic = heuristic  
        self.f = self.g + self.heuristic 
        self.parent = parent  

    def __hash__(self) -> int:
        return hash((self.x, self.y, tuple(sorted(self.food_collected))))

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.x, self.y, sorted(self.food_collected)) == (other.x, other.y, sorted(other.food_collected))

    def __str__(self) -> str:
        return f"({self.x}, {self.y}), {self.value}"

def csv_format(filename):
    grid = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            grid.append(row)
    return grid

def convert_to_node_list(grid_path):
    grid = csv_format(grid_path)
    nodes_list = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[i])):
            node = Node(j, i, grid[i][j])
            row.append(node)
        nodes_list.append(row)
    return nodes_list

def heuristic(curr, goal):
    return abs(curr.x - goal.x) + abs(curr.y - goal.y)

def isValidMove(node, direction, grid):
    max_rows = len(grid)
    max_columns = len(grid[0])

    if direction == "UP":
        return node.y - 1 >= 0
    elif direction == "DOWN":
        return node.y + 1 < max_rows
    elif direction == "LEFT":
        return node.x - 1 >= 0
    elif direction == "RIGHT":
        return node.x + 1 < max_columns
    else:
        return False

def getStart(grid):
    for row in grid:
        for node in row:
            if node.value == 'S':
                return node
    return None

def getGoal(grid):
    for row in grid:
        for node in row:
            if node.value == 'H':
                return node
    return None

def getFrogs(grid):
    frogs = set()
    for row in grid:
        for node in row:
            if node.value == 'F':
                frogs.add((node.x, node.y))
    return frogs

def getLegalMoves(node, grid):
    directions = []
    if isValidMove(node, "UP", grid):
        directions.append("UP")
    if isValidMove(node, "DOWN", grid):
        directions.append("DOWN")
    if isValidMove(node, "LEFT", grid):
        directions.append("LEFT")
    if isValidMove(node, "RIGHT", grid):
        directions.append("RIGHT")
    return directions

def reconstruct_path(node):
    cost = node.g
    path = []
    while node is not None:
        path.append((node.y, node.x))
        node = node.parent
    return path[::-1], cost

def pathfinding(grid_path):
    frontier = []
    explored = {}
    grid = convert_to_node_list(grid_path)
    start_node = getStart(grid)
    goal_node = getGoal(grid)
    all_frogs = getFrogs(grid)

    start_node.g = 0
    start_node.heuristic = heuristic(start_node, goal_node)
    start_node.f = start_node.g + start_node.heuristic
    heapq.heappush(frontier, (start_node.f, start_node))

    while frontier:
        _, curr = heapq.heappop(frontier)
        explored[curr] = curr

        if curr.x == goal_node.x and curr.y == goal_node.y and curr.food_collected == all_frogs:
            return reconstruct_path(curr)

        directions = getLegalMoves(curr, grid)
        for direction in directions:
            new_x, new_y = curr.x, curr.y
            if direction == "UP":
                new_y -= 1
            elif direction == "DOWN":
                new_y += 1
            elif direction == "LEFT":
                new_x -= 1
            elif direction == "RIGHT":
                new_x += 1

            node = grid[new_y][new_x]
            new_food_collected = copy.deepcopy(curr.food_collected)

            if node.value == 'F':
                new_food_collected.add((new_x, new_y))

            new_node = Node(new_x, new_y, node.value, new_food_collected, curr.g + 1, parent=curr)
            new_node.heuristic = heuristic(new_node, goal_node)
            new_node.f = new_node.g + new_node.heuristic

            if new_node not in explored and new_node not in [n for _, n in frontier]:
                heapq.heappush(frontier, (new_node.f, new_node))
            elif new_node in explored and new_node.f < explored[new_node].f:
                explored[new_node] = new_node
    return None

def main():
    path = pathfinding(r"c:/Users/haroo/Desktop/AIFINAL/test.csv")
    print("Path found:")
    print(path)

main()
