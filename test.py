import heapq
import csv
import random


# CSV parsing function
def csv_format(filename):
    grid = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            grid.append(row)
    return grid


# Define Node class
class Node:
    def __init__(self, x, y, value, g=0, heuristic=0, parent=None):
        self.x = x
        self.y = y
        self.value = value
        self.g = g
        self.heuristic = heuristic
        self.f = self.g + self.heuristic
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f


# Heuristic function: Manhattan distance
def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


# Get legal moves: considering obstacles and grid bounds
def get_legal_moves(node, grid):
    directions = []
    rows, cols = len(grid), len(grid[0])
    if node.y - 1 >= 0 and grid[node.y - 1][node.x] != '#':  # UP
        directions.append((node.x, node.y - 1))
    if node.y + 1 < rows and grid[node.y + 1][node.x] != '#':  # DOWN
        directions.append((node.x, node.y + 1))
    if node.x - 1 >= 0 and grid[node.y][node.x - 1] != '#':  # LEFT
        directions.append((node.x - 1, node.y))
    if node.x + 1 < cols and grid[node.y][node.x + 1] != '#':  # RIGHT
        directions.append((node.x + 1, node.y))
    return directions


# A* pathfinding
def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    start_node = Node(start[0], start[1], grid[start[1]][start[0]])
    goal_node = Node(goal[0], goal[1], grid[goal[1]][goal[0]])

    frontier = []
    heapq.heappush(frontier, start_node)
    explored = set()
    came_from = {}

    while frontier:
        current = heapq.heappop(frontier)

        if (current.x, current.y) in explored:
            continue
        explored.add((current.x, current.y))

        if (current.x, current.y) == (goal_node.x, goal_node.y):
            path = []
            while current:
                path.append((current.x, current.y))
                current = came_from.get((current.x, current.y))
            return path[::-1]

        for move in get_legal_moves(current, grid):
            next_node = Node(move[0], move[1], grid[move[1]][move[0]])
            next_node.g = current.g + 1
            next_node.heuristic = heuristic(next_node, goal_node)
            next_node.f = next_node.g + next_node.heuristic
            if (next_node.x, next_node.y) not in explored:
                came_from[(next_node.x, next_node.y)] = current
                heapq.heappush(frontier, next_node)

    return None


# Update grid with predator and prey positions
def update_grid(grid, prey, predator):
    # Clear previous positions
    for row in grid:
        for i, cell in enumerate(row):
            if cell in ['D', 'L']:
                row[i] = '.'
    # Place prey and predator in their new positions
    grid[prey.y][prey.x] = 'D'
    grid[predator.y][predator.x] = 'L'



# Predator movement using A*
def predator_move(grid, predator, prey):
    path_to_prey = a_star(grid, (predator.x, predator.y), (prey.x, prey.y))
    if path_to_prey and len(path_to_prey) > 1:
        predator.x, predator.y = path_to_prey[1]  # Move one step toward the prey




# Prey movement using A*
def prey_move(grid, prey, food_locations, goal, collected_food):
    if food_locations:
        # Find nearest food
        nearest_food = min(food_locations, key=lambda f: heuristic(prey, Node(f[0], f[1], 'F')))
        path_to_food = a_star(grid, (prey.x, prey.y), nearest_food)
        if path_to_food and len(path_to_food) > 1:
            prey.x, prey.y = path_to_food[1]  # Move to the next step
            if (prey.x, prey.y) in food_locations:
                collected_food.add((prey.x, prey.y))  # Mark food as collected
                food_locations.remove((prey.x, prey.y))  # Remove from food list
    else:
        # Move to goal if all food is collected
        path_to_goal = a_star(grid, (prey.x, prey.y), (goal.x, goal.y))
        if path_to_goal and len(path_to_goal) > 1:
            prey.x, prey.y = path_to_goal[1]  # Move to the next step
            
def predict_prey_path(grid, prey, goal, food_locations):
    if food_locations:
        nearest_food = min(food_locations, key=lambda f: heuristic(prey, Node(f[0], f[1], 'F')))
        return a_star(grid, (prey.x, prey.y), nearest_food)
    else:
        return a_star(grid, (prey.x, prey.y), (goal.x, goal.y))



# Simulate the predator-prey game
def simulate_game(grid_path):
    grid = csv_format(grid_path)

    # Locate prey, predator, food, and goal
    food_locations = set()
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == 'F':
                food_locations.add((x, y))
            elif cell == 'D':
                prey = Node(x, y, 'D')
            elif cell == 'L':
                predator = Node(x, y, 'L')
            elif cell == 'H':
                goal = Node(x, y, 'H')

    collected_food = set()

    while True:
        # Update grid
        update_grid(grid, prey, predator)

        # Display current state of the grid
        for row in grid:
            print(" ".join(row))
        print()

        # Debugging info
        print(f"Prey position: ({prey.x}, {prey.y})")
        print(f"Predator position: ({predator.x}, {predator.y})")
        print(f"Food locations: {food_locations}")

        # Prey moves
        prey_move(grid, prey, food_locations, goal, collected_food)

        # Check if prey reached the goal
        if prey.x == goal.x and prey.y == goal.y:
            print("Prey reached the goal!")
            break

        # Predator moves
        predator_move(grid, predator, prey)

        # Check if predator caught the prey
        if predator.x == prey.x and predator.y == prey.y:
            print("Predator caught the prey!")
            break



# Run the simulation with a sample grid file
grid_path = r"C:\Users\haroo\Desktop\AIFINAL\test.csv"
simulate_game(grid_path)
