import heapq
import csv

#NOVELTY FOR THIS PROJECT IS MULTI AGENT WHICH USES MINMAX ALOGRITHIM.


#THIS GAME IS IN FAVOUR OF THE PREDATOR
def csv_format(filename):
    grid = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            grid.append(row)
    return grid


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
    
#Manhatten distance
def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)



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

# A* search
def a_star(grid, start, goal):
    start_node = Node(start[0], start[1], grid[start[1]][start[0]])
    goal_node = Node(goal[0], goal[1], grid[goal[1]][goal[0]])

    frontier = []
     # priority queue to store nodes whcih  starts with the start node
    heapq.heappush(frontier, start_node)
    explored = set()
    came_from = {}
    # a set for the already found nodes
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
def update_grid(grid, prey, predator, food_locations):
    # Clear only non-food positions of previous 'D' and 'L'
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == 'D' or cell == 'L':
                grid[y][x] = '.' if (x, y) not in food_locations else 'F'

    # Place prey and predator in their new positions
    if (prey.x, prey.y) in food_locations:
        grid[prey.y][prey.x] = 'F'  # Keep food if prey is on food
    else:
        grid[prey.y][prey.x] = 'D'

    if (predator.x, predator.y) in food_locations:
        grid[predator.y][predator.x] = 'F'  # Keep food if predator is on food
    else:
        grid[predator.y][predator.x] = 'L'

# Minimax algorithm
def minimax(grid, predator, prey, depth, is_maximizer):
    #depth is how many steps in advanced or if the predator catchs the prey
    if depth == 0 or predator.x == prey.x and predator.y == prey.y:
        return -heuristic(predator, prey)  # Predator wants to minimize distance
   #BASICALLY ONLY GOING TO CALL THIS FOR RPEDATOR
    if is_maximizer:
        max_eval = float('-inf')
        # Initialize to negative infinity
        # Explore all legal moves for the predator
        # predator (maximizer): its goal is to reduce the distance to the prey
        for move in get_legal_moves(predator, grid):
            next_predator = Node(move[0], move[1], predator.value)
            # Recursively evaluate the move from the minimizer's perspective
            eval = minimax(grid, next_predator, prey, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        #prey (minimizer): Its goal is to increase the distance from the predator
        min_eval = float('inf')
        for move in get_legal_moves(prey, grid):
            next_prey = Node(move[0], move[1], prey.value)
            eval = minimax(grid, predator, next_prey, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

# Predator movement using Minimax
def predator_move(grid, predator, prey, goal, depth=3):
    best_move = None
    max_eval = float('-inf')

    # Get all possible legal moves for the predator
    legal_moves = get_legal_moves(predator, grid)
    
    # Avoid moving beyond the goal 'H'
    legal_moves = [move for move in legal_moves if not (move[0] == goal.x and move[1] == goal.y)]

    for move in legal_moves:
        next_predator = Node(move[0], move[1], predator.value)
        #false because it is the maxmixer turn
        eval = minimax(grid, next_predator, prey, depth - 1, False)
        if eval > max_eval:
            max_eval = eval
            best_move = move

    if best_move:
        predator.x, predator.y = best_move


# Prey movement using A*
def prey_move(grid, prey, food_locations, goal, collected_food):
    if food_locations:
        #check the nearest food by using the key lambda function
        nearest_food = min(food_locations, key=lambda f: heuristic(prey, Node(f[0], f[1], 'F')))
        path_to_food = a_star(grid, (prey.x, prey.y), nearest_food)
        if path_to_food and len(path_to_food) > 1:
            prey.x, prey.y = path_to_food[1]
            if (prey.x, prey.y) in food_locations:
                collected_food.add((prey.x, prey.y))
                food_locations.remove((prey.x, prey.y))
    else:
        path_to_goal = a_star(grid, (prey.x, prey.y), (goal.x, goal.y))
        if path_to_goal and len(path_to_goal) > 1:
            prey.x, prey.y = path_to_goal[1]


#this function simulates the preys next move and then makes the predator choose wisely.
#THIS IS MEANT FOR PREY_MOVE_WITH_PREDICTION
def predator_move_with_prediction(grid, predator, prey, goal, food_locations, collected_food, depth=3):
    best_move = None
    max_eval = float('-inf')

    # Predict prey's next position using its strategy
    predicted_prey = Node(prey.x, prey.y, prey.value)
    predicted_food_locations = food_locations.copy()
    predicted_collected_food = collected_food.copy()
    
    # Simulate prey's move
    prey_move(grid, predicted_prey, predicted_food_locations, goal, predicted_collected_food)
    
    # Get all possible legal moves for the predator
    legal_moves = get_legal_moves(predator, grid)
    
    # Exclude moves that would place the predator on the home state 'H'
    legal_moves = [move for move in legal_moves if (move[0], move[1]) != (goal.x, goal.y)]

    for move in legal_moves:
        next_predator = Node(move[0], move[1], predator.value)
        
        # Evaluate the position based on the predicted prey position
        eval = minimax(grid, next_predator, predicted_prey, depth - 1, True)
        if eval > max_eval:
            max_eval = eval
            best_move = move

    # Update predator's position to the best move
    if best_move:
        predator.x, predator.y = best_move

#prey adjusts its position based off predators movement
def prey_move_with_prediction(grid, prey, predator, food_locations, goal, collected_food, depth=3):
    # Predict the predator's next move
    predicted_predator = Node(predator.x, predator.y, predator.value)
    predator_move_with_prediction(grid, predicted_predator, prey, goal, food_locations, collected_food, depth)

    # If food is available, prioritize collecting it
    if food_locations:
        nearest_food = min(food_locations, key=lambda f: heuristic(prey, Node(f[0], f[1], 'F')))
        path_to_food = a_star(grid, (prey.x, prey.y), nearest_food, predicted_predator)
        if path_to_food and len(path_to_food) > 1:
            prey.x, prey.y = path_to_food[1]
            if (prey.x, prey.y) in food_locations:
                collected_food.add((prey.x, prey.y))
                food_locations.remove((prey.x, prey.y))
    else:
        # If no food is available, move towards the goal while avoiding the predator
        path_to_goal = a_star(grid, (prey.x, prey.y), (goal.x, goal.y), predicted_predator)
        if path_to_goal and len(path_to_goal) > 1:
            prey.x, prey.y = path_to_goal[1]


def simulate_game(grid_path):
    grid = csv_format(grid_path)

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
    predator_steps = 0
    prey_steps = 0

    while True:
        update_grid(grid, prey, predator, food_locations)

        for row in grid:
            print(" ".join(row))
        print()

        print(f"Prey position: ({prey.x}, {prey.y})")
        print(f"Predator position: ({predator.x}, {predator.y})")
        print(f"Food locations: {food_locations}")

        # Prey move
        prey_move(grid, prey, food_locations, goal, collected_food)
        prey_steps += 1  

        if prey.x == goal.x and prey.y == goal.y:
            print("Prey reached the goal!")
            break

        # Predator move
        predator_move(grid, predator, prey, goal)
        predator_steps += 1  

        if predator.x == prey.x and predator.y == prey.y:
            print("Predator caught the prey!")
            break

    # Final output with step counts
    print(f"Game over! Predator took {predator_steps} steps. Prey took {prey_steps} steps.")


grid_path = r"C:\Users\haroo\Desktop\AIFINAL\test1.csv"
simulate_game(grid_path)
