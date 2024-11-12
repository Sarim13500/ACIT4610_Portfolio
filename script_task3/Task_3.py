import numpy as np
import random
import matplotlib.pyplot as plt


# Function to load customers from the "c101.txt" file
def load_customers_from_file(filename):
    customers = []
    with open(filename, 'r') as f:
        lines = f.readlines()[7:]  # Skip first 7 lines (headers)

        for line in lines:
            parts = list(map(int, line.split()))  # Convert to integers
            customer_id = parts[0]
            x_coord = parts[1]
            y_coord = parts[2]
            demand = parts[3]
            ready_time = parts[4]
            due_date = parts[5]
            service_time = parts[6] if len(parts) > 6 else 0  # Service time (optional)
            customers.append((customer_id, x_coord, y_coord, demand, ready_time, due_date, service_time))

    return customers


# Load customers from file
customers_data = load_customers_from_file('c101.txt')


# Distance calculation function
def calculate_distance(customer1, customer2):
    return np.sqrt((customer1[1] - customer2[1]) ** 2 + (customer1[2] - customer2[2]) ** 2)


# Plotting function to display customer coordinates on a grid
def plot_customers(customers):
    x_coords = [customer[1] for customer in customers]
    y_coords = [customer[2] for customer in customers]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, color='blue', marker='o')

    # Annotate customers and depot
    for customer in customers:
        if customer[0] == 0:
            plt.text(customer[1], customer[2], f"Depot {customer[0]}", fontsize=12, ha='right', color='red')
        else:
            plt.text(customer[1], customer[2], f"Customer {customer[0]}", fontsize=12, ha='right')

    plt.grid(True)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Customer Locations on Grid")
    plt.show()

# ACO class
class AntColony:
    def __init__(self, customers, n_ants=10, n_iterations=100, pheromone_factor=1, distance_factor=1):
        self.customers = customers
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromone_factor = pheromone_factor
        self.distance_factor = distance_factor
        self.pheromone = np.ones((len(customers), len(customers)))  # Initialize pheromone matrix
        self.best_solution = None
        self.best_cost = float('inf')

    def run(self):
        for _ in range(self.n_iterations):
            for _ in range(self.n_ants):
                solution = self.construct_solution()
                cost = self.calculate_cost(solution)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution
        return self.best_solution, self.best_cost

    def construct_solution(self):
        unvisited = set(range(1, len(self.customers)))  # Exclude depot
        solution = [0]  # Start at depot
        current_customer = 0

        while unvisited:
            next_customer = self.choose_next_customer(current_customer, unvisited)
            solution.append(next_customer)
            current_customer = next_customer
            unvisited.remove(next_customer)

        solution.append(0)  # Return to depot
        return solution

    def choose_next_customer(self, current_customer, unvisited):
        # Calculate probabilities based on distance and pheromone levels
        probabilities = []
        for next_customer in unvisited:
            pheromone_level = self.pheromone[current_customer][next_customer] ** self.pheromone_factor
            distance = calculate_distance(self.customers[current_customer], self.customers[next_customer]) ** (-self.distance_factor)
            probabilities.append(pheromone_level * distance)

        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Choose next customer based on probability
        return random.choices(list(unvisited), probabilities)[0]

    def calculate_cost(self, solution):
        total_cost = 0
        for i in range(len(solution) - 1):
            total_cost += calculate_distance(self.customers[solution[i]], self.customers[solution[i + 1]])
        return total_cost

# PSO class
class ParticleSwarm:
    def __init__(self, customers, n_particles=10, n_iterations=100):
        self.customers = customers
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.best_solution = None
        self.best_cost = float('inf')
        self.particles = [self.construct_solution() for _ in range(n_particles)]
        self.particle_best_solutions = self.particles[:]
        self.particle_best_costs = [self.calculate_cost(sol) for sol in self.particles]

    def run(self):
        for _ in range(self.n_iterations):
            for i in range(self.n_particles):
                solution = self.particles[i]
                cost = self.calculate_cost(solution)

                # Update personal best
                if cost < self.particle_best_costs[i]:
                    self.particle_best_costs[i] = cost
                    self.particle_best_solutions[i] = solution

                # Update global best
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution

                # Move particle
                self.particles[i] = self.move_particle(solution)

        return self.best_solution, self.best_cost

    def construct_solution(self):
        unvisited = set(range(1, len(self.customers)))  # Exclude depot
        solution = [0]  # Start at depot
        current_customer = 0

        while unvisited:
            next_customer = random.choice(list(unvisited))  # Randomly pick a customer
            solution.append(next_customer)
            current_customer = next_customer
            unvisited.remove(next_customer)

        solution.append(0)  # Return to depot
        return solution

    def move_particle(self, solution):
        # Randomly swap two customers to simulate particle movement
        if len(solution) > 3:  # Avoid swapping depot
            i, j = random.sample(range(1, len(solution) - 1), 2)
            solution[i], solution[j] = solution[j], solution[i]
        return solution

    def calculate_cost(self, solution):
        total_cost = 0
        for i in range(len(solution) - 1):
            total_cost += calculate_distance(self.customers[solution[i]], self.customers[solution[i + 1]])
        return total_cost

# Running ACO and PSO
if __name__ == "__main__":
    print("Running Ant Colony Optimization...")
    aco = AntColony(customers_data)
    aco_best_solution, aco_best_cost = aco.run()
    print(f"Best Solution (ACO): {aco_best_solution}")
    print(f"Best Cost (ACO): {aco_best_cost}")

    print("Running Particle Swarm Optimization...")
    pso = ParticleSwarm(customers_data)
    pso_best_solution, pso_best_cost = pso.run()
    print(f"Best Solution (PSO): {pso_best_solution}")
    print(f"Best Cost (PSO): {pso_best_cost}")

    # Plot customer locations
    print("Plotting customer locations...")
    plot_customers(customers_data)
