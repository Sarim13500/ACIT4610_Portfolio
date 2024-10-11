import random
import numpy as np


def calculate_distance(customer1, customer2):
    """ Calculate Euclidean distance between two customers. """
    return np.sqrt((customer1['xcoord'] - customer2['xcoord']) ** 2 +
                   (customer1['ycoord'] - customer2['ycoord']) ** 2)


class AntColonyOptimizer:
    def __init__(self, customers, vehicle_capacity, iterations, num_ants):
        self.customers = customers
        self.vehicle_capacity = vehicle_capacity
        self.iterations = iterations
        self.num_ants = num_ants
        self.pheromone = np.ones((len(customers), len(customers)))  # Initialize pheromone levels
        self.alpha = 1  # Pheromone importance
        self.beta = 2  # Distance importance
        self.evaporation_rate = 0.5  # Rate at which pheromone evaporates

    def run(self):
        for _ in range(self.iterations):
            all_routes = []
            for _ in range(self.num_ants):
                route = self.construct_route()
                print(f"Constructed Route: {route}")  # Debugging line
                all_routes.append(route)

            self.update_pheromone(all_routes)
            self.evaluate_routes(all_routes)

    def construct_route(self):
        route = [0]  # Start at the depot (customer 0)
        total_demand = 0
        current_time = 0

        while len(route) < len(self.customers) - 1:  # -1 to keep room for returning to depot
            next_customer = self.select_next_customer(route, total_demand, current_time)
            if next_customer is not None:
                route.append(next_customer)
                total_demand += self.customers[next_customer]['demand']
                travel_time = calculate_distance(self.customers[route[-2]], self.customers[next_customer])
                current_time += travel_time + self.customers[next_customer]['service_time']
            else:
                break

        route.append(0)  # Return to depot
        print(f"Final Route: {route}")  # Debugging line
        return route

    def select_next_customer(self, route, total_demand, current_time):
        """ Select the next customer based on pheromone and distance. """
        last_customer = route[-1]
        probabilities = []
        customer_indices = []  # Keep track of valid customer indices
        for i in range(1, len(self.customers)):  # Skip the depot
            if i not in route:  # Avoid visiting already visited customers
                demand = self.customers[i]['demand']
                if total_demand + demand <= self.vehicle_capacity:
                    pheromone_level = self.pheromone[last_customer][i]
                    distance = calculate_distance(self.customers[last_customer], self.customers[i])
                    probabilities.append((pheromone_level ** self.alpha) * ((1 / distance) ** self.beta))
                    customer_indices.append(i)  # Store valid customer index

        if not probabilities:
            return None  # No valid next customer

        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return np.random.choice(customer_indices, p=probabilities)  # Randomly select the next customer

    def update_pheromone(self, all_routes):
        """ Update pheromone levels based on the routes found. """
        for route in all_routes:
            distance = self.calculate_route_distance(route)
            pheromone_contribution = 1 / distance if distance > 0 else 0

            for i in range(len(route) - 1):
                self.pheromone[route[i]][route[i + 1]] += pheromone_contribution

        # Apply evaporation
        self.pheromone *= (1 - self.evaporation_rate)

    def calculate_route_distance(self, route):
        """ Calculate total distance of the given route. """
        if len(route) < 2:
            print(f"Invalid route: {route}")  # Debugging line
            return float('inf')  # No valid route to calculate

        total_distance = 0
        for i in range(len(route) - 1):
            print(f"Calculating distance between {route[i]} and {route[i + 1]}")  # Debugging line
            total_distance += calculate_distance(self.customers[route[i]], self.customers[route[i + 1]])
        return total_distance

    def evaluate_routes(self, all_routes):
        """ Evaluate the routes and find the best one. """
        # Placeholder for evaluating routes if needed
        pass


def main():
    customers = [
        {'xcoord': 40, 'ycoord': 50, 'demand': 0, 'service_time': 0},
        {'xcoord': 45, 'ycoord': 68, 'demand': 10, 'service_time': 90},
        {'xcoord': 45, 'ycoord': 70, 'demand': 30, 'service_time': 90},
        # ... Add other customers here, following the same structure
        {'xcoord': 55, 'ycoord': 85, 'demand': 20, 'service_time': 90},
    ]

    vehicle_capacity = 200
    iterations = 100
    num_ants = 10

    aco = AntColonyOptimizer(customers, vehicle_capacity, iterations, num_ants)
    aco.run()


if __name__ == "__main__":
    main()
