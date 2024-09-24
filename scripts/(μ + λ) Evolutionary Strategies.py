import numpy as np
import pandas as pd
import os


def mu_plus_lambda_es(mu, lambda_):
    # Define paths
    stocks_path = '../stocks_csv'
    results_path = '../results'

    # Load stock data
    stock_files = [f for f in os.listdir(stocks_path) if f.endswith('.csv')]
    returns_matrix = pd.concat([
        pd.read_csv(os.path.join(stocks_path, file), index_col=0)['Adj Close'].pct_change().dropna()
        for file in stock_files
    ], axis=1)
    returns_matrix.columns = [file.replace('.csv', '') for file in stock_files]

    num_stocks = len(returns_matrix.columns)

    # Initialize parameters
    population_size = lambda_ + mu
    num_generations = 50
    mutation_stddev = 0.1

    # Initialize population
    population = np.random.dirichlet(np.ones(num_stocks), size=population_size)

    def fitness(weights):
        # Calculate fitness as negative of portfolio variance (minimization problem)
        portfolio_return = np.dot(returns_matrix.mean(), weights)
        portfolio_variance = np.dot(weights.T, np.dot(returns_matrix.cov(), weights))
        return -portfolio_return

    # Evolutionary strategy algorithm
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_values = np.array([fitness(individual) for individual in population])

        # Select the best individuals
        sorted_indices = np.argsort(fitness_values)
        best_individuals = population[sorted_indices][:mu]

        # Create offspring through mutation
        offspring = []
        for individual in best_individuals:
            for _ in range(lambda_ // mu):
                mutant = np.clip(individual + np.random.normal(0, mutation_stddev, size=num_stocks), 0, 1)
                mutant /= mutant.sum()  # Normalize weights
                offspring.append(mutant)

        # Combine parent and offspring populations
        population = np.vstack((best_individuals, np.array(offspring)))

    # Save optimized portfolio weights
    best_individual = population[np.argmin([fitness(ind) for ind in population])]
    pd.DataFrame(best_individual, index=returns_matrix.columns, columns=['Weight']).to_csv(
        os.path.join(results_path, 'optimized_portfolio_mu_plus_lambda_es.csv'))


if __name__ == "__main__":
    mu_plus_lambda_es(mu=10, lambda_=20)
