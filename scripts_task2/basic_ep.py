import numpy as np
import pandas as pd
import os

def calculate_expected_return(weights, returns):
    """Calculate the expected return for the portfolio given the weights and returns."""
    mean_returns = returns.apply(pd.to_numeric, errors='coerce').mean()
    return np.dot(weights, mean_returns)

def basic_ep_optimization(generations=100, population_size=50):
    """Basic Evolutionary Programming (EP) to optimize portfolio weights."""
    stock_folder = "../results/stocks_results"
    stock_files = [f for f in os.listdir(stock_folder) if f.endswith('.csv')]

    # Read monthly returns for all stocks
    returns_matrix = pd.DataFrame()
    for stock_file in stock_files:
        stock_path = os.path.join(stock_folder, stock_file)
        returns = pd.read_csv(stock_path, index_col=0, header=None)

        # Ensure the returns are numeric
        returns_numeric = returns.apply(pd.to_numeric, errors='coerce').squeeze()
        returns_matrix[stock_file.replace('.csv', '')] = returns_numeric

    num_assets = len(stock_files)
    tickers = [file.replace('.csv', '') for file in stock_files]

    # Initialize random portfolio weights (non-negative and normalized)
    population = np.random.dirichlet(np.ones(num_assets), population_size)

    for generation in range(generations):
        fitness = []
        for portfolio in population:
            # Calculate expected return
            expected_return = calculate_expected_return(portfolio, returns_matrix)
            fitness.append(expected_return)

        # Select the best portfolios (maximize expected return)
        best_indices = np.argsort(fitness)[-int(population_size / 2):]
        best_portfolios = population[best_indices]

        # Mutate the best portfolios, ensuring no negative weights
        mutations = np.random.normal(0, 0.1, best_portfolios.shape)
        mutated_portfolios = best_portfolios + mutations

        # Ensure non-negative weights and normalize to sum to 1
        mutated_portfolios = np.abs(mutated_portfolios)  # Make all values non-negative
        mutated_portfolios = mutated_portfolios / mutated_portfolios.sum(axis=1, keepdims=True)

        population = mutated_portfolios

    # Get the best portfolio from the final generation
    best_portfolio = population[np.argmax(fitness)]

    # Convert the portfolio into a DataFrame
    portfolio_df = pd.DataFrame(best_portfolio, index=tickers, columns=["Weight"])

    # Add a sum row at the bottom
    sum_row = pd.DataFrame([portfolio_df["Weight"].sum()], index=["Sum"], columns=["Weight"])
    portfolio_df = pd.concat([portfolio_df, sum_row])

    # Save the optimized portfolio weights with the sum row
    results_path = os.path.join(os.path.dirname(__file__), "../results", "optimized_portfolio_basic_ep.csv")
    portfolio_df.to_csv(results_path)

    print(f"Optimized portfolio weights saved to {results_path}")


if __name__ == "__main__":
    basic_ep_optimization()
