import numpy as np
import pandas as pd
import os

def calculate_expected_return(weights, returns):
    """
    Calculate the expected return of the portfolio.
    """
    mean_returns = returns.mean()
    return np.dot(weights, mean_returns)

def load_returns(file):
    """
    Load returns from a CSV file and handle potential errors.
    """
    try:
        df = pd.read_csv(file, index_col=0, encoding='ISO-8859-1')
        return df['Close'].pct_change().dropna()
    except KeyError as e:
        print(f"Column not found in file {file}: {e}")
        return pd.Series()
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file}")
        return pd.Series()

def advanced_ep_optimization(generations=100, population_size=50):
    """
    Perform advanced evolutionary programming to optimize portfolio weights.
    """
    stock_files = [f for f in os.listdir("../stocks_csv") if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in stock_files]

    # Read stock data and ensure correct encoding
    returns_matrix = pd.concat([
        load_returns(f"../stocks_csv/{file}") for file in stock_files
    ], axis=1)
    returns_matrix.columns = tickers

    num_assets = len(tickers)
    population = np.random.dirichlet(np.ones(num_assets), population_size)

    mutation_rate = 0.05  # Self-adaptive mutation rate

    for generation in range(generations):
        fitness = []
        for portfolio in population:
            expected_return = calculate_expected_return(portfolio, returns_matrix)
            fitness.append(expected_return)

        best_indices = np.argsort(fitness)[-int(population_size / 2):]
        best_portfolios = population[best_indices]

        # Self-adaptive mutation
        mutation_rate = mutation_rate * (1 + np.random.normal(0, 0.01))
        mutations = np.random.normal(0, mutation_rate, best_portfolios.shape)
        population = np.abs(best_portfolios + mutations)
        population = population / population.sum(axis=1)[:, None]

    best_portfolio = population[np.argmax(fitness)]
    results_path = os.path.join(os.path.dirname(__file__), "../results", "optimized_portfolio_advanced_ep.csv")
    pd.DataFrame(best_portfolio, index=tickers, columns=["Weight"]).to_csv(results_path)
    print(f"Optimized portfolio weights saved to {results_path}")

if __name__ == "__main__":
    advanced_ep_optimization()
