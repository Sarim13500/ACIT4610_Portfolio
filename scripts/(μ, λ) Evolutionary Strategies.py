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
        return df['Adj Close'].pct_change().dropna()  # Ensure column name is correct
    except KeyError as e:
        print(f"Column not found in file {file}: {e}")
        return pd.Series()
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file}")
        return pd.Series()


def mu_lambda_es_optimization(generations=100, mu=10, lambda_=50):
    """
    Perform (μ, λ) evolutionary strategies to optimize portfolio weights.

    Parameters:
    - generations: Number of generations to run the optimization
    - mu: Number of parents in the evolutionary strategy
    - lambda_: Number of offspring to generate per generation
    """
    stock_files = [f for f in os.listdir("../stocks_csv") if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in stock_files]

    # Read stock data and ensure correct encoding
    returns_matrix = pd.concat([
        load_returns(f"../stocks_csv/{file}") for file in stock_files
    ], axis=1)
    returns_matrix.columns = tickers

    num_assets = len(tickers)
    population = np.random.dirichlet(np.ones(num_assets), lambda_)  # Initialize population with lambda_

    for generation in range(generations):
        fitness = []
        for portfolio in population:
            if portfolio.sum() <= 0:  # Ensure no division by zero
                portfolio /= portfolio.sum() + 1e-6
            expected_return = calculate_expected_return(portfolio, returns_matrix)
            fitness.append(expected_return)

        fitness = np.array(fitness)
        best_indices = np.argsort(fitness)[-mu:]  # Select top mu portfolios
        best_portfolios = population[best_indices]

        offspring = []
        for i in range(lambda_):
            parent1, parent2 = best_portfolios[np.random.choice(len(best_portfolios), 2, replace=False)]
            child = (parent1 + parent2) / 2
            mutation = np.random.normal(0, 0.1, num_assets)
            child = np.abs(child + mutation)
            child /= child.sum()
            offspring.append(child)

        population = np.array(offspring)

    best_portfolio = population[np.argmax(fitness)]
    results_path = os.path.join(os.path.dirname(__file__), "../results", "optimized_portfolio_mu_lambda_es.csv")
    pd.DataFrame(best_portfolio, index=tickers, columns=["Weight"]).to_csv(results_path)
    print(f"Optimized portfolio weights saved to {results_path}")


if __name__ == "__main__":
    # Example of how to call the function with different mu and lambda values
    mu_lambda_es_optimization(generations=100, mu=10, lambda_=20)
