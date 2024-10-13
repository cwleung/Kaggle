from GPyOpt.methods import BayesianOptimization
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

np.random.seed(42)
n_samples = 500
prices = np.random.uniform(10, 80, n_samples)
seasons = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
promotions = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
competitors_price = prices + np.random.normal(0, 5, n_samples)

true_alpha = 100
true_beta_price = -1.5
true_beta_season = {'Spring': 5, 'Summer': 10, 'Fall': 0, 'Winter': -5}
true_beta_promotion = 15
true_beta_competitor = 0.5

base_demand = (true_alpha +
               true_beta_price * prices +
               np.array([true_beta_season[s] for s in seasons]) +
               true_beta_promotion * promotions +
               true_beta_competitor * (prices - competitors_price))
noise = np.random.normal(0, 10, n_samples)
demand = np.maximum(base_demand + noise, 0)  # Ensure non-negative demand

df = pd.DataFrame({
    'price': prices,
    'season': seasons,
    'promotion': promotions,
    'competitor_price': competitors_price,
    'demand': demand
})

with pm.Model() as pricing_model:
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    beta_price = pm.Normal('beta_price', mu=0, sigma=10)
    beta_season = pm.Normal('beta_season', mu=0, sigma=10, shape=4)
    beta_promotion = pm.Normal('beta_promotion', mu=0, sigma=10)
    beta_competitor = pm.Normal('beta_competitor', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=10)

    season_idx = pm.math.switch(df.season == 'Spring', 0,
                                pm.math.switch(df.season == 'Summer', 1,
                                               pm.math.switch(df.season == 'Fall', 2, 3)))

    demand_pred = (alpha +
                   beta_price * df.price +
                   beta_season[season_idx] +
                   beta_promotion * df.promotion +
                   beta_competitor * (df.price - df.competitor_price))

    y = pm.Normal('y', mu=demand_pred, sigma=sigma, observed=df.demand)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

az.plot_trace(trace)
plt.show()

summary = az.summary(trace, round_to=2)
print(summary)


def expected_revenue(price, season, promotion, competitor_price, trace):
    alpha = trace.posterior['alpha'].mean().item()
    beta_price = trace.posterior['beta_price'].mean().item()
    beta_season = trace.posterior['beta_season'].mean(axis=(0,1))
    beta_promotion = trace.posterior['beta_promotion'].mean().item()
    beta_competitor = trace.posterior['beta_competitor'].mean().item()

    season_idx = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}[season]

    demand = (alpha +
              beta_price * price +
              beta_season[season_idx] +
              beta_promotion * promotion +
              beta_competitor * (price - competitor_price))

    return price * max(demand, 0)

def objective(params):
    price = params[0][0]
    season_idx = int(params[0][1])
    promotion = int(params[0][2])
    competitor_price = params[0][3]

    if price <= 0 or competitor_price <= 0:
        return 1e6

    season = ['Spring', 'Summer', 'Fall', 'Winter'][season_idx]
    revenue = expected_revenue(price, season, promotion, competitor_price, trace)
    return -revenue  # We minimize the negative revenue (equivalent to maximizing revenue)

# Define the domain for Bayesian optimization
domain = [
    {'name': 'price', 'type': 'continuous', 'domain': (1, 100)},
    {'name': 'season', 'type': 'discrete', 'domain': (0, 1, 2, 3)},
    {'name': 'promotion', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'competitor_price', 'type': 'continuous', 'domain': (1, 100)}
]

# Run Bayesian optimization
optimizer = BayesianOptimization(f=objective, domain=domain, maximize=False)
optimizer.run_optimization(max_iter=50)

# Get the optimal parameters
optimal_params = optimizer.x_opt
optimal_price, season_idx, promotion, competitor_price = optimal_params
season = ['Spring', 'Summer', 'Fall', 'Winter'][int(season_idx)]

print(f"\nOptimal Price: ${optimal_price:.2f}")
print(f"Optimal Season: {season}")
print(f"Optimal Promotion: {'Yes' if promotion == 1 else 'No'}")
print(f"Competitor Price: ${competitor_price:.2f}")
print(f"Maximum Expected Revenue: ${-optimizer.fx_opt:.2f}")


# Plot optimization progress
plt.figure(figsize=(10, 6))
plt.plot(range(len(optimizer.Y)), -optimizer.Y)
plt.xlabel('Iteration')
plt.ylabel('Best Revenue Found')
plt.title('Bayesian Optimization Progress')
plt.show()

# Visualize the optimization landscape
if optimal_params[1] == 0 and optimal_params[2] == 0:  # Spring and No Promotion
    prices = np.linspace(0, 100, 100)
    competitor_prices = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(prices, competitor_prices)
    Z = np.array([[expected_revenue(p, 'Spring', 0, cp, trace) for p, cp in zip(x, y)] for x, y in zip(X, Y)])

    plt.figure(figsize=(12, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Expected Revenue')
    plt.xlabel('Price')
    plt.ylabel('Competitor Price')
    plt.title('Revenue Landscape (Spring, No Promotion)')
    plt.plot(optimal_price, competitor_price, 'r*', markersize=15, label='Optimal Point')
    plt.legend()
    plt.show()

# Uncertainty quantification
def _optimal_price(alpha, beta_price):
    return -alpha / (2 * beta_price)


optimal_prices = _optimal_price(trace.posterior['alpha'], trace.posterior['beta_price'])
credible_interval = np.percentile(optimal_prices.values.flatten(), [2.5, 97.5])

print(f"\n95% Credible Interval for Optimal Price (base scenario): ${credible_interval[0]:.2f} to ${credible_interval[1]:.2f}")


# Plot histogram of optimal prices
plt.figure(figsize=(10, 6))
plt.hist(optimal_prices.values.flatten(), bins=50, density=True)
plt.axvline(optimal_price, color='r', linestyle='--', label=f'Bayesian Opt Price: ${float(optimal_price):.2f}')
plt.axvline(credible_interval[0], color='g', linestyle='--', label='95% Credible Interval')
plt.axvline(credible_interval[1], color='g', linestyle='--')
plt.xlabel('Optimal Price')
plt.ylabel('Density')
plt.title('Posterior Distribution of Optimal Price (Base Scenario)')
plt.legend()
plt.show()