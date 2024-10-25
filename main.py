from src.environment.building_env_simulation import BuildingEnvironment  
from src.agents.td3_agent import TD3Agent 
from src.utils.data_preprocessing import preprocess_industry_data 
from src.utils.simulation_utils import simulate_leasing_with_vacating
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time
from results.metrics.calculate_metrics import calculate_rl_metrics  # Import the function
import logging

gdp_growths, unemployment_rates, inflations = [], [], []

def main():

    def generate_synthetic_economic_factors(years, scenario="recession"):
        """
        Generate synthetic economic factors based on the given scenario.

        Args:
        - years (int): Number of years for which to generate synthetic data.
        - scenario (str): Economic scenario to simulate ("boom", "recession", "stability").

        Returns:
        - dict: Dictionary containing 'gdp_growth', 'unemployment', and 'inflation' time series.
        """
        if scenario == "boom":
            # Simulate economic boom: High GDP growth, low unemployment, moderate to high inflation
            gdp_growth = np.random.uniform(0.02, 0.05, years)  # 2% to 5% growth
            unemployment = np.random.uniform(0.03, 0.05, years)  # 3% to 5% unemployment
            inflation = np.random.uniform(0.03, 0.06, years)  # 3% to 6% inflation
        elif scenario == "recession":
            # Simulate recession: Negative GDP growth, high unemployment, low inflation
            gdp_growth = np.random.uniform(-0.05, 0.0, years)  # -5% to 0% growth
            unemployment = np.random.uniform(0.07, 0.12, years)  # 7% to 12% unemployment
            inflation = np.random.uniform(0.0, 0.02, years)  # 0% to 2% inflation
        else:  # Default to stability
            # Simulate economic stability: Moderate GDP growth, stable unemployment, low to moderate inflation
            gdp_growth = np.random.uniform(0.0, 0.03, years)  # 0% to 3% growth
            unemployment = np.random.uniform(0.04, 0.06, years)  # 4% to 6% unemployment
            inflation = np.random.uniform(0.02, 0.04, years)  # 2% to 4% inflation

        return {
            'gdp_growth': gdp_growth,
            'unemployment': unemployment,
            'inflation': inflation
        }

    # Set up logging configuration
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'training.log'

    logging.basicConfig(
        level=logging.INFO,  # Set logging level to INFO (can also be DEBUG)
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
    # File path for industry data
    file_path = 'data/raw/DOL_data.csv'

    # Preprocess industry data
    industry_data = preprocess_industry_data(file_path)

    # Initialize environment and agent
    years = 32  # Number of years to simulate
    scenario = "stability"  # Choose from "boom", "recession", or "stability"

    # Generate synthetic economic factors
    econ_factors = generate_synthetic_economic_factors(years, scenario)
    # Collect economic factors over simulation
    gdp_growths.extend(econ_factors['gdp_growth'])
    unemployment_rates.extend(econ_factors['unemployment'])
    inflations.extend(econ_factors['inflation'])

    # Create a DataFrame for economic factors
    econ_factors_df = pd.DataFrame({
        'Year': np.arange(1990, 1990 + years),
        'GDP Growth': gdp_growths,
        'Unemployment Rate': unemployment_rates,
        'Inflation': inflations
    })

    env = BuildingEnvironment(num_tenants=500, industry_data=industry_data, econ_factors=econ_factors)
    agent = TD3Agent(state_dim=12, max_rent=10000, max_lease_length=15)

    # Hyperparameters
    num_episodes = 500
    max_steps = 500

    # Lists to collect state variables and RL metrics
    industry_growths, lease_lengths, rsf_occupieds = [], [], []
    occupancy_rates, incentives_available, economic_indicators, vacancy_rates = [], [], [], []
    success_threshold = 4  # Define a success threshold
    successful_episodes = 0
    all_rewards = []
    episode_lengths = []
    exploration_ratios = []
    logging.info("Starting RL training...")

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0

        for step in range(max_steps):
            # Agent selects action based on current state
            rent_amount, lease_length, vacancy_rate = agent.act(state)
            action = [rent_amount, lease_length, vacancy_rate]

            # Environment steps forward based on action
            next_state, reward, done = env.step(action)

            # Agent learns from the experience
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            # Collect state variables for analysis or plotting
            industry_growths.append(next_state[0])
            lease_lengths.append(next_state[1])
            rsf_occupieds.append(next_state[2])
            occupancy_rates.append(next_state[4])
            incentives_available.append(next_state[5])
            economic_indicators.append(next_state[7])
            vacancy_rates.append(next_state[8])

            # Update state and total reward
            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break
        # Store total reward and episode length
        all_rewards.append(total_reward)
        episode_lengths.append(step_count)

        if total_reward >= success_threshold:
            successful_episodes += 1
            # Calculate exploration ratio periodically
        if (episode + 1) % 10 == 0:
            exploration_ratio = agent.get_exploration_ratio()
            exploration_ratios.append(exploration_ratio)
            # Log every 50 episodes
        if (episode + 1) % 50 == 0:
            success_rate = (successful_episodes / (episode + 1)) * 100
            reward_variance = np.var(all_rewards)
            logging.info(
                f"Episode {episode + 1}/{num_episodes}, Success Rate: {success_rate:.2f}%, Reward Variance: {reward_variance:.2f}, Exploration Ratio: {exploration_ratio:.2f}")

        # Update console output in place using sys.stdout.write and flush
        sys.stdout.write(f"\rEpisode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        sys.stdout.flush()
        time.sleep(0.01)

    # Final output to indicate training completion
    sys.stdout.write("\nTraining completed.\n")
    # Final log to indicate training completion
    logging.info("Training completed.")

    # Calculate and log RL metrics
    rl_metrics = calculate_rl_metrics(all_rewards, episode_lengths)
    logging.info("Reinforcement Learning Metrics:")
    for metric, value in rl_metrics.items():
        logging.info(f"{metric}: {value}")
    # Final log after training
    final_success_rate = (successful_episodes / num_episodes) * 100
    logging.info(f"Final Success Rate: {final_success_rate:.2f}%")
    # Log the final exploration vs. exploitation ratio
    exploration_ratio = agent.get_exploration_ratio()
    logging.info(f"Exploration vs. Exploitation Ratio: {exploration_ratio:.2f}")
    # Create DataFrame from collected state variables
    state_df = pd.DataFrame({
        'IndustryGrowth': industry_growths,
        'LeaseLength': lease_lengths,
        'RSFOccupied': rsf_occupieds,
        'OccupancyRate': occupancy_rates,
        'IncentivesAvailable': incentives_available,
        'EconomicIndicator': economic_indicators,
        'VacancyRate': vacancy_rates
    })

    # Define processed data directory
    processed_data_dir = Path('data/processed')
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Save state_df and leases_df to CSV
    output_file = processed_data_dir / 'leases_df_recession.csv'
    state_file = processed_data_dir / 'state_variables_recession.csv'

    leases_df = simulate_leasing_with_vacating(agent, env, start_year=1990, num_years=32)
    leases_df.to_csv(output_file, index=False)
    print(f"\nleases_df.csv saved at {output_file}")
    state_df.to_csv(state_file, index=False)
    print(f"State variables saved to {state_file}")
    # Save economic factors to CSV
    econ_factors_file = processed_data_dir / 'economic_factors_recession.csv'
    econ_factors_df.to_csv(econ_factors_file, index=False)
    print(f"Economic factors saved to {econ_factors_file}")


if __name__ == "__main__":
    main()
