from src.environment.building_environment import BuildingEnvironment  # Import your environment class
from src.agents.td3_agent import TD3Agent  # Import your agent class (assuming it's in a module)
from src.utils.data_preprocessing import preprocess_industry_data  # Import your data preprocessing function
from src.utils.simulation_utils import simulate_leasing_with_vacating


def main():
    # File path for industry data
    file_path = 'data/raw/DOL_data.csv'

    # Preprocess industry data
    industry_data = preprocess_industry_data(file_path)

    # Initialize environment and agent
    env = BuildingEnvironment(num_tenants=5000, industry_data=industry_data)
    agent = TD3Agent(state_dim=9, max_rent=10000, max_lease_length=15)

    # Hyperparameters
    num_episodes = 1000
    max_steps = 200

    # Lists to collect state variables over episodes
    industry_growths = []
    lease_lengths = []
    rsf_occupieds = []
    occupancy_rates = []
    incentives_available = []
    economic_indicators = []
    vacancy_rates = []

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

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

            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    print("Training completed.")

    leases_df = simulate_leasing_with_vacating(agent, env, start_year=1990, num_years=20)


if __name__ == "__main__":
    main()
