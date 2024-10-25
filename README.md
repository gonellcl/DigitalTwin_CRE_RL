# DigitalTwin_CRE_RL
Digital Twin Simulation for Commercial Real Estate with Reinforcement Learning agents simulating leasing trends, tenant behaviors, and economic events.
## Project Description
The project provides an adaptive digital twin model of a commerical building designed to demonstrate the potential benefits of using digital twins for forecasting, analysis, and decision-making in real estate management. While the model is intentionally limited in scope and complexity, it offers a foundational framework for optimizing leasing strategies through simulated interactions between tenants and economic variables. 

This digital twin captures tenant behavior in response to macroeconomic trends between 1990-2023. By simulating diverse economic conditions such as growth periods, downturns, and market shocks, the model provides insignt into how broader economic forces impact tenant decisions, lease lengths, and oocupancy rates. The project acts as a scalable framework that can be expanded to model more complex relationships and dynmaics within the commerical real estate sector. 

Ultimately, the adaptive digital twin servies as a decision-support tool for real estate professionals as it supports strategic evaluation of leasing policies, tenant retention efforts, and economic resilience in simulated real-world condiitons. 

## Features
- **Digital Twin Simulation:** Models leasing dynamics in a multi-floor commercial building.
- **Reinforcement Learning Optimization:** Uses the TD3 algorithm to optimize lease decisions based on state variables.
- **Dynamic State Variables:** Simulates economic growth, occupancy rates, vacancy rates, and other real estate metrics.
- **Sensitivity Analysis:** Analyzes the impact of various state variables on lease outcomes.
- **Interactive Visualization:** Provides Jupyter Notebooks for exploratory analysis and visualization.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DigitalTwin_CRE_RL.git
   cd DigitalTwin_CRE_RL
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install Jupyter (if needed):**
   ```bash
   pip install jupyter
   ```
## Usage
- **Run the main simulation:**
  ```bash
  python main.py
  ```
- **Run the Real Estate Digital Twin Dashboard:**
  ```bash
  python streamlit run streamlit_app.py

  ```
- **Explore Jupyter Notebooks:**
  Open the `notebooks/` or  `plots/` directory and explore individual notebooks like `analysis.ipynb` , `sensitivity_analysis.ipynb` or `forecasting.ipynb` for interactive exploration.
- **Run sensitivity analysis:**
  Go to the `simulations/` directory and execute scripts like `run_sensitivity_analysis.py` to explore the impact of different state variables.

## Data Sources
- **Department of Labor Data:** Raw data is sourced from DOL datasets on industry growth and other economic indicators. (https://catalog.data.gov/dataset/department-of-labor-office-of-research-current-employment-statistics-nsa-1990-current)
- **Synthetic Data:** Synthetic datasets are generated to simulate tenant behavior, lease terms, and other real estate metrics.
