import numpy as np


class Suite:
    def __init__(self, suite_id, suite_sf, floor_level):
        self.suite_id = suite_id
        self.suite_sf = suite_sf
        self.occupied = False
        self.tenant_id = None
        self.lease_id = None
        self.lease_start_year = None
        self.lease_end_year = None
        self.floor_level = floor_level

    def lease(self, tenant_id, lease_id, lease_start_year, lease_end_year):
        self.occupied = True
        self.tenant_id = tenant_id
        self.lease_id = lease_id
        self.lease_start_year = lease_start_year
        self.lease_end_year = lease_end_year

    def vacate(self):
        self.occupied = False
        self.tenant_id = None
        self.lease_id = None
        self.lease_start_year = None
        self.lease_end_year = None


class BuildingEnvironment:
    def __init__(self, num_tenants, industry_data, max_floor_sf=20000, initial_vacancy=0.1, econ_factors=None):
        """
        Initialize the BuildingEnvironment class.

        Args:
        - num_tenants (int): Number of tenants in the building.
        - industry_data (DataFrame): Industry data for state initialization.
        - max_floor_sf (int): Maximum floor square footage.
        - initial_vacancy (float): Initial vacancy rate.
        - econ_factors (dict): Dictionary containing synthetic economic factors.
        """
        self.num_tenants = num_tenants
        self.industry_data = industry_data
        self.max_floor_sf = max_floor_sf
        self.vacancy_rate = initial_vacancy
        self.econ_factors = econ_factors if econ_factors is not None else self._generate_default_factors()
        self.state_dim = 12  # Adjusted state dimension to include economic indicators
        self.action_dim = 3
        self.suites = []
        self.reset()

    def _generate_default_factors(self):
        """
        Generate default economic factors if none are provided.
        """
        years = 32  # Default simulation duration
        return {
            'gdp_growth': np.random.uniform(0.0, 0.03, years),
            'unemployment': np.random.uniform(0.04, 0.06, years),
            'inflation': np.random.uniform(0.02, 0.04, years)
        }

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.suites = self._create_suites()
        self.state = self._get_initial_state()
        return self.state

    def _create_suites(self):
        """
        Create suites in the building with random sizes.
        """
        total_sf = 0
        suite_counter = 1
        suites = []

        while total_sf < self.max_floor_sf:
            suite_sf = np.random.uniform(500, 5000)  # Random suite size
            floor_level = np.random.randint(1, 31)  # Random floor level

            if total_sf + suite_sf > self.max_floor_sf:
                break

            suite_id = f"{floor_level}_{suite_counter}"
            suites.append(Suite(suite_id, suite_sf, floor_level))
            total_sf += suite_sf
            suite_counter += 1

        return suites

    def _get_initial_state(self):
        """
        Generate the initial state, including economic indicators.
        """
        # Select a random year index for the economic indicators
        year_idx = np.random.randint(0, len(self.econ_factors['gdp_growth']))

        # Extract economic indicators for the selected year
        economic_growth = 1 if self.econ_factors['gdp_growth'][year_idx] > 0.02 else \
            -1 if self.econ_factors['gdp_growth'][year_idx] < -0.02 else 0
        unemployment_rate = self.econ_factors['unemployment'][year_idx]
        gdp_growth_rate = self.econ_factors['gdp_growth'][year_idx]
        inflation_rate = self.econ_factors['inflation'][year_idx]

        # Other initial state variables
        industry_growth = np.random.choice(self.industry_data['growth_rate'])
        lease_length = np.random.choice([5, 7, 15])
        rsf_occupied = np.random.uniform(500, 5000)
        tia_status = np.random.choice([0, 1])
        occupancy_rate = np.random.uniform(0.5, 1.0)
        incentives_available = np.random.choice([0, 1, 2])
        floor_level = np.random.randint(1, 31)
        vacancy_rate = self.vacancy_rate

        return np.array([
            industry_growth, lease_length, rsf_occupied, tia_status,
            occupancy_rate, incentives_available, economic_growth,
            unemployment_rate, gdp_growth_rate, inflation_rate,
            floor_level, vacancy_rate
        ])

    def step(self, action):
        """
        Take a step in the environment based on the action provided.
        """
        reward = 0
        done = False

        if action == 0:  # Renew lease
            reward = self._renew_lease()
        elif action == 1:  # Expand space
            reward = self._expand_space()
        elif action == 2:  # Terminate lease
            reward = self._terminate_lease()

        # Adjust vacancy and reward based on economic indicators
        if self.state[11] > 0.5:  # High vacancy rate penalty
            reward -= 5
        if self.state[6] == 1 and self.state[4] > 0.8:  # Economic growth with high occupancy
            reward += 5

        if self.state[1] <= 0:  # Lease term ends
            vacate_prob = self._calculate_vacate_probability()
            if np.random.rand() < vacate_prob:
                self._vacate_suites()
                reward -= 10
            else:
                self.state[1] = np.random.choice([5, 7, 15])  # Renew lease

        # Update the state based on the action
        self.state = self._update_state(action)
        done = self._check_done()

        return self.state, reward, done

    def _vacate_suites(self):
        """
        Vacate suites where leases have ended.
        """
        for suite in self.suites:
            if suite.occupied and suite.lease_end_year <= self.state[1]:
                suite.vacate()

    def _renew_lease(self):
        """
        Renew leases and update the reward.
        """
        lease_bonus = 0
        if self.state[1] > 10:
            lease_bonus += 5
            self.state[2] *= 1.3

        for suite in self.suites:
            if suite.occupied:
                suite.lease(suite.tenant_id, suite.lease_id, self.state[1], suite.lease_end_year + self.state[1])

        return 10 + lease_bonus if np.random.random() < 0.7 else -5

    def _expand_space(self):
        """
        Expand space for unoccupied suites.
        """
        reward = 5 if np.random.random() < 0.5 else -5

        for suite in self.suites:
            if not suite.occupied:
                potential_sf = suite.suite_sf * 1.2
                available_sf = self.max_floor_sf - suite.suite_sf
                suite.suite_sf += min(potential_sf - suite.suite_sf, available_sf)
                reward += 5
                break

        return reward

    def _terminate_lease(self):
        """
        Terminate leases when conditions are met.
        """
        for suite in self.suites:
            if suite.occupied:
                suite.vacate()
                return -5
        return 5

    def _calculate_vacate_probability(self):
        """
        Calculate the probability of vacating based on state variables.
        """
        industry_growth, lease_length, rsf_occupied, tia_status, \
        occupancy_rate, incentives_available, economic_growth, \
        unemployment_rate, gdp_growth_rate, inflation_rate, \
        floor_level, vacancy_rate = self.state

        vacate_prob = 0.3
        if economic_growth == -1:
            vacate_prob += 0.5
        elif economic_growth == 1:
            vacate_prob -= 0.2

        if occupancy_rate < 0.7:
            vacate_prob += 0.2
        if incentives_available == 0:
            vacate_prob += 0.3

        if rsf_occupied > 3000:
            vacate_prob -= 0.1
        if lease_length > 7:
            vacate_prob -= 0.1

        return min(vacate_prob, 1.0)

    def _update_state(self, action):
        """
        Update the environment's state based on the agent's action and economic indicators.
        """
        # Unpack current state variables
        industry_growth, lease_length, rsf_occupied, tia_status, \
        occupancy_rate, incentives_available, economic_growth, \
        unemployment_rate, gdp_growth_rate, inflation_rate, \
        floor_level, vacancy_rate = self.state

        # Apply action effects (Renew, Expand, Terminate)
        if action == 0:  # Renew
            lease_length += np.random.choice([5, 7, 15])
            if incentives_available > 0:
                rsf_occupied *= 1.2
                vacancy_rate -= 0.1

        elif action == 1:  # Expand
            rsf_occupied *= 1.1
            if incentives_available > 0:
                rsf_occupied *= 1.1
                vacancy_rate -= 0.05

        elif action == 2:  # Terminate
            rsf_occupied = 0

        # Adjust state based on economic indicators
        if economic_growth == -1:  # Recession
            vacancy_rate += 0.1
            lease_length = max(1, lease_length - 2)  # Shorter leases
        elif economic_growth == 1:  # Boom
            vacancy_rate -= 0.05
            lease_length = min(15, lease_length + 1)  # Longer leases

        # Adjust occupancy and vacancy rates based on unemployment and GDP growth
        if unemployment_rate > 0.07:
            vacancy_rate += 0.05
        elif unemployment_rate < 0.04:
            vacancy_rate -= 0.05

        if gdp_growth_rate > 0.02:
            occupancy_rate += 0.05
        elif gdp_growth_rate < -0.02:
            occupancy_rate -= 0.05

        # Inflation's impact on incentives
        if inflation_rate > 0.04:
            incentives_available -= 1
        elif inflation_rate < 0.02:
            incentives_available += 1

        # Keep vacancy and occupancy rates within bounds
        vacancy_rate = max(0, min(vacancy_rate, 1))
        occupancy_rate = max(0, min(occupancy_rate, 1))
        incentives_available = max(0, incentives_available)

        return np.array([
            industry_growth, lease_length, rsf_occupied, tia_status,
            occupancy_rate, incentives_available, economic_growth,
            unemployment_rate, gdp_growth_rate, inflation_rate,
            floor_level, vacancy_rate
        ])

    def _check_done(self):
        """
        Check if all suites are vacated.
        """
        return all(not suite.occupied for suite in self.suites)
