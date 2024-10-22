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
    def __init__(self, num_tenants, industry_data, max_floor_sf=20000, initial_vacancy=0.1):
        self.num_tenants = num_tenants
        self.industry_data = industry_data
        self.max_floor_sf = max_floor_sf
        self.state_dim = 9
        self.action_dim = 3
        self.vacancy_rate = initial_vacancy
        self.suites = []
        self.reset()

    def reset(self):
        self.suites = self._create_suites()
        self.state = self._get_initial_state()
        return self.state

    def _create_suites(self):
        total_sf = 0
        suite_counter = 1
        suites = []

        # Initialize suites until reaching max floor SF
        while total_sf < self.max_floor_sf:
            suite_sf = np.random.uniform(500, 5000)  # Random suite size
            floor_level = np.random.randint(1, 31)  # Random floor between 1 and 30

            if total_sf + suite_sf > self.max_floor_sf:
                break

            suite_id = f"{floor_level}_{suite_counter}"
            suites.append(Suite(suite_id, suite_sf, floor_level))
            total_sf += suite_sf
            suite_counter += 1

        return suites

    def _get_initial_state(self):
        industry_growth = np.random.choice(self.industry_data['growth_rate'])
        lease_length = np.random.choice([5, 7, 15])
        rsf_occupied = np.random.uniform(500, 5000)
        tia_status = np.random.choice([0, 1])
        occupancy_rate = np.random.uniform(0.5, 1.0)
        incentives_available = np.random.choice([0, 1, 2])
        floor_level = np.random.randint(1, 31)
        economic_indicator = np.random.choice([-1, 0, 1])
        vacancy_rate = self.vacancy_rate

        return np.array([
            industry_growth, lease_length, rsf_occupied, tia_status,
            occupancy_rate, incentives_available, floor_level,
            economic_indicator, vacancy_rate
        ])

    def step(self, action):
        reward = 0
        done = False

        if action == 0:  # Renew
            reward = self._renew_lease()
        elif action == 1:  # Expand
            reward = self._expand_space()
        elif action == 2:  # Terminate
            reward = self._terminate_lease()

        # Penalize high vacancy rates more severely
        if self.state[8] > 0.5:
            reward -= 10

        # Reward high occupancy rates during economic growth
        if self.state[7] == 1 and self.state[4] > 0.8:
            reward += 5

        # Add tenant vacating logic
        if self.state[1] <= 0:  # Lease term ends
            vacate_prob = self._calculate_vacate_probability()
            if np.random.rand() < vacate_prob:
                self._vacate_suites()
                reward -= 10
            else:
                self.state[1] = np.random.choice([5, 7, 15])  # Renew lease

        self.state = self._update_state(action)
        done = self._check_done()

        return self.state, reward, done

    def _vacate_suites(self):
        for suite in self.suites:
            if suite.occupied and suite.lease_end_year <= self.state[1]:
                suite.vacate()

    def _renew_lease(self):
        lease_bonus = 0
        if self.state[1] > 10:
            lease_bonus = 5
            self.state[2] *= 1.3

        for suite in self.suites:
            if suite.occupied:
                suite.lease(suite.tenant_id, suite.lease_id, self.state[1], suite.lease_end_year + self.state[1])

        return 10 + lease_bonus if np.random.random() < 0.7 else -5

    def _expand_space(self):
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
        for suite in self.suites:
            if suite.occupied:
                suite.vacate()
                return -10
        return 0

    def _calculate_vacate_probability(self):
        industry_growth, lease_length, rsf_occupied, tia_status, \
        occupancy_rate, incentives_available, floor_level, economic_indicator, vacancy_rate = self.state

        vacate_prob = 0.3
        if economic_indicator == -1:
            vacate_prob += 0.5
        elif economic_indicator == 1:
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
        industry_growth, lease_length, rsf_occupied, tia_status, \
        occupancy_rate, incentives_available, floor_level, economic_indicator, vacancy_rate = self.state

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

        if incentives_available > 0:
            occupancy_rate += 0.05

        if economic_indicator == -1:
            vacancy_rate += 0.1
        elif economic_indicator == 1:
            vacancy_rate -= 0.05

        vacancy_rate = max(0, min(vacancy_rate, 1))

        return np.array([
            industry_growth, lease_length, rsf_occupied, tia_status,
            occupancy_rate, incentives_available, floor_level,
            economic_indicator, vacancy_rate
        ])

    def _check_done(self):
        return all(not suite.occupied for suite in self.suites)
