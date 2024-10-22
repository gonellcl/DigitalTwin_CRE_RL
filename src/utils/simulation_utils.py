import numpy as np
import pandas as pd


def simulate_leasing_with_vacating(agent, environment, start_year=1990, num_years=20, vacate_prob=0.04):
    leases = []
    lease_id = 1
    tenant_id = 1

    # Track total floor space for each floor level
    floor_space = {floor: environment.max_floor_sf for floor in range(1, 31)}  # 1 to 30 floors

    for year in range(start_year, start_year + num_years):
        # Reset the building state at the start of each year
        environment.reset()

        # Simulate vacating suites at the start of each year
        for suite in environment.suites:
            if suite.occupied and np.random.rand() < vacate_prob:
                floor_space[suite.floor_level] += suite.suite_sf  # Update vacant space
                suite.vacate()

        # Loop over the number of tenants for the current year
        for _ in range(environment.num_tenants):
            state = environment.state.copy()
            state = np.nan_to_num(state, nan=0.0)

            lease_start_year = year
            state_values = state.tolist()

            # Predict variables using the agent
            rent_amount, lease_length, expand = agent.act(state_values)

            # Find a vacant suite to lease
            leased_suite = None
            for suite in environment.suites:
                if not suite.occupied and suite.suite_sf <= floor_space[suite.floor_level]:
                    suite.lease(tenant_id, lease_id, lease_start_year, lease_start_year + lease_length)
                    leased_suite = suite
                    floor_space[suite.floor_level] -= suite.suite_sf  # Decrease vacant space
                    break

            if leased_suite is None:
                continue  # Skip if no suitable vacant suite is found

            # Calculate base rent based on floor level
            floor_level = state[6]
            base_rent = 12 + (1 if 9 <= floor_level <= 15 else 2 if floor_level > 15 else 0)

            # Adjust base rent for the current year
            base_rent += (year - start_year) * 0.5
            base_rent *= (1 + 0.03) ** (year - start_year)

            # Split the lease into annual records
            for lease_year in range(lease_start_year, lease_start_year + lease_length):
                if lease_year >= start_year + num_years:
                    break  # Stop if exceeding simulation period

                # Calculate annual rent
                annual_rent = base_rent * leased_suite.suite_sf

                # Track vacant space for each record
                current_vacant_space = floor_space[leased_suite.floor_level]

                # Store lease data
                leases.append({
                    'LeaseID': lease_id,
                    'TenantID': tenant_id,
                    'StartYear': lease_start_year,
                    'LeaseYear': lease_year,
                    'FloorLevel': leased_suite.floor_level,
                    'SuiteID': leased_suite.suite_id,
                    'SuiteSquareFootage': leased_suite.suite_sf,
                    'RentAmount': round(base_rent, 2),
                    'AnnualRent': round(annual_rent, 2),
                    'Occupied': True,
                    'VacantSpace': round(current_vacant_space, 2)  # Track vacant space
                })

                base_rent += 0.5  # Annual increase in base rent

            # Lease renewal or termination
            if lease_start_year + lease_length <= start_year + num_years:
                renew_decision = np.random.choice([True, False], p=[0.7, 0.3])
                if renew_decision:
                    lease_id += 1
                    leased_suite.lease(tenant_id, lease_id, lease_start_year + lease_length,
                                       lease_start_year + 2 * lease_length)
                else:
                    floor_space[leased_suite.floor_level] += leased_suite.suite_sf  # Increase vacant space
                    leased_suite.vacate()
                    tenant_id += 1
                    lease_id += 1

        # Add records for unoccupied suites at the end of each year
        for suite in environment.suites:
            if not suite.occupied:
                # Track vacant space for unoccupied suites
                current_vacant_space = floor_space[suite.floor_level]

                # Store vacant suite data
                leases.append({
                    'LeaseID': None,
                    'TenantID': None,
                    'StartYear': year,
                    'LeaseYear': year,
                    'FloorLevel': suite.floor_level,
                    'SuiteID': suite.suite_id,
                    'SuiteSquareFootage': suite.suite_sf,
                    'RentAmount': 0,
                    'AnnualRent': 0,
                    'Occupied': False,
                    'VacantSpace': round(current_vacant_space, 2)  # Track vacant space
                })

    leases_df = pd.DataFrame(leases)
    return leases_df
