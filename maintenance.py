import streamlit as st
import numpy as np

def birth_death_stationary_distribution(mu, gamma, warm_standby, n, m, k):
    # Total number of states: from 0 (all components up) to n (no components up)
    states = n + 1 
    birth_rates = np.zeros(states)
    death_rates = np.zeros(states)

    for i in range(states):  # i = number of failed components
        birth_rates[i] = min(i, k) * gamma

        # Number of working components
        working = n - i  
        if warm_standby:
            # In warm standby, all components can fail
            death_rates[i] = working * mu  
        else:
            # In cold standby, only the active components can fail 
            if working >= m:
                death_rates[i] = m * mu  
            else:
                death_rates[i] = 0  

    # Use birth-death formulas to compute stationary distribution
    pi = np.zeros(states)
    pi[0] = 1.0
    for i in range(1, states):
        if birth_rates[i] > 0:
            pi[i] = pi[i - 1] * death_rates[i - 1] / birth_rates[i]
        else:
            pi[i] = 0

    pi_sum = np.sum(pi)
    pi /= pi_sum
    return pi

def system_availability(mu, gamma, warm_standby, n, m, k):
    pi = birth_death_stationary_distribution(mu, gamma, warm_standby, n, m, k)

    # System is up in states n-m
    max_failed_allowed = n - m
    availability = np.sum(pi[:max_failed_allowed + 1])

    return availability

def optimize_components_and_repairmen(mu, gamma, warm_standby, m, component_cost, repairman_cost, downtime_cost, n_range, k_range):
    best_n, best_k = None, None
    min_cost = float('inf')

    for n in n_range:
        for k in k_range:
            if m > n:
                continue

            availability = system_availability(mu, gamma, warm_standby, n, m, k)
            cost = (component_cost * n) + (repairman_cost * k) + (downtime_cost * (1 - availability))

            if cost < min_cost:
                min_cost = cost
                best_n = n
                best_k = k

    return best_n, best_k, min_cost

def main():
    st.title("k-out-of-n System Maintenance Optimizer")

    mu = st.number_input("Failure rate (mu)", min_value=0.0, value=0.01)
    gamma = st.number_input("Repair rate (gamma)", min_value=0.0, value=0.1)
    standby_type = st.radio("Are unused components in warm standby?", ("Yes", "No"))
    warm_standby = standby_type == "Yes"

    n = st.number_input("Number of components (n)", min_value=1, value=5, step=1)
    m = st.number_input("Components required to function (m)", min_value=1, max_value=n, value=3, step=1)
    k = st.number_input("Number of repairmen (k)", min_value=1, value=2, step=1)

    component_cost = st.number_input("Cost per component", min_value=0.0, value=5.0)
    repairman_cost = st.number_input("Cost per repairman", min_value=0.0, value=10.0)
    downtime_cost = st.number_input("Downtime cost (per unit time)", min_value=0.0, value=1000.0)

    if st.button("Calculate Availability and Optimize"):
        availability = system_availability(mu, gamma, warm_standby, n, m, k)
        st.success(f"System Availability: {availability:.4f}")

        n_range = range(m, n + 100)
        k_range = range(1, k + 10)
        best_n, best_k, min_cost = optimize_components_and_repairmen(
            mu, gamma, warm_standby, m,
            component_cost, repairman_cost, downtime_cost,
            n_range, k_range
        )

        st.info(f"Optimal number of components (n): {best_n}")
        st.info(f"Optimal number of repairmen (k): {best_k}")
        st.info(f"Minimum total cost: {min_cost:.2f}")


if __name__ == "__main__":
    main()

