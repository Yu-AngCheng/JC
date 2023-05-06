import numpy as np
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets  # interactive display
from scipy.stats import poisson


class JacksCarRental:
    def __init__(self, max_cars=20, max_move=5, request_rate=(3, 4), return_rate=(3, 2),
                 rental_income=10, move_car_cost=2, gamma=0.9):
        self.max_cars = max_cars
        self.max_move = max_move
        self.request_rate = request_rate
        self.return_rate = return_rate
        self.rental_income = rental_income
        self.move_car_cost = move_car_cost
        self.gamma = gamma
        self.compute_rental_and_return_p()
        self.transition_prob()

    # Pre-compute the probability of rental and return
    def compute_rental_and_return_p(self):
        # probability mass function: Pr(X = n)
        self.rental_pmf_loc1 = np.array([poisson.pmf(n, self.request_rate[0]) for n in range(self.max_cars + 1)])
        self.rental_pmf_loc2 = np.array([poisson.pmf(n, self.request_rate[1]) for n in range(self.max_cars + 1)])
        self.return_pmf_loc1 = np.array([poisson.pmf(n, self.return_rate[0]) for n in range(self.max_cars + 1)])
        self.return_pmf_loc2 = np.array([poisson.pmf(n, self.return_rate[1]) for n in range(self.max_cars + 1)])
        # survival function: P[n] = Pr(X > n-1) = Pr(X >= n)
        self.rental_sf_loc1 = np.array([poisson.sf(n - 1, self.request_rate[0]) for n in range(self.max_cars + 1)])
        self.rental_sf_loc2 = np.array([poisson.sf(n - 1, self.request_rate[1]) for n in range(self.max_cars + 1)])
        self.return_sf_loc1 = np.array([poisson.sf(n - 1, self.return_rate[0]) for n in range(self.max_cars + 1)])
        self.return_sf_loc2 = np.array([poisson.sf(n - 1, self.return_rate[1]) for n in range(self.max_cars + 1)])

    # Compute the rental reward
    # Since the rental reward is only based on the number of cars after moving, we can pre-compute the rental reward
    def transition_prob(self):
        self.state_transition_prob = np.zeros(
            (self.max_cars + 1, self.max_cars + 1, self.max_cars + 1, self.max_cars + 1))
        self.state_transition_reward = np.zeros(
            (self.max_cars + 1, self.max_cars + 1, self.max_cars + 1, self.max_cars + 1))

        for s in [(car_loc1, car_loc2) for car_loc1 in range(self.max_cars + 1) for car_loc2 in
                  range(self.max_cars + 1)]:
            cars_loc1, cars_loc2 = s
            for rent1 in range(cars_loc1 + 1):
                for rent2 in range(cars_loc2 + 1):
                    reward_rent = self.rental_income * (rent1 + rent2)
                    if rent1 < cars_loc1:
                        prob_rent1 = self.rental_pmf_loc1[rent1]
                    else:
                        prob_rent1 = self.rental_sf_loc1[cars_loc1]
                    if rent2 < cars_loc2:
                        prob_rent2 = self.rental_pmf_loc2[rent2]
                    else:
                        prob_rent2 = self.rental_sf_loc2[cars_loc2]
                    prob_rent = prob_rent1 * prob_rent2

                    remaining_cars_loc1 = cars_loc1 - rent1
                    remaining_cars_loc2 = cars_loc2 - rent2

                    for return1 in range(self.max_cars - remaining_cars_loc1 + 1):
                        for return2 in range(self.max_cars - remaining_cars_loc2 + 1):
                            if return1 < self.max_cars - remaining_cars_loc1:
                                prob_return1 = self.return_pmf_loc1[return1]
                            else:
                                prob_return1 = self.return_sf_loc1[self.max_cars - remaining_cars_loc1]
                            if return2 < self.max_cars - remaining_cars_loc2:
                                prob_return2 = self.return_pmf_loc2[return2]
                            else:
                                prob_return2 = self.return_sf_loc2[self.max_cars - remaining_cars_loc2]
                            prob_return = prob_return1 * prob_return2

                            new_cars_loc1 = remaining_cars_loc1 + return1
                            new_cars_loc2 = remaining_cars_loc2 + return2
                            new_s = (new_cars_loc1, new_cars_loc2)

                            prob = prob_rent * prob_return

                            self.state_transition_prob[cars_loc1, cars_loc2, new_cars_loc1, new_cars_loc2] = prob
                            self.state_transition_reward[
                                cars_loc1, cars_loc2, new_cars_loc1, new_cars_loc2] = reward_rent

    def value_update(self, s, a, v):
        cars_loc1, cars_loc2 = s

        cars_loc1 -= a
        cars_loc2 += a
        cars_loc1 = min(cars_loc1, self.max_cars)
        cars_loc2 = min(cars_loc2, self.max_cars)

        move_cost = self.move_car_cost * abs(a)

        return np.sum((self.state_transition_reward[cars_loc1, cars_loc2, :,
                       :] - move_cost + self.gamma * v) * self.state_transition_prob[cars_loc1, cars_loc2, :, :])


env = JacksCarRental()

values = np.zeros((env.max_cars + 1, env.max_cars + 1))
policy = np.zeros((env.max_cars + 1, env.max_cars + 1), dtype=np.int)
while True:
    # Policy evaluation
    while True:
        delta = 0
        for s in [(car_loc1, car_loc2) for car_loc1 in range(env.max_cars + 1) for car_loc2 in range(env.max_cars + 1)]:
            v = values[s]
            a = policy[s]
            values[s] = env.value_update(s, a, values)
            delta = max(delta, abs(v - values[s]))
        if delta < 1e-4:
            break

    # Policy improvement
    policy_stable = True
    for s in [(car_loc1, car_loc2) for car_loc1 in range(env.max_cars + 1) for car_loc2 in range(env.max_cars + 1)]:
        old_action = policy[s]
        state_values = []
        actions_valid = [a for a in range(max(-s[1], -env.max_move), min(s[0], env.max_move) + 1)]
        for a in actions_valid:
            state_value = env.value_update(s, a, values)
            state_values.append(state_value)
        new_action = actions_valid[np.argmax(state_values)]
        if old_action != new_action:
            policy_stable = False
        policy[s] = new_action

    if policy_stable:
        break
