# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import numpy as np
import pandas as pd
import itertools as it
from statistics import mode


# I could have implemented the solution inside a function but found
# it more efficient to quickly write a class for easier storage of states.

# I apply Q-learning, since it was part of the course.

# This solution integrates a prediction of Abbey's behaviour. It can beat Abbey
# over 70% of the time. However, be aware of the following:

# A problem comes with Abbey when played over several matches: Abbey will store all plays
# that my player has played against her during *all* (including past) matches. But my player will re-init
# its internal history with every new match/adversary. Hence, my player's predictions of Abbey's
# behaviour will be off from the second match onwards. Above's win rate will only
# be achieved from a fresh Python session when the first match is against Abbey and no other
# match against Abbey follows this initial match.

# ==> When running from the command line, you should be fine.

# See RPS_1.py for a simpler solution with lower performance.


class Player:
    def __init__(
        self,
        learning_rate=0.5,
        gamma=0.5,
        exploration_period=75,
        reexploration_period=50,
        min_observation_period=75,
        random_seed=10,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.opponent_history = []
        self.own_history = []
        self.reward_history = np.array(
            (), dtype=int
        )  # I'll make use of the numpy functionality here.
        self.action_space = ["R", "P", "S"]
        # The state space is the 5-fold cartesian product of the action space
        self.state_space = list(
            it.product(
                self.action_space,
                self.action_space,
                self.action_space,
                self.action_space,
                self.action_space,
            )
        )
        self.state_space = [i[0] + i[1] + i[2] + i[3] + i[4] for i in self.state_space]
        self.Q = pd.DataFrame(
            np.zeros((3**5, 3)), index=self.state_space, columns=self.action_space
        )
        self.epsilon = 1.0
        self.exploration_period = exploration_period
        self.epsilon_diminish = round(self.epsilon / self.exploration_period, 10)
        self.reexploration_period = reexploration_period
        if self.reexploration_period > 0:
            self.epsilon_reexploration_diminish = round(
                self.epsilon / self.reexploration_period, 10
            )
            self.min_observation_period = min_observation_period
            self.n_reexploration_entered = 0
            self.plays_since_end_of_exploration = 0
        self.abbeys_next_plays = []
        self.abbeys_play_order = {
            "RR": 0,
            "RP": 0,
            "RS": 0,
            "PR": 0,
            "PP": 0,
            "PS": 0,
            "SR": 0,
            "SP": 0,
            "SS": 0,
        }
        self.update_cycles = 0
        np.random.seed(random_seed)

    # Outwit Abbey by including a prediction
    # of her nex move into the Q table.
    def update_abbeys_next_plays(self):
        # Abbey asumes that I played "R" before the match starts
        if len(self.own_history) == 0:
            own_prev_play = "R"
        else:
            own_prev_play = self.own_history[-1]

        last_two = "".join(self.own_history[-2:])
        if len(last_two) == 2:
            self.abbeys_play_order[last_two] += 1

        potential_plays = [
            own_prev_play + "R",
            own_prev_play + "P",
            own_prev_play + "S",
        ]

        sub_order = {
            k: self.abbeys_play_order[k]
            for k in potential_plays
            if k in self.abbeys_play_order
        }

        prediction = max(sub_order, key=sub_order.get)[-1:]

        ideal_response = {"P": "S", "R": "P", "S": "R"}

        self.abbeys_next_plays.append(ideal_response[prediction])

    def update_Q(self):
        # Updates Q based on the current values stored in the play histories
        state = (
            self.opponent_history[-3]
            + self.opponent_history[-2]
            + self.own_history[-2]
            + mode(self.own_history[-11:-1])
            + self.abbeys_next_plays[-2]
        )
        action = self.own_history[-1]
        new_state = (
            self.opponent_history[-2]
            + self.opponent_history[-1]
            + self.own_history[-1]
            + mode(self.own_history[-10:])
            + self.abbeys_next_plays[-1]
        )

        own_prev_play = self.own_history[-1]
        opp_prev_play = self.opponent_history[-1]

        # Get the reward for last play outcome
        # Tie: 0
        if opp_prev_play == own_prev_play:
            reward = 0
        # Win: 1
        elif (
            sum(
                (own_prev_play == np.array(["R", "P", "S"]))
                & (opp_prev_play == np.array(["S", "R", "P"]))
            )
            == 1
        ):
            reward = 1
        # Lose: -1
        else:
            reward = -1

        self.reward_history = np.append(self.reward_history, reward)

        # Now update the Q table based on reward, history and parameters
        update = self.learning_rate * (
            reward
            + self.gamma * np.max(self.Q.loc[new_state, :])
            - self.Q.loc[state, action]
        )

        self.Q.loc[state, action] += update

        self.update_cycles += 1

    def play(self, opp_prev_play):
        # own_history is still one entry ahead of opponent_history
        # Hence, firstly update:
        if not opp_prev_play == "":
            self.opponent_history.append(opp_prev_play)

        # Update Abbey's next plays before making the own choice
        self.update_abbeys_next_plays()

        # As long as history still not long enough for learning, play at random
        if len(self.opponent_history) < 3:
            own_play = np.random.choice(["R", "P", "S"])
            self.own_history.append(own_play)
            return own_play  # leave play function here

        # Update Q based on new information
        self.update_Q()

        state = (
            self.opponent_history[-2]
            + self.opponent_history[-1]
            + self.own_history[-1]
            + mode(self.own_history[-10:])
            + self.abbeys_next_plays[-1]
        )

        # Given state, decide what to play now
        if self.epsilon > 0.0 and np.random.uniform(0, 1) < self.epsilon:
            own_play = np.random.choice(["R", "P", "S"])
        elif (
            self.Q.loc[state, :].nunique() == 1
        ):  # Hence, if there is no max value in Q given state
            own_play = np.random.choice(["R", "P", "S"])
        else:
            own_play = self.Q.loc[state, :].idxmax()

        # Set own_history again one entry ahead of opponent_history:
        self.own_history.append(own_play)

        # Diminish exploration rate
        if self.epsilon > 0.0:
            self.epsilon -= self.epsilon_diminish
        elif self.reexploration_period > 0:
            self.plays_since_end_of_exploration += 1

        # Restart exploration if performance is/drops below 60% after exploration complete
        if (
            self.reexploration_period > 0
            and self.epsilon <= 0
            and self.plays_since_end_of_exploration >= self.min_observation_period
        ):
            # Track the strategy performance
            avg_number_of_wins = np.mean(
                self.reward_history[-self.plays_since_end_of_exploration :] == 1
            )
            if avg_number_of_wins < 0.6:
                self.epsilon = 1.0
                self.epsilon_diminish = self.epsilon_reexploration_diminish
                self.plays_since_end_of_exploration = 0
                self.n_reexploration_entered += 1

        return own_play


def player(prev_play, opponent_history=[]):
    global my_player

    if prev_play == "":
        my_player = Player()

    return my_player.play(prev_play)
