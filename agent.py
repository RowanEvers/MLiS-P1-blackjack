# code for the agent 
import random
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1,gamma=0.9,epsilon=0.1):
        self.env = env
        self.policy = [] 
        self.state = 0 # initial state
        self.Q_values = {}
        for state in range(2, 22):
            self.Q_values[(state, 'hit')] = 0.0
            self.Q_values[(state, 'stick')] = 0.0
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = 0.1 # explortation rate 

    def Q(self,state,action):
        # returns the Q value from the Q table given a state and action, if not present returns 0
        if (state,action) in self.Q_values:
            return self.Q_values[(state,action)]
        else:
            return 0
    
    def epsilon_greedy_action(self,state,epsilon):
        # selects an action using epsilon greedy policy
        if random.uniform(0,1) < epsilon:
            # explore: select a random action
            return random.choice(['hit','stick'])
        else:
            # exploit: select the action with highest Q value
            q_hit = self.Q(state,'hit')
            q_stick = self.Q(state,'stick')
            if q_hit > q_stick:
                return 'hit'
            else:
                return 'stick'

    def set_policy(self,policy):
        #sets the policy the agent will use 
        self.policy = policy
    
    def play_hand_q_learning(self):
        # runs a single hand of blackjack (episode) using Q-learning to update Q values
        self.env.reset_hand()

        # make initial deal 
        self.env.stick_or_hit(hit=True)
        self.state = self.env.player_hand_value
        #keep hitting until policy says to stick or agent goes bust

        while True:
            action = self.epsilon_greedy_action(self.state,self.epsilon)
            current_state = self.state 

            if action == 'stick': # (terminal state)
                reward = (self.env.player_hand_value)**2
                self.update_q_values(current_state, action, reward, None, True)
                return reward
            
            elif action == 'hit':
                self.env.stick_or_hit(hit=True)
                next_state = self.env.player_hand_value

                if None in self.env.player_hand:   # None in player hand means no cards left in deck (equivalent to sticking) (terminal state)
                    reward = (self.env.player_hand_value)**2
                    self.update_q_values(current_state, action, reward, None, True)
                    return reward
                
                if self.env.player_hand_value <= 21:  # player sticks without going bust
                    reward = 0 # no reward unil the end of the hand since premptively rewarding a hit can lead to suboptimal policies
                    self.update_q_values(current_state, action, reward, next_state, False)

                self.state = self.env.player_hand_value

                if self.env.player_hand_value > 21:  # player goes bust (terminal state)
                    reward = 0
                    self.update_q_values(current_state, action, reward, None, True)
                    return 0 


    def update_q_values(self, state, action, reward, next_state, is_terminal):
        current_q = self.Q(state, action)

        # bootstrap from the value of next state
        if is_terminal:
            max_next_q = 0
        else:
            # find max Q value for next state from the possible actions 
            next_q_hit = self.Q(next_state, 'hit')
            next_q_stick = self.Q(next_state, 'stick')
            max_next_q = max(next_q_hit, next_q_stick)
        # Q-learning update rule
        # Q(S,A) <- (1-alpha)*Q(S,A) + alpha*[R + gamma*maxQ(S',a)]
        new_q = (1-self.alpha)*current_q + self.alpha*(reward + self.gamma*max_next_q)
        self.Q_values[(state, action)] = new_q
        

    def run_episode_reward(self):
        # runs an episode and returns the total reward
        episodic_reward = 0
        # if deck if infinite an episode is just one hand
        if self.env.dealer.playing_deck == 'infinite deck':
            return self.play_hand_q_learning()
        
        else:
            while self.env.dealer.playing_deck_length > 0:
                episodic_reward += self.play_hand()
            return episodic_reward

    def learn(self, num_episodes):
        # runs multiple episodes to learn Q-values
        per_episode_rewards = []
        for episode in range(num_episodes):
            episode_reward = self.run_episode_reward()
            per_episode_rewards.append(episode_reward)
            print(f'Episode {episode+1} completed. Average reward: {np.mean(per_episode_rewards)}')
        
        return np.mean(per_episode_rewards)
        

    