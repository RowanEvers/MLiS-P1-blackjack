# code for the agent 
import random
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1,gamma=0.9,epsilon_end=0.01):
        self.env = env
        self.state = 0 # initial state
        self.Q_values = {}
        for state in range(2, 22):
            self.Q_values[(state, 'hit')] = 0.0
            self.Q_values[(state, 'stick')] = 0.0

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon_start = 1 # explortation rate 
        self.epsilon_end = epsilon_end
        self.epsilon = self.epsilon_start

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
                
                if self.env.player_hand_value <= 21:  # player hits without going bust
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

    def learn(self, num_episodes, decay_epsilon=False,epsilon = 0.1):
        # runs multiple episodes to learn Q-values
        per_episode_rewards = []
        if not decay_epsilon:
            self.epsilon = epsilon
        for episode in range(num_episodes):
            if decay_epsilon:
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1*episode / (num_episodes / 10)) # decay epsilon 
            episode_reward = self.run_episode_reward()
            per_episode_rewards.append(episode_reward)

            if (episode+1) % 100 == 0:

                print(f'Episode {episode+1} completed ({(episode+1)/num_episodes*100:.2f}%). Moving Average Reward: {np.mean(per_episode_rewards[-100:])}')
        
        return per_episode_rewards
        






class ReinforceAgent:
    def __init__(self,env):
        self.env = env 
        self.weights = np.linspace(0.01, 0.01, 15) # weights for each state feature, including bias
        self.state = np.zeros(15)

        # vectors for ADAM optimizer
        self.m = np.zeros(15)
        self.v = np.zeros(15)

    def sigmoid(self,x):
        x = np.clip(x, -20, 20) # to prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def parameterised_policy(self,state):
        '''
        Given a state, returns the probability of hitting using a parameterised policy
        '''
        z = np.dot(self.weights, state)
        prob_hit = self.sigmoid(z)
        return prob_hit
    
    def select_action(self,state):
        '''
        Selects an action based on the parameterised policy
        '''
        prob_hit = self.parameterised_policy(state)
        if random.uniform(0,1) < prob_hit:
            return 'hit'
        else:
            return 'stick'
    
    def update_state(self,player_hand,player_hand_value):
        self.state[0] = player_hand_value/21 # player hand value
        # update card counts
        if None in player_hand:
            player_hand.remove(None)

            
        self.state[player_hand[-1][1]] += 1
        self.state[14] = 1  # bias term
    
    def play_hand_reinforce(self):
        # runs a single hand of blackjack using REINFORCE to update the policy weight vector
        self.env.reset_hand()
        

        # make initial deal 
        self.env.stick_or_hit(hit=True)
        self.update_state(self.env.player_hand, self.env.player_hand_value)
        hand_trajectory = []  # to store (state, action, reward) tuples

        while True:
            current_state = self.state.copy()
            action = self.select_action(self.state)
            
            if action == 'stick': # (terminal state)
                reward = (self.env.player_hand_value)**2
                hand_trajectory.append((current_state, action, reward,True))
                #self.update_state(self.env.player_hand, self.env.player_hand_value)
                return hand_trajectory
            
            elif action == 'hit':
                self.env.stick_or_hit(hit=True)
                
                if None in self.env.player_hand:   # None in player hand means no cards left in deck (equivalent to sticking) (terminal state)
                    reward = (self.env.player_hand_value)**2
                    hand_trajectory.append((current_state, action, reward,True))
                    self.update_state(self.env.player_hand, self.env.player_hand_value)
                    return hand_trajectory
                
                if self.env.player_hand_value <= 21:  # player hits without going bust
                    hand_trajectory.append((current_state, action, 0,False))
                    self.update_state(self.env.player_hand, self.env.player_hand_value)
                    reward = 0 # no reward unil the end of the hand since premptively rewarding a hit can lead to suboptimal policies

                if self.env.player_hand_value > 21:  # player goes bust (terminal state)
                    reward = 0
                    hand_trajectory.append((current_state, action, reward,True))
                    self.update_state(self.env.player_hand, self.env.player_hand_value)
                    return hand_trajectory 
                
                
    
    def run_episode(self):
        episodic_trajectory = []
        self.env.reset_deck()
        self.state = np.linspace(0, 0, 15)
        while self.env.dealer.playing_deck_length > 0:
            episodic_trajectory += self.play_hand_reinforce()
        return episodic_trajectory
    
        
    def train_agent(self,num_episodes, learning_rate=0.001, gamma = 1,ADAM=False):
        
        episode_rewards_avg_per_hand = []

        for episode in range(num_episodes):
            number_of_hands = 0
            episode_trajectory = self.run_episode()
            hand_rewards = [item[-2] for item in episode_trajectory]
            episode_reward = sum(hand_rewards)
            Gt_list = []
            G = 0 

            for i, r in reversed(list(enumerate(hand_rewards))):
                
                if episode_trajectory[i][-1]:  # if terminal state
                    number_of_hands += 1
                    G = 0

                G = r + gamma * G
                Gt_list.insert(0, G) # calculate return from time t


            assert len(Gt_list) == len(episode_trajectory) # ensure Gt_list is the correct length 

            Gt_array = np.array(Gt_list)
            # standardise returns for variance reduction
            Gt_array = (Gt_array - np.mean(Gt_array)) / (np.std(Gt_array) + 1e-8)

            # update weights using policy gradient
            for t in range(len(episode_trajectory)):
                state = episode_trajectory[t][0]
                action = episode_trajectory[t][1]
                Gt = Gt_array[t]

                # get the probability of a hit under the current policy 
                prob_hit = self.parameterised_policy(state)

                h = 1 if action == 'hit' else 0

                #calculate gradient of log policy
                ''' 
                For a Sigmoid policy, the gradient of log-probability is:
                 (Action_Taken - Prob_Hit) * State 
                 '''
                grad_log_policy = (h - prob_hit) * state

                # update weights
                if not ADAM:
                    self.weights += learning_rate * gamma**t * Gt * grad_log_policy

                # ADAM update 
                elif ADAM:
                    # Adam parameters
                    beta1 = 0.9
                    beta2 = 0.999
                    epsilon = 1e-8

                    gt = gamma**t * Gt * grad_log_policy

                    self.m = beta1 * self.m + (1 - beta1) * gt
                    self.v = beta2 * self.v + (1 - beta2) * (gt ** 2)
                    m_hat = self.m / (1 - beta1 ** (t + 1))
                    v_hat = self.v / (1 - beta2 ** (t + 1))
                    self.weights += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    


            episode_reward_avg_per_hand = episode_reward / number_of_hands
            episode_rewards_avg_per_hand.append(episode_reward_avg_per_hand)
            if (episode+1) % 100 == 0:
                print(f'Episode {episode+1} completed ({(episode+1)/num_episodes*100:.2f}%). Average reward per hand: {episode_reward_avg_per_hand}')
        return episode_rewards_avg_per_hand

        