# code for the agent 


class Agent:
    def __init__(self, env):
        self.env = env
        self.policy = [] 
        self.state = 0 # initial state
        self.episodic_reward = 0

    def set_policy(self,policy):
        #sets the policy the agent will use 
        self.policy = policy
    
    def play_hand(self):
        # runs a single hand of blackjack according to the agent's policy
        self.env.reset_hand()
        
        # make initial deal 
        self.env.stick_or_hit(hit=True)
        self.state = self.env.player_hand_value
        #keep hitting until policy says to stick or agent goes bust
        while self.policy[self.state-1] == 'hit' and self.env.player_hand_value < 21:

            self.env.stick_or_hit(hit=True)
            if None in self.env.player_hand:
                return (self.env.player_hand_value)**2
            
            self.state = self.env.player_hand_value
            print(f'state{self.state}')

            if self.env.player_hand_value > 21:
                return 0 
        
        #returns the value of the hand 
        return (self.env.player_hand_value)**2
    
    def run_episode_reward(self):
        # runs an episode and returns the total reward
        episodic_reward = 0

        self.episodic_reward = 0
        # if deck if infinite an episode is just one hand
        if self.env.dealer.playing_deck == 'infinite deck':
            return self.play_hand()
        else:
            while self.env.dealer.playing_deck_length > 0:
                episodic_reward += self.play_hand()
            return episodic_reward

            
        

    