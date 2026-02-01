import numpy as np
import random
class Dealer:
    def __init__(self):
        self.ordered_deck = []
        self._create_deck_ordered()

    def _create_deck_ordered(self):
        for i in range(52):
            if i < 13:
                suit = 'hearts'
                self.ordered_deck.append((suit, i % 13 + 1))
            elif i < 26:
                suit = 'diamonds'
                self.ordered_deck.append((suit, i % 13 + 1))    
            elif i < 39:
                suit = 'clubs'
                self.ordered_deck.append((suit, i % 13 + 1))    
            else:
                suit = 'spades'
                self.ordered_deck.append((suit, i % 13 + 1))


    def create_playing_deck(self,D):
        # shuffles D decks together 
        if D != np.inf:
        
            self.playing_deck = self.ordered_deck * D

            np.random.shuffle(self.playing_deck)
            self.playing_deck_length = len(self.playing_deck)
            
        elif D == np.inf:
            self.playing_deck = 'infinite deck'
            self.playing_deck_length = np.inf

        return self.playing_deck


    def deal_card(self):
        if self.playing_deck == 'infinite deck':
            card = random.choice(self.ordered_deck)
            return card
        
        elif len(self.playing_deck) > 0:
            card = self.playing_deck.pop()
            self.playing_deck_length -= 1 
            return card
        else:
            return None



class BlackjackEnv:
    def __init__(self, D=np.inf):
        self.dealer = Dealer()
        self.D = D
        self.playing_deck = self.dealer.create_playing_deck(D)
        self.player_hand = []

    def card_value(self,card):
        if card is None:
            return 0
        value = card[1]
        if value > 10:
            value = 10

        if value == 1:
            value = 11
        return value
    
    def stick_or_hit(self,hit):
        if hit:
            card = self.dealer.deal_card()
            self.player_hand.append(card)
    
        self.player_hand_value = np.sum(self.card_value(card) for card in self.player_hand)
        if self.player_hand_value > 21:
            # check for aces and convert them from 11 to 1
            for i in range(len(self.player_hand)):
                if self.player_hand[i][1] == 1:
                    self.player_hand_value -= 10
                    if self.player_hand_value <= 21:
                        break
        
    def reset_hand(self):
        self.player_hand = []
        self.player_hand_value = 0

    def reset_deck(self):
        self.playing_deck = self.dealer.create_playing_deck(self.D)