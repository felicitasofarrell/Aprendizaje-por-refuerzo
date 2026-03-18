from abc import ABC, abstractmethod
from random import choice
import utils
import numpy as np

class Agent(ABC):
    @abstractmethod
    def play(self, state, valid_actions):
        '''Dado un estado y acciones válidas, retorna una acción.'''
        pass

class RandomAgent(Agent):
    ''' Agente que juega siempre al azar. '''
    def __init__(self, name: str):
        self.name = name
    def play(self, state, valid_actions):
        return choice(valid_actions)

class HumanAgent(Agent):
    ''' Agente humano que interactúa por consola. '''
    def __init__(self, name: str):
        self.name = name
    def play(self, state, valid_actions):
        return int(input("Ingrese la columna donde desea jugar: "))

class DefenderAgent(Agent):
    ''' Agente que revisa si el oponente está por ganar e intenta bloquearlo;
        si no, juega al azar. '''
    def __init__(self, name: str):
        self.name = name
    def play(self, state, valid_actions):
        me:int = state.current_player
        opponent:int = 3 - me
        for col in valid_actions:
            new_board:np.ndarray = np.array(state.board, dtype=int)
            utils.insert_token(new_board, col, opponent)
            if utils.check_game_over(new_board)[0]:
                return col
        return choice(valid_actions)

class HeuristicAgent(Agent):
    '''Agente que bloquea si el rival está por ganar; si no, prioriza
       la columna central; si no, juega al azar.'''

    def __init__(self, name="Heuristic"):
        self.name = name

    def play(self, state, valid_actions):
        me = state.current_player
        opponent = 3 - me

        for col in valid_actions:
            b = np.array(state.board, dtype=int)
            utils.insert_token(b, col, opponent)
            done, winner = utils.check_game_over(b)
            if done and winner == opponent:
                return col

        center = state.cols // 2
        if center in valid_actions:
            return center

        return choice(valid_actions)

class Greedy2Agent(Agent):
    '''Agente que intenta ganar en su turno; si no puede, bloquea una
       victoria inminente del rival; si no, juega al azar.'''

    def __init__(self, name="Greedy"):
        self.name = name

    def play(self, state, valid_actions):
        me = state.current_player
        opponent = 3 - me

        for col in valid_actions:
            b = np.array(state.board, dtype=int)
            utils.insert_token(b, col, me)
            done, winner = utils.check_game_over(b)
            if done and winner == me:
                return col

        for col in valid_actions:
            b = np.array(state.board, dtype=int)
            utils.insert_token(b, col, opponent)
            done, winner = utils.check_game_over(b)
            if done and winner == opponent:
                return col

        return choice(valid_actions)
