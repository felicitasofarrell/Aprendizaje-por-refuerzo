import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agentes import Agent

class Connect4State:
    def __init__(self, rows: int = 6, cols: int = 7, board: Optional[np.ndarray] = None, current_player: int = 1):
        """
        Estado de Connect4.
        - board: matriz (rows x cols) con {0,1,2}
        - current_player: 1 o 2 (quien mueve ahora)
        """
        self.rows = rows
        self.cols = cols
        self.board: np.ndarray = board.copy() if board is not None else utils.create_board(rows, cols)
        self.current_player: int = current_player

    def copy(self):
        return Connect4State(self.rows, self.cols, self.board.copy(), self.current_player)

    def update_state(self, col_played: int):
        """Inserta ficha y cambia el jugador actual."""
        utils.insert_token(self.board, col_played, self.current_player)
        self.current_player = 2 if self.current_player == 1 else 1

    def __eq__(self, other):
        if not isinstance(other, Connect4State):
            return False
        return self.current_player == other.current_player and np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash((self.current_player, self.board.tobytes()))

    def __repr__(self):
        return f"Connect4State(player={self.current_player}, board=\n{self.board}\n)"


class Connect4Environment:
    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.state: Connect4State = Connect4State(rows, cols)

    def reset(self) -> Connect4State:
        self.state = Connect4State(self.rows, self.cols)
        return self.state.copy()

    def available_actions(self) -> List[int]:
        """Columnas donde la celda de arriba está libre (top == 0). 
        Si la primer celda de la columna esta libre, significa que se puede jugar esta columna"""
        return [c for c in range(self.cols) if self.state.board[0, c] == 0]

    def step(self, action: int) -> Tuple[Connect4State, float, bool, dict]:
        """
        Ejecuta la acción (columna). Devuelve:
        next_state, reward, done, info
        La reward está dada desde la perspectiva del jugador que realizó la jugada.
        """
        assert action in self.available_actions(), f"Acción inválida: col={action}"

        moving_player = self.state.current_player

        self.state.update_state(action)

        done, winner = utils.check_game_over(self.state.board)

        if done:
            if winner is None:
                reward = 0.5  
            elif winner == moving_player:
                reward = 1.0
            else:
                reward = -1.0 
        else:
            reward = 0.0

        info = {"winner": 0 if winner is None and done else (winner if winner is not None else None)}
        return self.state.copy(), reward, done, info

    def render(self):
        print("\n  " + " ".join(str(c) for c in range(self.cols)))
        print(" +" + "--" * self.cols + "+")
        for r in range(self.rows):
            fila = " ".join(str(int(x)) for x in self.state.board[r, :])
            print(f"{r}| {fila} |")
        print(" +" + "--" * self.cols + "+")
        print(f"Turno del jugador: {self.state.current_player}\n")


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        MLP simple. Input = features del estado; Output = Q por acción (cols).
        """
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepQLearningAgent:
    def __init__(self, state_shape: Tuple[int, int], n_actions: int, device: torch.device, *, gamma: float = 0.99, lr: float = 1e-3, batch_size: int = 128, target_update_every: int = 100, epsilon_decay: float = 0.995, epsilon: float = 1.0, epsilon_min: float = 0.1, memory_size: int = 10000):
        self.rows, self.cols = state_shape
        self.n_actions = n_actions
        self.device = device
        self.input_dim = 2 * self.rows * self.cols + 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.q_network = DQN(self.input_dim, n_actions).to(self.device)
        self.target_network = DQN(self.input_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-6)
        self.train_steps = 0

    def _encode_state(self, state: Connect4State) -> np.ndarray:
        """
        Devuelve un vector (input_dim,) con:
        - plano jugador1 (one-hot por casilla==1)
        - plano jugador2 (one-hot por casilla==2)
        - flag turno (1 si current_player == 1, 0 si 2)
        """
        b = state.board
        p1 = (b == 1).astype(np.float32).reshape(-1)
        p2 = (b == 2).astype(np.float32).reshape(-1)
        turn = np.array([1.0 if state.current_player == 1 else 0.0], dtype=np.float32)
        return np.concatenate([p1, p2, turn], axis=0)

    def preprocess(self, state: Connect4State) -> torch.Tensor:
        arr = self._encode_state(state)
        t = torch.from_numpy(arr).float().to(self.device)
        return t.unsqueeze(0)

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q_values = self.q_network(self.preprocess(state)).squeeze(0)
            mask = torch.full_like(q_values, float("-inf"))
            mask[valid_actions] = 0.0
            return int(torch.argmax(q_values + mask).item())

    def store_transition(self, s: Connect4State, a: int, r: float, s_next: Connect4State, done: bool):
        self.memory.append((s.copy(), a, float(r), s_next.copy(), bool(done)))

    def _valid_actions_from_state(self, st: Connect4State) -> list[int]:
        top = st.board[0, :]
        return [c for c in range(self.n_actions) if top[c] == 0]
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = torch.cat([self.preprocess(st) for st in states], dim=0)          
        ns = torch.cat([self.preprocess(st) for st in next_states], dim=0)    
        a  = torch.tensor(actions, dtype=torch.long,    device=self.device)   
        r  = torch.tensor(rewards, dtype=torch.float32, device=self.device)    
        d  = torch.tensor(dones,   dtype=torch.bool,    device=self.device)  

        self.q_network.train()
        q_values = self.q_network(s)                                        
        q_sa = q_values.gather(1, a.view(-1,1)).squeeze(1)                   

        masks = []
        for st in next_states:
            va = [c for c in range(self.n_actions) if st.board[0, c] == 0]
            m = torch.full((self.n_actions,), float("-inf"), device=self.device)
            m[va] = 0.0
            masks.append(m)
        mask = torch.stack(masks, dim=0)                                      

        self.target_network.eval()
        with torch.no_grad():
            q_next = self.target_network(ns)                

            masks = []
            has_valid_list = []
            for st in next_states:
                va = [c for c in range(self.n_actions) if st.board[0, c] == 0]
                m = torch.full((self.n_actions,), float("-inf"), device=self.device)
                if len(va) > 0:
                    m[va] = 0.0
                    has_valid_list.append(True)
                else:
                    has_valid_list.append(False)
                masks.append(m)
            mask = torch.stack(masks, dim=0)                
            has_valid = torch.tensor(has_valid_list, dtype=torch.bool, device=self.device)

            masked_q_next = q_next + mask
            max_q_next = masked_q_next.max(dim=1).values    

            max_q_next = torch.where(has_valid, max_q_next, torch.zeros_like(max_q_next))

            y = torch.where(d, r, r + self.gamma * max_q_next)  

        loss = F.smooth_l1_loss(q_sa, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu', name: str = "Trained"):
        self.rows, self.cols = state_shape
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.name = name

        input_dim = 2 * self.rows * self.cols + 1
        self.model = DQN(input_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _encode_state(self, state: Connect4State) -> torch.Tensor:
        b = state.board
        p1 = (b == 1).astype(np.float32).reshape(-1)
        p2 = (b == 2).astype(np.float32).reshape(-1)
        turn = np.array([1.0 if state.current_player == 1 else 0.0], dtype=np.float32)
        arr = np.concatenate([p1, p2, turn], axis=0)
        t = torch.from_numpy(arr).float().to(self.device)
        return t.unsqueeze(0)

    def play(self, state, valid_actions):
        with torch.no_grad():
            q = self.model(self._encode_state(state)).squeeze(0)
            mask = torch.full_like(q, float("-inf"))
            mask[valid_actions] = 0.0
            action = int(torch.argmax(q + mask).item())
        return action
