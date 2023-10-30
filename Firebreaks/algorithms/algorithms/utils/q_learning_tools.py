from cmath import inf
from distutils.command.sdist import sdist
from termios import N_MOUSE
import torch
import random
import itertools
import numpy as np
class Q_Table:
    def __init__(self, size, alpha = 1e-4, gamma = 0.99, n_envs = 8, epsilon = 0.1):
        self.device = torch.device('cpu')
        self.size = size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_envs = n_envs
        self.n_states = 0
        self.q_table = {}
        self.n_steps = int((self.size**2)*0.05)
        self.action_state = {}
        self.n_states_step = []
        self.create_table()

    def create_table(self):
        forbidden = int((self.size**2)*0.05)//2
        # Constuimos los estados por cada paso
        for i in range(self.n_steps + 1):
            # Calculamos las posibles combinaciones en que puede estar el estado
            combinations = list(itertools.combinations((j for j in range(forbidden, self.size**2)), i))
            self.n_states_step.append(len(combinations))
            # Por cada combinacion disponible
            for j in range(len(combinations)):
                # Agregamos una combinación (paso, combinación)
                self.action_state[(i,j)] = []
                # Rellenamos un tensor con '0's'
                state = torch.full((self.size, self.size), 0.0198).to(self.device)
                # Por cada combinación, marcamos el cortafuego asociado a la acción con un 1.
                for c in combinations[j]:
                    l = c // self.size 
                    m = c % self.size
                    state[l,m] = 1.
                # Calculamos las posibles combinaciones correspondientes al próximo estado
                next_state_comb = list(set(i for i in range(forbidden, self.size**2)) - set(combinations[j]))
                for action in next_state_comb:
                    self.q_table[(i, j, action)] = [0, state]
                    self.action_state[(i,j)].append(action)
                    self.n_states += 1      
    def find_state_indiv(self, state, step):
        found = False
        # Buscamos el número del estado asociado
        for i in range(int(self.n_states_step[step])):
            # Si son iguales, lo encontramos
            if torch.allclose(self.q_table[(step, i, self.action_state[(step,i)][0])][1], state.squeeze(0), atol = 1e-4):
                found = True
                break
        if not found:
            print(step)
            print(state)
            print("Error, state not found")
            raise("Error, state not found")
        # Retornamos el número del estado
        return i
    
    def find_state(self, state, step):
        # Ejecutamos la función find_state para cada agente en "paralelo"
        n_states = []
        for i in range(self.n_envs):
            n_states.append(self.find_state_indiv(state[i], step))
        # Retornamos una lista con los números de estados
        return n_states
    
    def pick_greedy_action_indiv(self, n_state, step):
        # Si epsilon > random.uniform, escogemos una acción aleatoria
        if self.epsilon > random.uniform(0, 1):
                action = random.choice(self.action_state[(step, n_state)])
                if not isinstance(action, int):
                    print(action)
                    print("action not integer!")
                    raise("action not integer!")
                # Retornamos la acción
                return action
        # Sino
        else:
            max_a_value = -inf
            max_action = None
            # Revisamos los valores en q_table para ese step y n_state
            for a in self.action_state[(step, n_state)]:
                # Si es mayor, actualizamos max_action
                if self.q_table[(step, n_state, a)][0] >= max_a_value:
                    max_a_value = self.q_table[(step, n_state, a)][0]
                    max_action = a
            if not isinstance(max_action, int):
                print(max_action)
                print("action not integer!")
                raise("action not integer!")
            if max_action == None:
                print("max action not found!")
                raise("max action not found!")
            # Retornamos max_action
            return max_action
        
    def pick_greedy_action(self, n_states, step):
        # Ejecutamos pick_greedy_action para cada uno de los estados
        actions = []
        for n in n_states:
            actions.append(self.pick_greedy_action_indiv(n, step))
        # Retornamos un stack de tensores con las acciones
        return torch.Tensor(actions).to(self.device)

    def update_table_indiv(self, n_state, step, action, next_state, reward, done):
        # Actualizamos el valor de la tabla
        action = int(action)
        # Si no se ha acabado el episodio
        if not done:
            # Buscamos el numero del próximo estado
            n_next_state = self.find_state_indiv(next_state, step + 1)
            max_a_value = -inf
            max_action = None
            # Buscamos la acción que maximiza el valor para el proximo estado
            for a in self.action_state[(step + 1, n_next_state)]:
                if self.q_table[(step + 1, n_next_state, a)][0] >= max_a_value:
                    max_a_value = self.q_table[(step + 1, n_next_state, a)][0]
                    max_action = a
            if not isinstance(max_action, int):
                print(max_action)
                print("action not integer!")
                raise("action not integer!")
            if not isinstance(action, int):
                print(action)
                print("action not integer!")
                raise("action not integer!")
            if max_action == None:
                print("max action not found!")
                raise("max action not found!")
            # Actualizamos según la regla de Q-Learning
            self.q_table[(step, n_state, action)][0] = self.q_table[(step, n_state, action)][0] + self.alpha*(reward + self.gamma*self.q_table[(step + 1, n_next_state, max_action)][0]-self.q_table[(step, n_state, action)][0])
        # Si se acabó el episodio
        else:
            # Actualizamos según Q(s,a) = alpha*r + (1-alpha)*Q(s,a)
            self.q_table[(step, n_state, action)][0] = self.alpha * reward + (1 - self.alpha) * self.q_table[(step, n_state, action)][0]
    def update_table(self, n_states, step, actions, next_states, rewards, done):
        # Ejecutamos update_table_indiv en cada ambiente
        for i in range(self.n_envs):
            self.update_table_indiv(n_states[i], step, actions[i].item(), next_states[i], rewards[i].item(), done)

    def max_action(self, state, step):
        # Buscamos la accion que maximiza Q en ese estado
        # Buscamos el numero del estado
        n_state = self.find_state_indiv(state, step)
        max_a_value = -inf
        max_action = None
        # Iteramos sobre las acciones disponibles buscando la que maximiza
        for a in self.action_state[(step, n_state)]:
            if self.q_table[(step, n_state,a)][0] >= max_a_value:
                max_a_value = self.q_table[(step, n_state,a)][0]
                max_action = a
        if max_action == None:
            print("max action not found!")
            raise("max action not found!")
        # Retornamos un tensor con la acción
        return torch.Tensor([max_action]).to(self.device)