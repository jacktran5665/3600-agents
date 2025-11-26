import numpy as np
import time
import traceback
from typing import List, Tuple, Callable

from game.enums import Direction, MoveType
from game.game_map import prob_hear, prob_feel

"""
D5: "The Thief"
Improvement over D4:
1. Smarter Evaluation: Calculates "Contested Eggs".
   - If an egg is reachable by the enemy, it is worth DOUBLE points to take it.
2. Turd Discipline: Only drops turds if enemy is within 2 steps.
3. Beam Search: Keeps looking 3 steps ahead.
"""

class PlayerAgent:
    def __init__(self, board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.trapdoor_probs = [np.zeros((self.map_size, self.map_size)), 
                               np.zeros((self.map_size, self.map_size))]
        self.initialize_priors()

    def initialize_priors(self):
        for x in range(self.map_size):
            for y in range(self.map_size):
                dist_x = min(x, self.map_size - 1 - x)
                dist_y = min(y, self.map_size - 1 - y)
                dist_edge = min(dist_x, dist_y)
                weight = 0.0
                if dist_edge == 1: weight = 1.0
                elif dist_edge >= 2: weight = 2.0
                parity = (x + y) % 2
                self.trapdoor_probs[parity][x, y] = weight
        for i in range(2):
            if np.sum(self.trapdoor_probs[i]) > 0:
                self.trapdoor_probs[i] /= np.sum(self.trapdoor_probs[i])

    def update_beliefs(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        cx, cy = current_loc
        for t_idx in range(2):
            heard, felt = sensor_data[t_idx]
            likelihood = np.zeros((self.map_size, self.map_size))
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x + y) % 2 != t_idx: continue
                    dx = abs(x - cx); dy = abs(y - cy)
                    p_h = prob_hear(dx, dy)
                    p_hear_cond = p_h if heard else (1.0 - p_h)
                    p_f = prob_feel(dx, dy)
                    p_feel_cond = p_f if felt else (1.0 - p_f)
                    likelihood[x, y] = p_hear_cond * p_feel_cond
            self.trapdoor_probs[t_idx] *= likelihood
            if np.sum(self.trapdoor_probs[t_idx]) > 0:
                self.trapdoor_probs[t_idx] /= np.sum(self.trapdoor_probs[t_idx])

    def get_trapdoor_risk(self, loc: Tuple[int, int]) -> float:
        return self.trapdoor_probs[(loc[0] + loc[1]) % 2][loc[0], loc[1]]

    def manhattan_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def evaluate_node(self, board, my_start_eggs) -> float:
        score = 0.0
        
        # 1. EGGS (Base Score)
        eggs_gained = board.chicken_player.get_eggs_laid() - my_start_eggs
        score += eggs_gained * 1000

        # 2. THE STEAL BONUS (New in D5)
        # Check if we are closer to the enemy than before.
        # If we take an egg that was close to the enemy, that's huge.
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        dist_to_enemy = self.manhattan_dist(my_loc, enemy_loc)
        
        # 3. SAFETY
        risk = self.get_trapdoor_risk(my_loc)
        if risk > 0.20: score -= 50000 
        elif risk > 0.10: score -= 2000

        # 4. MOBILITY (Don't get blocked)
        # Only check this if we haven't already crashed the simulation
        try:
            valid_moves = len(board.get_valid_moves())
            score += valid_moves * 10
            # If we are strangling the enemy (they have few moves), that's good too
            # (Requires reversing perspective briefly, costly but maybe worth it. Skipped for speed)
        except: pass

        return score

    def play(self, board, sensor_data, time_left):
        try:
            start_time = time.perf_counter()
            self.update_beliefs(board.chicken_player.get_location(), sensor_data)
            
            valid_moves = board.get_valid_moves()
            if not valid_moves: return (Direction.UP, MoveType.PLAIN)


            enemy_loc = board.chicken_enemy.get_location()
            my_loc = board.chicken_player.get_location()
            dist_to_enemy = self.manhattan_dist(my_loc, enemy_loc)
            
            # Filter out turd moves unless enemy is close
            if dist_to_enemy > 2:
                valid_moves = [m for m in valid_moves if m[1] != MoveType.TURD]
                if not valid_moves: # If we filtered everything (unlikely), reset
                     valid_moves = board.get_valid_moves()

            BEAM_WIDTH = 4  
            DEPTH = 3 
            if time_left() < 5: DEPTH = 1

            start_eggs = board.chicken_player.get_eggs_laid()
            
            # (Root Move, Board State, Score)
            beam = []
            
            # Depth 1 Expansion
            for move in valid_moves:
                next_board = board.forecast_move(move[0], move[1])
                if next_board:
                    # HEURISTIC TWEAK: 
                    # If this move takes an egg near the enemy, boost score artificially
                    bonus = 0
                    if move[1] == MoveType.EGG and dist_to_enemy <= 3:
                        bonus = 500 # The "Steal" incentive
                    
                    score = self.evaluate_node(next_board, start_eggs) + bonus
                    beam.append( (move, next_board, score) )

            beam.sort(key=lambda x: x[2], reverse=True)
            beam = beam[:BEAM_WIDTH]

            # Depth 2..DEPTH Expansion
            for d in range(1, DEPTH):
                new_beam = []
                for root_move, state_board, prev_score in beam:
                    if time.perf_counter() - start_time > 1.3: break
                    
                    poss_moves = state_board.get_valid_moves()
                    # Optimize: Check Eggs first
                    poss_moves.sort(key=lambda m: 0 if m[1] == MoveType.EGG else 2)

                    for m in poss_moves:
                        future = state_board.forecast_move(m[0], m[1])
                        if future:
                            score = self.evaluate_node(future, start_eggs)
                            # Time decay: Eggs now > Eggs later
                            score -= d * 20
                            new_beam.append( (root_move, future, score) )
                
                if not new_beam: break
                new_beam.sort(key=lambda x: x[2], reverse=True)
                beam = new_beam[:BEAM_WIDTH]

            return beam[0][0]

        except Exception as e:
            moves = board.get_valid_moves()
            if moves: return moves[0]
            return (Direction.UP, MoveType.PLAIN)