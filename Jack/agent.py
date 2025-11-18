from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple

import numpy as np
from game import *

"""
Melvin is the dumbest agent of all. He randomly selects a move from the list of valid moves.
"""


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        # lightweight RNG and tiny state for heuristics
        self.rng = np.random.RandomState()
        self.last_think = 0.0
        # remember previous location to avoid immediate back-and-forth
        self.prev_loc = None
    
    
    
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")

        # quick heuristic: prefer eggs, then plain moves that move away from center,
        # finally turds. Use sensor_data to bias away from center if we heard/felt.

        # local imports so we don't change the file-level imports the tournament expects
        from game.enums import MoveType, loc_after_direction

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        moves = board.get_valid_moves()
        if not moves:
            return None

        # Additional safety filter (re-implementing constraints):
        # - cannot move into enemy chicken, enemy eggs, enemy turds
        # - cannot move into a square that shares an edge with an enemy turd
        # - eggs/turds cannot be placed where an egg/turd already exists
        # - respect turd availability
        from game.enums import Direction

        filtered_no_own_eggs = []
        allowed_but_own_eggs = []
        for m in moves:
            d, mt = m
            dest = loc_after_direction(my_loc, d)

            # cannot move into enemy chicken
            if dest == enemy_loc:
                continue

            # cannot move into squares containing opponent eggs/turds
            # (but moving into your OWN egg/turd is allowed)
            if dest in board.eggs_enemy or dest in board.turds_enemy:
                continue

            # do NOT move into your own eggs (avoid bouncing back to egg squares)
            if dest in board.eggs_player:
                # keep this option as a fallback but prefer other moves
                allowed_but_own_eggs.append(m)
                continue

            # avoid immediate backtracking to previous location
            if self.prev_loc is not None and dest == self.prev_loc:
                # prefer other moves, but keep as fallback
                allowed_but_own_eggs.append(m)
                continue
            # moving into your own turd is allowed by the engine; do not block

            # cannot move into square that shares an edge with an enemy turd
            blocked_by_enemy_turd = False
            for dd in Direction:
                adj = loc_after_direction(dest, dd)
                if adj in board.turds_enemy:
                    blocked_by_enemy_turd = True
                    break
            if blocked_by_enemy_turd:
                continue

            # if laying egg, enforce parity and that current square is empty
            if mt == MoveType.EGG:
                if not board.can_lay_egg_at_loc(my_loc):
                    continue

            # if laying turd, enforce cannot be on existing egg/turd and not adjacent to enemy
            if mt == MoveType.TURD:
                if not board.can_lay_turd_at_loc(my_loc):
                    continue

            filtered_no_own_eggs.append(m)

        # choose moves: prefer those that are not own-egg destinations and not backtracking
        if filtered_no_own_eggs:
            moves = filtered_no_own_eggs
        elif allowed_but_own_eggs:
            # if no other moves, allow moves into own eggs or backtracking
            moves = allowed_but_own_eggs
        else:
            # no allowed moves under our stricter filter -> return None so engine handles
            return None

        # helper: compute euclidean distance from center
        def center_dist(loc):
            cx = (board.game_map.MAP_SIZE - 1) / 2.0
            cy = (board.game_map.MAP_SIZE - 1) / 2.0
            return ((loc[0] - cx) ** 2 + (loc[1] - cy) ** 2) ** 0.5

        # helper: get destination after moving in a direction
        def dest_for(move):
            d, _ = move
            return loc_after_direction(my_loc, d)

        # sensor_data: two tuples (white_trapdoor=(heard,felt), black_trapdoor=(heard,felt))
        heard_or_felt = False
        try:
            heard_or_felt = any(a or b for (a, b) in sensor_data)
        except Exception:
            heard_or_felt = False

        # Score a move: prefer greater distance from center; if sensor indicates
        # nearby trapdoor, bias stronger away from center.
        def diagonal_coverage(loc):
            # count diagonal neighbours where we could lay an egg in the future
            count = 0
            for dx in (-1, 1):
                for dy in (-1, 1):
                    cand = (loc[0] + dx, loc[1] + dy)
                    if not board.is_valid_cell(cand):
                        continue
                    try:
                        if board.can_lay_egg_at_loc(cand):
                            count += 1
                    except Exception:
                        # safe fallback
                        pass
            return count

        def score(move):
            dir, mv = move
            dest = dest_for(move)

            # base terms
            base = center_dist(dest)
            enemy_dist = ( (dest[0]-enemy_loc[0])**2 + (dest[1]-enemy_loc[1])**2 ) ** 0.5

            # diagonal coverage: for egg moves the egg is placed at my_loc;
            # for other moves we consider diagonal opportunities around dest.
            if mv == MoveType.EGG:
                cover = diagonal_coverage(my_loc)
            else:
                cover = diagonal_coverage(dest)

            # combine with weights: prioritize diagonal coverage, then distance from center/enemy
            score_val = cover * 5.0 + base * 0.5 + enemy_dist * 0.2

            if heard_or_felt:
                # amplify center avoidance when sensors trigger
                score_val += base * 0.3

            return score_val

        eggs = [m for m in moves if m[1] == MoveType.EGG]
        plains = [m for m in moves if m[1] == MoveType.PLAIN]
        turds = [m for m in moves if m[1] == MoveType.TURD]

        # Prefer eggs (corner eggs are auto-rewarded by engine). Choose egg that
        # results in maximal center distance after stepping out.
        if eggs:
            # pick egg that maximizes diagonal coverage first
            best = max(eggs, key=score)
            result = best
        elif plains:
            result = max(plains, key=score)
        elif turds:
            # for turds, prefer ones that increase distance from enemy
            def turd_score(m):
                dest = dest_for(m)
                return abs(dest[0] - enemy_loc[0]) + abs(dest[1] - enemy_loc[1])

            result = max(turds, key=turd_score)
        else:
            result = moves[self.rng.randint(len(moves))]

        # record current location as previous for next turn (helps avoid backtracking)
        try:
            self.prev_loc = my_loc
        except Exception:
            pass

        print(f"I have {time_left()} seconds left. Playing {result}.")
        return result
