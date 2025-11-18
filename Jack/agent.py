from collections import deque
from typing import List, Tuple
import numpy as np
from game import *

"""
Melvin is a random agent that selects moves heuristically to maximize egg laying
and minimize opponent advantage.
"""


class PlayerAgent:
    """
    Entry points: __init__ and play should not be changed.
    """

    def __init__(self, board: board.Board, time_left):
        self.rng = np.random.RandomState()
        self.prev_positions = deque(maxlen=8)
        self.moves_made = 0
        self.spawn_turd_placed = False
        self.own_eggs = set()

    def play(self, board: board.Board, sensor_data: List[Tuple[bool, bool]], time_left):
        from game.enums import MoveType, loc_after_direction, Direction

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        self.own_eggs = set(getattr(board, "eggs_player", set()))

        moves = board.get_valid_moves()
        if not moves:
            return None

        filtered_no_own_eggs = []
        allowed_but_own_eggs = []

        for m in moves:
            d, mt = m
            dest = loc_after_direction(my_loc, d)

            if dest == enemy_loc or dest in board.eggs_enemy or dest in board.turds_enemy:
                continue

            if dest in board.eggs_player or dest in self.own_eggs or dest in self.prev_positions:
                allowed_but_own_eggs.append(m)
                continue

            if any(loc_after_direction(dest, dd) in board.turds_enemy for dd in Direction):
                continue

            if mt == MoveType.EGG and not board.can_lay_egg_at_loc(my_loc):
                continue
            if mt == MoveType.TURD and not board.can_lay_turd_at_loc(my_loc):
                continue

            filtered_no_own_eggs.append(m)

        moves = filtered_no_own_eggs or allowed_but_own_eggs
        if not moves:
            return None

        # Place a turd on our starting point on the first opportunity so the
        # opponent can't easily block us by stepping onto our start square.
        # We prefer doing this before any egg/plain heuristics on the first
        # meaningful move.
        if not self.spawn_turd_placed:
            # TURD move places a turd at our current location (then moves us),
            # so look for any TURD move in the valid moves list. Previously we
            # checked for a "stay" direction which doesn't exist, so the list
            # was always empty.
            turd_here = [m for m in moves if m[1] == MoveType.TURD]
            if turd_here and board.can_lay_turd_at_loc(my_loc):
                # pick the first available turd move (could be improved to
                # choose a safer direction)
                chosen = turd_here[0]
                self.spawn_turd_placed = True
                # update simple bookkeeping the same way we do for regular moves
                self.prev_positions.append(my_loc)
                self.moves_made += 1
                return chosen

        def center_dist(loc):
            cx = cy = (board.game_map.MAP_SIZE - 1) / 2.0
            return ((loc[0] - cx) ** 2 + (loc[1] - cy) ** 2) ** 0.5

        def dest_for(move):
            return loc_after_direction(my_loc, move[0])

        heard_or_felt = any(a or b for (a, b) in sensor_data)

        def diagonal_coverage(loc):
            count = 0
            for dx in (-1, 1):
                for dy in (-1, 1):
                    cand = (loc[0] + dx, loc[1] + dy)
                    if board.is_valid_cell(cand) and board.can_lay_egg_at_loc(cand):
                        count += 1
            return count

        def score(move):
            _, mv = move
            dest = dest_for(move)
            base = center_dist(dest)
            enemy_dist = ((dest[0] - enemy_loc[0])**2 + (dest[1] - enemy_loc[1])**2)**0.5
            cover = diagonal_coverage(my_loc if mv == MoveType.EGG else dest)
            val = cover * 8.0 + base * 0.4 + enemy_dist * 0.15
            if heard_or_felt:
                val += base * 0.3
            return val

        eggs = [m for m in moves if m[1] == MoveType.EGG]
        plains = [m for m in moves if m[1] == MoveType.PLAIN]
        turds = [m for m in moves if m[1] == MoveType.TURD]

        # Heuristic selection
        result = None
        if eggs:
            result = max(eggs, key=score)
        elif plains:
            result = max(plains, key=score)
        elif turds:
            def turd_score(m):
                dest = dest_for(m)
                return abs(dest[0] - enemy_loc[0]) + abs(dest[1] - enemy_loc[1])
            result = max(turds, key=turd_score)

        if result and result[1] == MoveType.EGG:
            self.own_eggs.add(my_loc)

        self.prev_positions.append(my_loc)
        self.moves_made += 1

        return result
