import numpy as np
import time
import traceback
from typing import List, Tuple, Callable

from game.enums import Direction, MoveType
from game.game_map import prob_hear, prob_feel


class BaseCapitalist:
    def __init__(self, board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.corner_reward = board.game_map.CORNER_REWARD
        # 0 = Even (White), 1 = Odd (Black)
        self.trapdoor_probs = [
            np.zeros((self.map_size, self.map_size)),
            np.zeros((self.map_size, self.map_size)),
        ]
        self.initialize_priors()

    def initialize_priors(self):
        for x in range(self.map_size):
            for y in range(self.map_size):
                dist_x = min(x, self.map_size - 1 - x)
                dist_y = min(y, self.map_size - 1 - y)
                dist_edge = min(dist_x, dist_y)

                weight = 0.0
                if dist_edge == 1:
                    weight = 1.0
                elif dist_edge >= 2:
                    weight = 2.0

                parity = (x + y) % 2
                self.trapdoor_probs[parity][x, y] = weight

        for i in range(2):
            total = np.sum(self.trapdoor_probs[i])
            if total > 0:
                self.trapdoor_probs[i] /= total

    def update_beliefs(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        cx, cy = current_loc
        for t_idx in range(2):
            heard, felt = sensor_data[t_idx]
            likelihood = np.zeros((self.map_size, self.map_size))

            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x + y) % 2 != t_idx:
                        continue
                    dx = abs(x - cx)
                    dy = abs(y - cy)

                    p_h = prob_hear(dx, dy)
                    p_hear_cond = p_h if heard else (1.0 - p_h)

                    p_f = prob_feel(dx, dy)
                    p_feel_cond = p_f if felt else (1.0 - p_f)

                    likelihood[x, y] = p_hear_cond * p_feel_cond

            self.trapdoor_probs[t_idx] *= likelihood
            s = np.sum(self.trapdoor_probs[t_idx])
            if s > 0:
                self.trapdoor_probs[t_idx] /= s

    def get_trapdoor_risk(self, loc: Tuple[int, int]) -> float:
        return self.trapdoor_probs[(loc[0] + loc[1]) % 2][loc[0], loc[1]]

    @staticmethod
    def manhattan_dist(p1, p2) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def nearest_safe_corner(self, board, my_loc) -> Tuple[Tuple[int, int] | None, int]:
        corners = [
            (0, 0),
            (0, self.map_size - 1),
            (self.map_size - 1, 0),
            (self.map_size - 1, self.map_size - 1),
        ]
        my_parity = board.chicken_player.even_chicken
        best_corner = None
        best_dist = 999

        blocked = set(board.eggs_player) | set(board.eggs_enemy) | set(board.turds_player) | set(board.turds_enemy)

        for cx, cy in corners:
            if (cx + cy) % 2 != my_parity:
                continue
            if (cx, cy) in blocked:
                continue
            d = self.manhattan_dist(my_loc, (cx, cy))
            if d < best_dist:
                best_dist = d
                best_corner = (cx, cy)

        return best_corner, best_dist

    def evaluate_node(self, board, my_start_eggs) -> float:
        raise NotImplementedError

    def order_root_moves(self, moves, board):
        moves = list(moves)
        moves.sort(key=lambda m: 0 if m[1] == MoveType.EGG else (1 if m[1] == MoveType.TURD else 2))
        return moves

    def order_inner_moves(self, moves, board):
        return self.order_root_moves(moves, board)

    def turd_sanity(self, best_move, board):
        if best_move[1] != MoveType.TURD:
            return best_move

        enemy_loc = board.chicken_enemy.get_location()
        my_loc = board.chicken_player.get_location()
        if self.manhattan_dist(my_loc, enemy_loc) > 6:
            return (best_move[0], MoveType.PLAIN)
        return best_move

    def play(self, board, sensor_data, time_left):
        try:
            start_time = time.perf_counter()
            self.update_beliefs(board.chicken_player.get_location(), sensor_data)

            valid_moves = board.get_valid_moves()
            if not valid_moves:
                return (Direction.UP, MoveType.PLAIN)

            tl = time_left()
            if tl > 2.0:
                BEAM_WIDTH = 5
                DEPTH = 4
            elif tl > 0.5:
                BEAM_WIDTH = 4
                DEPTH = 3
            else:
                BEAM_WIDTH = 2
                DEPTH = 1

            start_eggs = board.chicken_player.get_eggs_laid()

            beam = []

            ordered_root = self.order_root_moves(valid_moves, board)
            for move in ordered_root:
                next_board = board.forecast_move(move[0], move[1])
                if not next_board:
                    continue

                score = self.evaluate_node(next_board, start_eggs)

                if move[1] == MoveType.EGG:
                    loc = board.chicken_player.get_location()
                    if (loc[0] in [0, self.map_size - 1] and loc[1] in [0, self.map_size - 1]):
                        score += 2000

                beam.append((move, next_board, score))

            if not beam:
                return ordered_root[0]

            beam.sort(key=lambda x: x[2], reverse=True)
            beam = beam[:BEAM_WIDTH]

            for d in range(1, DEPTH):
                new_beam = []
                for root_move, state_board, _ in beam:
                    if time.perf_counter() - start_time > (tl * 0.6):
                        break

                    poss_moves = state_board.get_valid_moves()
                    ordered_inner = self.order_inner_moves(poss_moves, state_board)

                    for m in ordered_inner[:4]:
                        future = state_board.forecast_move(m[0], m[1])
                        if not future:
                            continue
                        score = self.evaluate_node(future, start_eggs)
                        score -= d * 10
                        new_beam.append((root_move, future, score))

                if not new_beam:
                    break
                new_beam.sort(key=lambda x: x[2], reverse=True)
                beam = new_beam[:BEAM_WIDTH]

            best_move = beam[0][0]
            best_move = self.turd_sanity(best_move, board)
            return best_move

        except Exception:
            moves = board.get_valid_moves()
            if moves:
                return moves[0]
            return (Direction.UP, MoveType.PLAIN)


class DominantCapitalist(BaseCapitalist):
    def _count_safe_neighbors(self, board, loc: Tuple[int, int], for_enemy: bool) -> int:
        if loc is None:
            return 0

        x, y = loc
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        count = 0

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self.map_size or ny < 0 or ny >= self.map_size:
                continue

            blocked = board.is_cell_blocked((nx, ny))
            if for_enemy:
                if not blocked:
                    count += 1
            else:
                if blocked:
                    continue
                risk = self.get_trapdoor_risk((nx, ny))
                if risk < 0.10:
                    count += 1

        return count

    def _approx_two_ply_mobility(self, board) -> float:
        my_moves = board.get_valid_moves(enemy=False)
        if not my_moves:
            return 0.0

        sample = my_moves[:3]
        total_next = 0
        count = 0

        for m in sample:
            future = board.forecast_move(m[0], m[1])
            if not future:
                continue
            total_next += len(future.get_valid_moves(enemy=False))
            count += 1

        if count == 0:
            return 0.0
        return total_next / count

    def evaluate_node(self, board, my_start_eggs) -> float:
        score = 0.0

        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        score_lead = my_eggs - enemy_eggs

        score += score_lead * 1100
        score += my_eggs * 90

        my_moves = board.get_valid_moves(enemy=False)
        enemy_moves = board.get_valid_moves(enemy=True)

        if not my_moves:
            return -1_000_000
        if not enemy_moves:
            return 1_000_000

        score += (len(my_moves) - len(enemy_moves)) * 60

        two_ply_my = self._approx_two_ply_mobility(board)
        score += two_ply_my * 15

        occupied = (
            len(board.eggs_player)
            + len(board.eggs_enemy)
            + len(board.turds_player)
            + len(board.turds_enemy)
        )
        total_cells = self.map_size * self.map_size
        progress_ratio = occupied / max(1, total_cells)

        if score_lead < 0 or (score_lead == 0 and progress_ratio < 0.5):
            score += -len(enemy_moves) * 35
        if score_lead > 0 and progress_ratio > 0.4:
            score += len(my_moves) * 25

        my_spawn = board.chicken_player.get_spawn()
        enemy_spawn = board.chicken_enemy.get_spawn()

        my_free = self._count_safe_neighbors(board, my_spawn, for_enemy=False)
        enemy_free = self._count_safe_neighbors(board, enemy_spawn, for_enemy=True)

        score += (my_free - 2) * 80
        score += (2 - enemy_free) * 130

        my_loc = board.chicken_player.get_location()
        risk_here = self.get_trapdoor_risk(my_loc)

        if score_lead >= 3 and progress_ratio > 0.5:
            risk_tolerance = 0.005
        elif score_lead >= 2:
            risk_tolerance = 0.01
        elif score_lead <= -1:
            risk_tolerance = 0.15
        else:
            risk_tolerance = 0.04

        if risk_here > risk_tolerance:
            spawn = board.chicken_player.get_spawn()
            if board.is_cell_blocked(spawn):
                score -= risk_here * 1_000_000
            else:
                score -= (risk_here - risk_tolerance) * 200_000

        corner, dist_to_corner = self.nearest_safe_corner(board, my_loc)
        if corner is not None:
            base_corner_weight = 25
            if score_lead < 0:
                base_corner_weight *= 1.5
            elif score_lead > 2 and progress_ratio > 0.6:
                base_corner_weight *= 0.7

            score += (10 - dist_to_corner) * base_corner_weight

        enemy_loc = board.chicken_enemy.get_location()
        dist_to_enemy = self.manhattan_dist(my_loc, enemy_loc)

        my_turds = board.chicken_player.get_turds_left()
        enemy_turds = board.chicken_enemy.get_turds_left()

        score += (my_turds - enemy_turds) * 80

        if my_turds > 0:
            if 1 <= dist_to_enemy <= 3:
                score += 140
        else:
            if dist_to_enemy <= 2:
                score -= 120

        if dist_to_enemy <= 2:
            score -= 80
        elif dist_to_enemy <= 4:
            score += 40

        return score

    def order_root_moves(self, moves, board):
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        losing = my_eggs < enemy_eggs

        moves = list(moves)
        if losing:
            key_order = {MoveType.TURD: 0, MoveType.EGG: 1, MoveType.PLAIN: 2}
        else:
            key_order = {MoveType.EGG: 0, MoveType.PLAIN: 1, MoveType.TURD: 2}

        moves.sort(key=lambda m: key_order[m[1]])
        return moves

    def order_inner_moves(self, moves, board):
        moves = list(moves)
        key_order = {MoveType.PLAIN: 0, MoveType.EGG: 1, MoveType.TURD: 2}
        moves.sort(key=lambda m: key_order[m[1]])
        return moves

    def turd_sanity(self, best_move, board):
        if best_move[1] != MoveType.TURD:
            return best_move

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        dist = self.manhattan_dist(my_loc, enemy_loc)

        if dist > 5:
            return (best_move[0], MoveType.PLAIN)
        return best_move


class PlayerAgent(DominantCapitalist):
    """Self-contained strong agent for Jack (DominantCapitalist).
    Inherits `play` behavior from BaseCapitalist/DominantCapitalist.
    """
    pass