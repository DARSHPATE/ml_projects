from copy import copy, deepcopy
import numpy as np
import pygame
from piece import BODIES, Piece
from board import Board
from random import randint

import json
import os

def column_heights(grid):
    return np.where(grid.any(axis=0), grid.shape[0] - np.argmax(grid[::-1, :], axis=0), 0)

def total_height(grid, column_heights):
    return np.sum(column_heights)

def count_holes(grid):
    return np.sum((np.cumsum(grid, axis=0) > 0) & (~grid))

def maximum_height(grid, column_heights):
    return np.max(column_heights)

def bumpiness(grid, column_heights):
    return np.sum(np.abs(np.diff(column_heights)))

def wells(grid, column_heights):
    h = grid.shape[0]
    padded = np.concatenate([[h], column_heights, [h]])
    return np.sum(np.maximum(0, np.minimum(padded[:-2], padded[2:]) - padded[1:-1]))

def row_transitions(grid):
    padded = np.pad(grid, ((0, 0), (1, 1)), constant_values=1)
    return np.sum(np.sum(padded[:, :-1] != padded[:, 1:], axis=1))

def column_transitions(grid):
    padded = np.pad(grid, ((0, 1), (0, 0)), constant_values=1)
    return np.sum(np.sum(padded[:-1, :] != padded[1:, :], axis=0))

def total_filled(grid):
    return np.sum(grid)

def valuation(board, chromosome):
    sim_board = deepcopy(board)
    rows_cleared = sim_board.clear_rows()
    grid = np.array(sim_board.board, dtype=bool)
    heights = column_heights(grid)
    features = np.array([
        total_height(grid, heights),
        maximum_height(grid, heights),
        count_holes(grid),
        bumpiness(grid, heights),
        rows_cleared,
        wells(grid, heights),
        row_transitions(grid),
        column_transitions(grid),
        total_filled(grid)
    ])
    return np.dot(features, chromosome)

class CUSTOM_AI_MODEL:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "weights.json"), "r") as f:
            self.chromosome = json.load(f)
    def evaluate_board(self, board):
        return valuation(board, self.chromosome)
    def get_possible_moves(self, board, piece):
        seen = set()
        rotations = []
        while True:
            body = tuple(sorted(piece.body))
            if body in seen:
                break
            seen.add(body)
            rotations.append(piece)
            piece = piece.get_next_rotation()
        moves_list = []
        for rotation in rotations:
            max_x = board.width - len(rotation.skirt)
            moves_list.extend([(x, rotation) for x in np.arange(max_x + 1)])
        moves = np.array(moves_list, dtype=object)
        return moves
    def get_best_move(self, board, piece):
        moves = self.get_possible_moves(board, piece)
        scores = []
        simulated_moves = []
        for x, move_piece in moves:
            temp_board = deepcopy(board)
            temp_board.place(x, temp_board.drop_height(move_piece, x), move_piece)
            scores.append(self.evaluate_board(temp_board))
            simulated_moves.append((x, move_piece))
        if not scores:
            return (0, piece)
        scores = np.array(scores)
        best = np.argmax(scores)
        return simulated_moves[best]