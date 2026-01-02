import os
import json
import csv
import dataclasses
import random
from tqdm import tqdm, trange

import pandas as pd
import chess
import stockfish
from IPython import get_ipython

import torch as t
from torch import Tensor

from transformer_lens import HookedTransformer

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

IPYTHON = get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

# =================== model ===================== #

def get_model_response(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    temperature: float = 1.0,
) -> str:
    messages = [{'role': 'user', 'content': prompt}]
    prompt_toks = model.tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.cfg.device)
    resp_toks = model.generate(
        prompt_toks,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    return model.tokenizer.decode(resp_toks[0])

# =================== dataset ===================== #

def get_dataset_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def convert_games_to_positions(
    dataset: pd.DataFrame,
    max_positions: int | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    positions = []
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        
        board = chess.Board()
        if not isinstance(row["lan"], str): continue
        moves = row["lan"].split(" ")
        for i, move in enumerate(moves):
            board.push_uci(move)
            if board.is_game_over(): break
            fen = board.fen()
            position_row = [row["game_id"], row["winner"], move, fen, " ".join(moves[:i+1])]
            positions.append(position_row)
            if max_positions is not None and len(positions) >= max_positions:
                break
        if max_positions is not None and len(positions) >= max_positions:
            break
    
    random.shuffle(positions)
    positions_dataset = pd.DataFrame(positions, columns=["game_id", "winner", "move", "fen", "moves"])
    if save_path is not None:
        positions_dataset.to_csv(save_path, index=False)
    return positions_dataset

def load_positions_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def try_parse_move(move_str: str, type: str = "san", board: chess.Board | None = None) -> chess.Move | None:
    try:
        if type == "san":
            if board is None:
                raise ValueError("Board is required for SAN move parsing")
            return board.parse_san(move_str)
        elif type == "uci":
            return chess.Move.from_uci(move_str)
        else:
            raise ValueError(f"Invalid move type: {type}")
    except ValueError:
        return None

def eval_board(board: chess.Board, engine: stockfish.Stockfish) -> float:
    engine.set_position(board.fen())
    board_eval = engine.get_evaluation()
    assert board_eval["type"] == "cp"
    return float(board_eval["value"])

def eval_move(board: chess.Board, move: chess.Move, engine: stockfish.Stockfish) -> float:
    board_eval = eval_board(board, engine)
    board.push(move)
    new_eval = eval_board(board, engine)
    board.pop()
    return board_eval - new_eval