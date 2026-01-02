#%%

from utils import *

#%%

DEVICE = "cuda"
DTYPE = t.bfloat16
MODEL_ID = "qwen3-4b"

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    device=DEVICE,
    dtype=DTYPE,
)

#%%

from utils import get_model_response

example_model_generation = False
if example_model_generation:
    # prompt = "What is the capital of France?"
    # prompt = "How do you take a derivative of a function?"
    # prompt = "if 2x + 3 = 11, what is x?"
    
    board_str = str(chess.Board("3q2k1/p6p/4ppp1/2p1Q3/2P5/4P1P1/r4P1P/1R4K1 w - - 0 27"))
    prompt = f"Here's a chess position:\n{board_str}\nWhat's the best move?"
    
    response = get_model_response(model, prompt, max_new_tokens=64)
    print(response)

    t.cuda.empty_cache()

#%%

from utils import convert_games_to_positions, load_positions_dataset

make_positions_dataset = False
if make_positions_dataset:
    dataset = get_dataset_csv("data/gm_games_original.csv")
    dataset = convert_games_to_positions(dataset, save_path="data/positions_dataset_small.csv", max_positions=1_000)

#%%

@dataclasses.dataclass
class GRPOTraningConfig:
    batch_size: int
    lr: float
    group_size: int
    kl_penalty: float
    prompt_format: str = "Here's a chess position:\n{position}\nWhat's the best move? Respond with only the move in UCI notation (like 'e2e4'), nothing else."
    
    invalid_move_reward: float = -100
    illegal_move_reward: float = -50

    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    bf16: bool = True


dataset = load_positions_dataset("data/positions_dataset_small.csv")

#%%


end_think_tok = "</think>"
end_turn_tok = "<|im_end|>"
end_think_tok_id = model.tokenizer.vocab[end_think_tok]
end_turn_tok_id = model.tokenizer.vocab[end_turn_tok]

engine = stockfish.Stockfish()
engine.set_depth(10)

cfg = GRPOTraningConfig(
    lr=3e-4,
    batch_size=4,
    group_size=2,
    kl_penalty=0.01,
)

model.requires_grad_(True)
model.train()

for i, example in tqdm(dataset.iterrows(), desc="Training GRPO"):
    with t.inference_mode():
        position = example["fen"]
        board = chess.Board(position)
        board_grid = str(board)
        board_fen = board.fen()
        print(board_fen)
        print(cyan, board_grid, endc)
        board_eval = eval_board(board, engine)

        user_prompt = cfg.prompt_format.format(position=board_fen)
        messages = [{"role": "user", "content": user_prompt}]
        prompt_toks = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(model.cfg.device)
        prompt_len = prompt_toks.shape[-1]

        completion_toks = model.generate(
            prompt_toks.repeat(cfg.group_size, 1),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            eos_token_id=end_turn_tok_id,
        )
        resp_strs = model.tokenizer.batch_decode(completion_toks)
        completion_len = completion_toks.shape[-1]
        print(yellow, resp_strs, endc)
        
        move_strs = [resp_str.split(end_think_tok)[-1].split(end_turn_tok)[0].strip() for resp_str in resp_strs]
        move_strs[0] = "e5e6"
        n_valid, n_legal = 0, 0
        rewards = []
        for move_str in move_strs:
            move = try_parse_move(move_str, type="uci")
            if move is not None:
                n_valid += 1
                if move in board.legal_moves:
                    n_legal += 1
                    move_eval = eval_move(board, move, engine)
                    rewards.append(move_eval)
                else:
                    rewards.append(cfg.illegal_move_reward)
            else:
                rewards.append(cfg.invalid_move_reward)

        print(move_strs)
        print(f"valid moves: {n_valid}, legal moves: {n_legal}")
        print(rewards)

        rewards = t.tensor(rewards, dtype=t.float32, device=model.cfg.device)
        avg_reward = rewards.mean()

        advantages = rewards - avg_reward
        print(advantages.tolist())

    advantages = advantages.clone()
    completion_toks = completion_toks.clone()

    logits = model(completion_toks)
    logprobs = t.log_softmax(logits, dim=-1)
    print(logits.shape)
    print(logprobs.shape)
    print(completion_toks.shape)
    # Use gather to select the logprobs associated with the chosen tokens after prompt
    
    # seq_indices = t.arange(prompt_len, prompt_len + completion_len).unsqueeze(0)
    chosen_toks = completion_toks[:, prompt_len:]
    chosen_tok_logprobs = logprobs[:, prompt_len:].gather(2, chosen_toks.unsqueeze(-1)).squeeze(-1)
    print(chosen_toks)
    print(chosen_tok_logprobs)
    x = chosen_tok_logprobs.sum()
    x.backward()
    
    # print(chosen_tok_logprobs[0].tolist())
    # print(chosen_tok_logprobs[1].tolist())
    # print(chosen_tok_logprobs[2].tolist())
    # print(chosen_tok_logprobs[3].tolist())

    # print([logprobs[0, j, k].item() for j, k in enumerate(completion_toks[0])])
    # print([logprobs[1, j, k].item() for j, k in enumerate(completion_toks[1])])
    # print([logprobs[2, j, k].item() for j, k in enumerate(completion_toks[2])])
    # print([logprobs[3, j, k].item() for j, k in enumerate(completion_toks[3])])

    break

#%%

