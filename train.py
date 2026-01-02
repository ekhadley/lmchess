#%%

from utils import *

#%%

DEVICE = "cuda"
DTYPE = t.bfloat16
MODEL_ID = "qwen3-1.7b"
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    device=DEVICE,
    dtype=DTYPE,
)

RUNNING_LOCAL = 'arch' in platform.release()


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
    kl_beta: float
    prompt_format: str = "Here's a chess position:\n{position}\nWhat's the best move? Respond with only the move in UCI notation (like 'e2e4'), nothing else."
    
    invalid_move_reward: float = -100
    illegal_move_reward: float = -50
    clip_eps: float = 0.05

    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    bf16: bool = True


dataset = load_positions_dataset("data/positions_dataset_small.csv")

ref_model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    device=DEVICE,
    dtype=DTYPE,
)
ref_model.requires_grad_(False)
ref_model.eval()

#%%

end_think_tok = "</think>"
end_turn_tok = "<|im_end|>"
end_think_tok_id = model.tokenizer.vocab[end_think_tok]
end_turn_tok_id = model.tokenizer.vocab[end_turn_tok]

engine_path = "/usr/bin/stockfish" if RUNNING_LOCAL else "/home/ehadley/.local/bin/stockfish"
engine = stockfish.Stockfish(path=engine_path)
engine.set_depth(10)

cfg = GRPOTraningConfig(
    lr=3e-4,
    batch_size=4,
    group_size=2,
    kl_beta=0.01,
)

#%%

for i, example in tqdm(dataset.iterrows(), desc=f"{pink}Training"):
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
        reward_mean = rewards.mean()
        reward_std = rewards.std()

        advantages = (rewards - reward_mean) / reward_std
        print(advantages.tolist())
        
        chosen_toks = completion_toks[:, prompt_len:]

        ref_logits = ref_model(completion_toks)
        ref_logprobs = t.log_softmax(ref_logits, dim=-1)
        ref_chosen_tok_logprobs = ref_logprobs[:, prompt_len:].gather(2, chosen_toks.unsqueeze(-1)).squeeze(-1)

    chosen_toks = chosen_toks.clone()
    advantages = advantages.clone()
    completion_toks = completion_toks.clone()

    logits = model(completion_toks)
    logprobs = t.log_softmax(logits, dim=-1)
    print(logits.shape)
    print(logprobs.shape)
    print(completion_toks.shape)
    
    # seq_indices = t.arange(prompt_len, prompt_len + completion_len).unsqueeze(0)
    chosen_tok_logprobs = logprobs[:, prompt_len:].gather(2, chosen_toks.unsqueeze(-1)).squeeze(-1)
    print(chosen_toks)
    print(chosen_tok_logprobs)

    prox = t.exp(chosen_tok_logprobs - ref_chosen_tok_logprobs)
    print(prox)
    prox_adv = prox*advantages.unsqueeze(-1)
    print(prox_adv)
    prox_clipped = t.min(prox_adv, t.clip(prox_adv, 1-cfg.clip_eps, 1+cfg.clip_eps))
    print(prox_clipped)

    kl_div = 
    
    
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

