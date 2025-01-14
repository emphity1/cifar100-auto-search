import random
import json
import re
import torch
import torch.nn as nn
import torch.onnx
import onnx_tool
import numpy as np
import sys
import io
import contextlib

########################################
# GLOBAL PARAMETERS
########################################

N_BLOCKS = 7
MAX_SAMPLES = 50
MAX_ATTEMPTS = 100000

MIN_PARAM, MAX_PARAM = 200_000, 240_000
MIN_FLOPS, MAX_FLOPS = 10_000_000, 15_000_000

# CHANNEL_SET: multiples of 16 from 16 to 192, or any custom set
# CHANNEL_SET = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192]
CHANNEL_SET = [16, 24, 32, 40, 48, 64, 72, 80, 96, 112]

# Expansion: first block => [1,2,3], other blocks => [3,4,5,6]
FIRST_EXPANSION_SET = [1, 2]
OTHER_EXPANSION_SET = [2, 3, 4]

STRIDE_SET = [1, 2]
KERNEL_SET = [3, 5]

# Maximum allowed skip connections
MAX_SKIPS = 2
# Min blocks with stride=1
BLOCKS_WITH_STRIDES = 2
# Min blocks with kernel=5
BLOCKS_KERNEL5=3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

########################################
# UTILITY: format_number
########################################
def format_number(num):
    """Returns the number in k/M format, or N/A if None."""
    if num is None:
        return "N/A"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1000:.1f}k"
    else:
        return str(num)

########################################
# DEFINITION OF MBConvBlock
########################################
class MBConvBlock(nn.Module):
    """
    Minimal MBConv:
      - Expansion
      - Depthwise
      - Projection
      - skip if stride=1 and in_ch=out_ch
    """
    def __init__(self, in_ch, out_ch, expansion, stride, kernel_size):
        super().__init__()
        hidden_dim = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # 1) Expansion
        if expansion != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())

        # 2) Depthwise
        pad = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=hidden_dim,
            bias=False
        ))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())

        # 3) Projection
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out

########################################
# MODEL: chain of MBConv
########################################
class ChainOfBlocks(nn.Module):
    """
    N MBConv blocks in sequence, without stem/linear layers.
    For parameter/FLOPs calculation, we use a dummy input
    shape (1, first_in, 32, 32).
    """
    def __init__(self, block_cfgs):
        super().__init__()
        blocks_list = []
        for (iC, oC, exp, st, ksz) in block_cfgs:
            mb = MBConvBlock(iC, oC, exp, st, ksz)
            blocks_list.append(mb)
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        return self.blocks(x)

########################################
# FLOPs CALCULATION FUNCTION WITH LOG FIX
########################################
def calculate_flops_onnx(model, first_in):
    model.eval()
    dummy_input = torch.randn(1, first_in, 32, 32).to(DEVICE)
    onnx_path = "tmp_model.onnx"
    profile_path = "profile.txt"

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=12,
        do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )

    # Capture onnx_tool logs
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        onnx_tool.model_profile(onnx_path, save_profile=profile_path)
    # logs = f.getvalue()  # if you want to inspect onnx_tool logs

    with open(profile_path, 'r') as f2:
        textp = f2.read()

    match = re.search(r'^Total\s+.*?([\d,]+)\s+100%', textp, flags=re.MULTILINE)
    if match:
        total_macs_str = match.group(1)
        total_macs = int(total_macs_str.replace(',', ''))
        return total_macs
    else:
        return None

########################################
# RANDOM CHAIN CONSTRUCTION
########################################

def build_chain(n_blocks, first_in, last_out):
    """
    Creates a chain of n_blocks:
      - Block0 => in_ch=first_in, out_ch >= in_ch
                  expansions => FIRST_EXPANSION_SET
      - Intermediate blocks => in_ch= previous out_ch, out_ch >= in_ch
                               expansions => OTHER_EXPANSION_SET
      - Final block => in_ch= previous out_ch, out_ch= last_out
                       expansions => OTHER_EXPANSION_SET
    """
    if n_blocks < 2:
        return None

    chain = []

    # 1) First block
    out_candidates = [c for c in CHANNEL_SET if c >= first_in]
    if not out_candidates:
        return None
    outC0 = random.choice(out_candidates)
    exp0 = random.choice(FIRST_EXPANSION_SET)
    st0  = random.choice(STRIDE_SET)
    ksz0 = random.choice(KERNEL_SET)
    chain.append([first_in, outC0, exp0, st0, ksz0])

    # 2) blocks [1..n_blocks-2]
    for i in range(1, n_blocks-1):
        prev_out = chain[-1][1]
        out_candidates_i = [c for c in CHANNEL_SET if c >= prev_out and c <= last_out]
        if not out_candidates_i:
            return None
        outC = random.choice(out_candidates_i)
        exp  = random.choice(OTHER_EXPANSION_SET)
        st   = random.choice(STRIDE_SET)
        ksz  = random.choice(KERNEL_SET)
        chain.append([prev_out, outC, exp, st, ksz])

    # 3) Last block => out_ch= last_out
    prev_out = chain[-1][1]
    if last_out < prev_out:
        return None
    exp_last = random.choice(OTHER_EXPANSION_SET)
    st_last  = random.choice(STRIDE_SET)
    ksz_last = random.choice(KERNEL_SET)
    chain.append([prev_out, last_out, exp_last, st_last, ksz_last])

    return chain

########################################
# MAIN
########################################
def main():
    valid_configs = []
    seen = set()
    attempts = 0

    # Indices: first 3 => CHANNEL_SET[:3], last 3 => CHANNEL_SET[-3:]
    first_in_candidates = CHANNEL_SET[:3]
    last_out_candidates = CHANNEL_SET[-3:]

    while len(valid_configs) < MAX_SAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1

        # 1) Choose first_in from the first 3
        first_in = random.choice(first_in_candidates)
        # 2) Choose last_out from the last 3, must be >= first_in
        possible_last = [c for c in last_out_candidates if c >= first_in]
        if not possible_last:
            continue
        last_out = random.choice(possible_last)

        # 3) Build the chain
        chain = build_chain(N_BLOCKS, first_in, last_out)
        if chain is None:
            continue

        # Requirements:
        # - At least 2 blocks with stride=1
        stride1_count = sum(1 for b in chain if b[3] == 1)
        if stride1_count < BLOCKS_WITH_STRIDES:
            continue

        # - At least 3 blocks with kernel=5
        kernel5_count = sum(1 for b in chain if b[4] == 5)
        if kernel5_count < BLOCKS_KERNEL5:
            continue

        # - skip connections => out_ch == in_ch => at most MAX_SKIPS
        skip_count = sum(1 for b in chain if b[1] == b[0])
        if skip_count > MAX_SKIPS:
            continue

        # Avoid duplicates
        chain_tup = tuple(tuple(b) for b in chain)
        if chain_tup in seen:
            continue
        seen.add(chain_tup)

        # Build the model
        net = ChainOfBlocks(chain).to(DEVICE)

        # Count parameters
        params_count = sum(p.numel() for p in net.parameters() if p.requires_grad)

        # FLOPs
        try:
            flops_onnx = calculate_flops_onnx(net, first_in)
        except:
            flops_onnx = None

        params_str = format_number(params_count)
        flops_str = format_number(flops_onnx)

        print(f"Attempt {attempts}, chain={chain}, "
              f"params={params_str}, flops={flops_str}", flush=True)

        if flops_onnx is None:
            continue

        # Param/FLOPs filter
        if (params_count >= MIN_PARAM and params_count <= MAX_PARAM and
            flops_onnx >= MIN_FLOPS and flops_onnx <= MAX_FLOPS):
            print("  => MATCH! Saving this config.")
            valid_configs.append({
                "block_cfgs": chain,
                "params": params_count,
                "flops": flops_onnx
            })

    print(f"\n=== SEARCH COMPLETE ===")
    print(f"Attempts: {attempts}")
    print(f"Valid configs found: {len(valid_configs)}")

    with open("valid_configs.json","w") as f:
        json.dump(valid_configs, f, indent=2)
    print("Saved to valid_configs.json")


if __name__ == "__main__":
    main()
