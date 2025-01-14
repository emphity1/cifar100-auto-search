import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

########################################
#         GLOBAL PARAMETERS
########################################
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 3e-4

NUM_CLASSES = 100  # CIFAR-100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Boolean flags for attention blocks
USE_ECA = False
USE_SE = True
USE_COORDATT = False

# Accuracy threshold for saving the model
ACCURACY_THRESHOLD = 72.0
# Early-stop threshold & epoch
EARLY_STOP_THRESHOLD = 60.0
EARLY_STOP_EPOCH = 24  # => 24th epoch (0-based)

# Whether to store configurations that exceed the threshold (optional)
SAVE_SELECTED_CONFIG = True


def format_number(num):
    """Format large numbers nicely (k or M)."""
    if num is None:
        return "N/A"
    elif abs(num) >= 1_000_000:
        return f'{num / 1_000_000:.1f}M'
    elif abs(num) >= 1_000:
        return f'{num / 1_000:.1f}k'
    else:
        return str(num)


########################################
# CUTOUT (for data augmentation)
########################################
class Cutout(object):
    """
    Applies the Cutout technique:
    Zeros out (fills with 0) a square region of size `length`
    at a random position in the image.
    """
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        # img: Tensor of shape [C,H,W]
        h = img.size(1)
        w = img.size(2)

        # Randomly choose the square's center
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Calculate the boundaries
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        # Zero out
        img[..., y1:y2, x1:x2] = 0
        return img

########################################
#   ATTENTION BLOCKS DEFINITION
########################################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_ch = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_ch, bias=False),
            nn.GELU(),
            nn.Linear(mid_ch, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECABlock(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECABlock, self).__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if (t % 2) else (t + 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # (B,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B,1,C)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.GELU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # Pool over width -> shape [B, C, H, 1]
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        # Pool over height -> shape [B, C, 1, W], then transpose
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        x_w = x_w.transpose(2, 3)

        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)

        out_h = torch.sigmoid(self.conv_h(y_h))
        out_w = torch.sigmoid(self.conv_w(y_w))
        out = x * out_h + x * out_w
        return out

########################################
#  MBConvBlock with GELU
########################################
class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, stride, kernel_size):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = in_ch * expansion

        layers = []
        # Expand
        if expansion != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())

        # Depthwise
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

        # Attention blocks
        if USE_SE:
            layers.append(SEBlock(hidden_dim))
        if USE_ECA:
            layers.append(ECABlock(hidden_dim))
        if USE_COORDATT:
            layers.append(CoordAtt(hidden_dim))

        # Projection
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out

########################################
#  MyEfficientNet -> CIFAR-100
########################################
class MyEfficientNet(nn.Module):
    def __init__(self, block_cfgs):
        super().__init__()
        first_in = block_cfgs[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_in, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_in),
            nn.GELU()
        )
        blocks_list = []
        for (in_ch, out_ch, expansion, stride, ksz) in block_cfgs:
            mb = MBConvBlock(in_ch, out_ch, expansion, stride, ksz)
            blocks_list.append(mb)
        self.blocks = nn.Sequential(*blocks_list)

        last_out = block_cfgs[-1][1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_out, NUM_CLASSES)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

########################################
#   TRAIN/TEST UTILS
########################################
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss = running_loss / len(loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main():
    # 1) CIFAR-100 loader (with Cutout in the training transform)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        ),
        # Add Cutout
        Cutout(length=16)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        )
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2) Read valid_configs.json
    with open("valid_configs.json", "r") as f:
        valid_list = json.load(f)

    print(f"Loaded {len(valid_list)} configurations from JSON.")

    # [Optional] Keep a list of configs that exceed the threshold
    selected_configs = []

    # 3) For each config, train up to 50 epochs, with early-stop
    for idx, config_item in enumerate(valid_list, start=1):
        block_cfgs = config_item["block_cfgs"]
        item_params = config_item.get("params", None)
        item_flops = config_item.get("flops", None)

        def format_n(num):
            return format_number(num) if num else "N/A"
        param_str = format_n(item_params)
        flops_str = format_n(item_flops)

        print(f"\n=== Config #{idx} ===")
        print(f"block_cfgs = {block_cfgs}")
        print(f"params={param_str}, flops={flops_str}")

        # Build the model
        model = MyEfficientNet(block_cfgs).to(DEVICE)

        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        discard_config = False

        for e in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            if test_acc > best_acc:
                best_acc = test_acc
            print(f"Epoch [{e+1}/{EPOCHS}] - "
                  f"TrainLoss={train_loss:.3f}, TestLoss={test_loss:.3f}, TestAcc={test_acc:.2f}%")

            scheduler.step()

            # Early stop at the 24th epoch if best_acc < EARLY_STOP_THRESHOLD
            if e == EARLY_STOP_EPOCH:
                if best_acc < EARLY_STOP_THRESHOLD:
                    print(f" => Early stop, best_acc < {EARLY_STOP_THRESHOLD}% at 24th epoch. Discarding config.")
                    discard_config = True
                    break

        if discard_config:
            continue

        # If we exceed ACCURACY_THRESHOLD, save
        if best_acc > ACCURACY_THRESHOLD:
            acc_str = f"{best_acc:.1f}".replace('.', '_')
            filename = f"{param_str}_{flops_str}_{acc_str}.pt"

            print(f"*** Threshold {ACCURACY_THRESHOLD}% exceeded (best_acc={best_acc:.2f}%) ***")
            print(f"Saving TorchScript model as: {filename}")
            scripted_model = torch.jit.script(model.cpu())
            scripted_model.save(filename)

            if SAVE_SELECTED_CONFIG:
                # Save the config
                selected_info = {
                    "block_cfgs": block_cfgs,
                    "params": item_params,
                    "flops": item_flops,
                    "best_acc": best_acc
                }
                selected_configs.append(selected_info)

    # If we want to store the configs that exceed the threshold
    if SAVE_SELECTED_CONFIG and len(selected_configs) > 0:
        with open("selected_config.json", "w") as f:
            # Compact writing style if preferred
            json.dump(selected_configs, f, separators=(",", ":"))
        print(f"\nSaved {len(selected_configs)} configs to 'selected_config.json'.")

    print("\n== END: training completed for all configurations ==")


if __name__ == "__main__":
    main()
