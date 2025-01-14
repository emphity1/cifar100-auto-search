# cifar100-optuna

---

nohup python auto.py > auto.log 2>/dev/null &

Define a searchspace: number of blocks, range of desirable parameters etc...
It will save a json file with all configs which fits.
First block channel_in and last block channel_out are respectively selected from first 3 indexes and last 3 indexes of CHANNEL_SET[].
Blocks are setted to have alteast 2blocks with stride=1 and 3blocks with kernel=5 to reduce number of combinations and trying to maximize the accuracy of classification.
All these things can be changed.


nohup python train_valid_configs.py > train_valid_configs.log 2>/dev/null &

This code takes all configs found by auto.py and trains them for 50epochs.
If config doesn't reach X% accuracy in Y epochs, discard the configs and goes to next one.
If it reaches X accuracy in Y epochs, trains for 50 epochs, and if reaches ACCURACY_THRESHOLD, saves the model with jit as "parameters_flops_accuracy.pt"