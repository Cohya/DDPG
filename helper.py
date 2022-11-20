import argparse


parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--hidden_layer_sizes', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_train_episodes', type=int, default=200)
parser.add_argument('--save_folder', type=str, default='ddpg_monitor')
args = parser.parse_args()