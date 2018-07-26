import argparse
import os
from datetime import datetime

import torch.optim
from tensorboardX import SummaryWriter

from data import DataLoader
from model import PGNN
from utils import bcolors, batch_format, save_model, load_model

# Parse arguments
parser = argparse.ArgumentParser(description='PGN seq2seq summarization model')
parser.add_argument('--proceed', type=bool, default=False, help='proceed learning, or start new one learning process')
parser.add_argument('--dataset', type=str, default='./dataset/', help='dataset directory')
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--vocab_size', type=int, default=25000, help='vocabulary size')
parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size')
parser.add_argument('--val_batch_size', type=int, default=32, help='validation batch size')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--eval_interval', type=int, default=100, help='validation interval')
parser.add_argument('--sample_interval', type=int, default=100, help='sample interval')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--hidden_size', type=int, default=250, help='hidden size of model')
parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--clip', type=float, default=50, help='gradient clipping threshold')
parser.add_argument('--logdir', type=str, default='./logs/', help='log directory')
parser.add_argument('--model_name', type=str, default='model.pkl', help='name of model file')

args = parser.parse_args()

models_folder = './models'
model_filename = os.path.join(models_folder, args.model_name)
os.makedirs(os.path.dirname(model_filename), exist_ok=True)

# Check for resuming learning process
if args.proceed:
    print('Proceed learning of exciting model')
    assert os.path.exists(model_filename), "Can't find pretrained model file"
    # Load model & meta
    model, optimizer, last_iter, args, log_folder = load_model(model_filename)
else:
    # TensorboardX log folder
    log_folder = os.path.join(args.logdir, datetime.today().strftime("%Y-%m-%d-%H:%M:%S"))
    last_iter = 0

writer = SummaryWriter(log_folder)

# Check CUDA and choose device
if torch.cuda.is_available():
    if not args.cuda:
        print(bcolors.OKGREEN + 'You have a CUDA device, so you should probably run with --cuda' + bcolors.ENDC)
else:
    if args.cuda:
        print(bcolors.FAIL + 'You have no CUDA device. Start learning on CPU.')

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

# Load dataset
print('Loading dataset...')
dataset = DataLoader(args.dataset, ['train', 'validation'], args.vocab_size)
print(
    'Dataset contains {} examples. Vocabulary size: {}.\nBegin training'.format(dataset.part_lens, dataset.vocab_size))

# Create model
if not args.proceed:
    model = PGNN(dataset.vocab_size, args.hidden_size, args.n_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Main loop:
num_iters = dataset.part_lens['train'] // args.train_batch_size * args.epochs

for i in range(last_iter, num_iters):
    try:
        loss, grads = model.train_step(optimizer, dataset.next_batch('train', args.train_batch_size, device), args.clip)
        writer.add_scalar('PGNN/loss', loss, i)
        writer.add_scalar('PGNN/grad', grads, i)
        if i % args.log_interval == 0:
            print('[{:.2f}%] Iteration {};\t Loss: {:.5f}\t Gradients: {:.5f}'.format(i / num_iters * 100, i, loss,
                                                                                      grads))

        if i % args.eval_interval == 0:
            eval = model.evaluate(dataset.next_batch('validation', args.val_batch_size, device))
            loss, target_seq, target_seq_distr, pointer_distr, att_distr = eval
            writer.add_scalar('PGNN/validation-loss', loss, i)
            print('\n{}Validation loss: {}{}'.format(bcolors.OKBLUE, loss, bcolors.ENDC))

        if i % args.sample_interval == 0:
            batch = dataset.next_batch('validation', 1, device)
            sample = model.evaluate(batch)
            loss, target_seq, target_seq_distr, pointer_distr, att_distr = sample

            target = batch_format(batch, batch.trg_ext[0])[0]
            result = batch_format(batch, target_seq)[0]
            article = batch_format(batch, batch.src[0])[0]
            print(bcolors.UNDERLINE + bcolors.BOLD + 'Sampling text:' + bcolors.ENDC)
            print('Source: {}\nSummarization: {}\nGenerated: {}\n'.format(article, target, result))
    except KeyboardInterrupt:
        last_iter = i
        break

save_model(model_filename, model, optimizer, last_iter, args, log_folder)
