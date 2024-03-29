import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic=True

import os
import csv
import numpy as np
np.random.seed(0)

import time
import torch.nn as nn
from glob import glob
import visdom
from visdom_scripts.vis import VisdomLinePlotter
from argparse import ArgumentParser
from scipy.stats import pearsonr
import random
random.seed(0)

import sys
from statistics import stdev, mean

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout_rate=0):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.multi_head_attention(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))
        x = x + self.dropout1(attn_output)
        
        # Feed Forward
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout2(ff_output)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_channels=1940, num_classes=1, dropout=90, d_model=512, nhead=8, num_layers=6,dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_channels, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, num_layers, dim_feedforward, dropout/100) for _ in range(num_layers)]
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            # Add more dense layers if necessary
            nn.Dropout(dropout/100),
            nn.Linear(d_model, num_classes)
        )
        self.dropout = nn.Dropout(dropout/100)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add a batch dimension
        for block in self.transformer_blocks:
            x = block(x)

        x = self.dense_layers(x)
        return x.squeeze(0)

class FC(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, dropout=90):
        super(FC, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_channels, 4096),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.decoder(x)

        return x

class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, dropout=90):
        super(CNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout/100),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout/100),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout/100),
            nn.Conv1d(in_channels=64, out_channels=100, kernel_size=5, stride=1, padding=2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(100 * input_channels, 100),
            nn.Dropout(dropout/100),
            nn.Linear(100, 64),
            nn.Dropout(dropout/100),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Input shape: batch_size x 500
        # Reshape for 1D-CNN: batch_size x 1 x 500
        x = x.unsqueeze(1)

        x = self.encoder(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.decoder(x)
        
        return x

def calculate_r(output, truth):
    if len(set(output)) == 1:
        r = 0
    else:
        r, _ = pearsonr(output, truth)

    return r

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# MSE Loss function
def MSE(output, target):
    criterion = nn.MSELoss()
    return criterion(output,target)


# Return a subject->label mapping for float labels
def get_labels_mapping(f):
    result = {}
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            result[row[0]] = float(row[1])
    return result

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folds, dataset_dir='/Users/aritchetchenian/Desktop/graph_cnn', mean=None, stdev=None, output_mean=None, output_stdev=None, standardise_output=False):
        # Data structure for storing all items
        self.subjects = []

        all_names = ['Endurance_AgeAdj', 'GaitSpeed_Comp', 'Dexterity_AgeAdj', 'Strength_AgeAdj', 'PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj']
        self.labels = [get_labels_mapping('./csvs/' + name + '.csv') for name in all_names]

        # Gather subject IDs
        subject_ids = []
        for fold in folds:
            subject_ids += ['/'.join(x.replace('\\', '/').split('/')[-2:]) for x in sorted(glob(dataset_dir + '/fold'+str(fold)+'/*.npy'))]
        subject_ids.sort()

        # Calculate the mean/stdev
        if mean is None:
            data = np.array([np.load(dataset_dir + '/' + subject) for subject in subject_ids])
            self.mean, self.stdev = np.mean(data,axis=0), np.std(data,axis=0)

            #self.labels maps subject_ID to score for each task, i.e. it should have shape 11 x 1206
            data = np.array([[self.labels[i][subject.split('/')[-1].split('.')[0]] for i in range(len(self.labels))] for subject in subject_ids])
            self.output_mean, self.output_stdev = np.mean(data,axis=0), np.std(data,axis=0)
        else:
            self.mean, self.stdev = mean, stdev
            self.output_mean, self.output_stdev = output_mean, output_stdev

        for subject in subject_ids:
            # Load the output data and standardise it, if output standardisation is enabled
            total_label = np.array([self.labels[i][subject.split('/')[-1].split('.')[0]] for i in range(len(self.labels))])
            if standardise_output:
                total_label = (total_label - self.output_mean) / (self.output_stdev + 0.00001)

            # Load the input data and standardise it
            input_vector = np.load(dataset_dir + '/' + subject)
            input_vector = (input_vector - self.mean) / (self.stdev + 0.00001)

            self.subjects.append([torch.from_numpy(input_vector).float(), torch.tensor(total_label).float()])

    def __getitem__(self, idx):
        return self.subjects[idx]

    def get_stats(self):
        return [self.mean, self.stdev, self.output_mean, self.output_stdev]

    def __len__(self):
        return len(self.subjects)

# Define the arguments
parser = ArgumentParser(description="Arguments for model training.")
parser.add_argument("-b", "--batch_size", help="Batch size.", default=10, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs.", default=150, type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate.", default=1e-3, type=float)
parser.add_argument("-vis", "--vis_mode", help="Presence of this flag enables plotting/visualisation of results.", action='store_true')
parser.add_argument("-s", "--save_name", help="Folder in which to save all results to.", type=str, default="dump")
parser.add_argument("-i", "--input_channels", help="Number of input channels (1 = num SL only, 2 = num SL and FA)", default=1, type=int)
parser.add_argument("-rd", "--results_dir", help="Results directory (no final slash).", type=str, default="./results")
parser.add_argument("-dd", "--dataset_dir", help="Dataset directory (no final slash).", type=str, default="../GraphConnectome/splitted")
parser.add_argument("-g", "--grid_search", help="If doing a grid search, must turn this flag on. It will disable folds and test-set evaluation.", action='store_true')
parser.add_argument("-dr", "--dropout", help="Dropout percentage, e.g. 90", default=50, type=int)
parser.add_argument("-mo", "--model", help="Indicates which model to use (transformer, 1dcnn, fc).", type=str, default="transformer")
parser.add_argument("-st", "--standardise_output", help="Presence of this flag enables standardisation of output data", action='store_true')

args = parser.parse_args()

# Make the results directory if it doesn't exist
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

# Create the results directory
# 'dump' is a special case for rapid prototyping (will override the existing 'dump' files)
if args.save_name != 'dump':
    while os.path.exists(args.results_dir + '/' + args.save_name):
        args.save_name = input("Already exists. Enter new save name:")
    os.mkdir(args.results_dir + '/' + args.save_name)
elif args.save_name == 'dump' and not os.path.exists(args.results_dir + '/dump'):
    os.mkdir(args.results_dir + '/dump')

# Print all specified arguments
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
args_string = '_'.join([str(getattr(args, arg)) for arg in vars(args)]) # create string for a unique ID

# Choosing a device (CPU vs. GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Training on: ", device)

# Create the metric lists
test_rs = []
test_all_maes = []
test_maes = []
test_accs = []
test_f1s = []
test_rocs = []

for fold in range(5):
    # Create the dataset
    train_val_dataset = CustomDataset(dataset_dir=args.dataset_dir, folds=[x for x in range(5) if x != fold], standardise_output=args.standardise_output)
    train_mean, train_stdev, train_out_mean, train_out_stdev = train_val_dataset.get_stats()
    test_dataset = CustomDataset(dataset_dir=args.dataset_dir, folds=[fold], mean=train_mean, stdev=train_stdev, output_mean=train_out_mean, output_stdev=train_out_stdev, standardise_output=args.standardise_output)

    num_train = int(0.75 * len(train_val_dataset))
    num_val = len(train_val_dataset) - num_train
    train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

    # Visdom plotting initialisation
    # Can ignore this unless you're interested in visualisations
    if args.vis_mode:
        vis = visdom.Visdom(server='127.0.0.1', port='1111')
        loss_plotter = VisdomLinePlotter(env_name='Age Prediction', viz=vis)
        score_plotter = VisdomLinePlotter(env_name='Age Prediction', viz=vis)
        train_opts = dict(title='Train Histogram', xtickmin=90, xtickmax=160)
        valid_opts = dict(title='Valid Histogram', xtickmin=90, xtickmax=160)
        truth_opts = dict(title='Truth Histogram', xtickmin=90, xtickmax=160)
        train_win = None
        valid_win = None
        truth_win = None

    # Initialising the model
    if args.model.lower() == 'transformer':
        model = TransformerModel(input_channels=args.input_channels, dropout=args.dropout, num_classes=11)
    elif args.model.lower() == '1dcnn':
        model = CNN(input_channels=args.input_channels, dropout=args.dropout, num_classes=11)
    elif args.model.lower() == 'fc':
        model = FC(input_channels=args.input_channels, dropout=args.dropout, num_classes=11)
    else:
        print('ERROR: Invalid model.')
        sys.exit()
    print("PARAMS: %d" %(count_params(model)))
    model.to(device)
    print(model)

    # Initialising the optimiser/scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=15)

    # Main training/validation loop for the current fold of cross-validation
    valid_losses, train_losses = [], []
    valid_stats, train_stats = [], []
    valid_scores, train_scores = [], []
    best_valid_loss = None
    lrs = []
    for epoch in range(args.epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_count = 0
        train_outs = []
        train_truths = []
        for data, label in trainloader:
            # Send data to device
            data,label = data.to(device), label.to(device)

            # Reset optimizer
            optimizer.zero_grad()

            # Feed into model
            out = model(data)

            # Compute and backprop the loss
            loss = MSE(out, label)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Store output/metrics
            output_data = out.cpu().detach().numpy()

            if args.standardise_output:
                output_data = output_data * train_out_stdev + train_out_mean

            train_outs += list(output_data)

            label_data = label.cpu().detach().numpy()
            if args.standardise_output:
                label_data = label_data * train_out_stdev + train_out_mean
            train_truths += list(label_data)

            train_loss += loss.item() 
            train_count += 1

        # Validation
        model.eval()
        valid_loss = 0
        valid_count = 0
        valid_outs = []
        valid_truths = []
        with torch.no_grad():
            for data, label in validloader:
                # Send data to device
                data, label = data.to(device), label.to(device)

                # Feed into model
                out = model(data)

                # Calculate loss
                loss = MSE(out, label)

                # Store output/metrics
                output_data = out.cpu().numpy()

                if args.standardise_output:
                    output_data = output_data * train_out_stdev + train_out_mean

                valid_outs += list(output_data)

                label_data = label.cpu().detach().numpy()
                if args.standardise_output:
                    label_data = label_data * train_out_stdev + train_out_mean
                valid_truths += list(label_data)

                valid_loss += loss.item()
                valid_count += 1

        # Step the scheduler and print the current LR
        scheduler.step(valid_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        lrs.append(curr_lr)
        print('Current learning rate: %f' % (curr_lr))

        train_rs = [calculate_r(list(np.array(train_outs)[:,i]),list(np.array(train_truths)[:,i])) for i in range(len(train_outs[0]))]
        valid_rs = [calculate_r(list(np.array(valid_outs)[:,i]),list(np.array(valid_truths)[:,i])) for i in range(len(valid_truths[0]))]
    
        train_outs = list(np.array(train_outs).flatten())
        valid_outs = list(np.array(valid_outs).flatten())
        train_truths = list(np.array(train_truths).flatten())
        valid_truths = list(np.array(valid_truths).flatten())

        train_mae = np.mean(np.abs(np.array(train_outs) - np.array(train_truths)))
        valid_mae = np.mean(np.abs(np.array(valid_outs) - np.array(valid_truths)))
        
        # Plotting
        if args.vis_mode:
            # Plot the losses
            loss_plotter.plot('score', 'valid loss', 'Metric Curves', epoch, valid_loss/valid_count, yaxis_type='log')
            loss_plotter.plot('score', 'train loss', 'Metric Curves', epoch, train_loss/train_count, yaxis_type='log')
            loss_plotter.plot('score', 'curr LR', 'Metric Curves', epoch, curr_lr, yaxis_type='log')

            # Plot the metrics
            for i, item in enumerate(train_rs):
                score_plotter.plot('score', 'train R' + str(i), 'Metric Curves', epoch, item, yaxis_type='linear')
            for i, item in enumerate(valid_rs):
                score_plotter.plot('score', 'valid R' + str(i), 'Metric Curves', epoch, item, yaxis_type='linear')

            score_plotter.plot('score', 'train MAE', 'Metric Curves', epoch, train_mae, yaxis_type='linear')
            score_plotter.plot('score', 'valid MAE', 'Metric Curves', epoch, valid_mae, yaxis_type='linear')

            # Plot the histograms
            train_win = vis.histogram(train_outs, win=train_win, opts=train_opts, env='Age Prediction')
            valid_win = vis.histogram(valid_outs, win=valid_win, opts=valid_opts, env='Age Prediction')
            truth_win = vis.histogram(train_truths + valid_truths, win=truth_win, opts=truth_opts, env='Age Prediction')

        # Update metrics
        valid_losses.append(valid_loss/valid_count)
        train_losses.append(train_loss/train_count)
        valid_stats.append([min(valid_outs), max(valid_outs), sum(valid_outs)/len(valid_outs), sum(valid_rs)/len(valid_rs)])
        train_stats.append([min(train_outs), max(train_outs), sum(train_outs)/len(train_outs), sum(train_rs)/len(train_rs)])
        valid_scores.append([sum(valid_rs)/len(valid_rs), valid_mae] + valid_rs)
        train_scores.append([sum(train_rs)/len(train_rs), train_mae] + train_rs)

        # Print the current epoch
        print(epoch)

        # Aggregrate the metrics and save them for the current epoch
        metric_save_object = {
            'epoch': epoch,
            'train_outs': train_outs,
            'valid_outs': valid_outs,
            'train_r': sum(train_rs)/len(train_rs),
            'valid_r': sum(valid_rs)/len(valid_rs),
            'train_truths': train_truths,
            'valid_truths': valid_truths,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'valid_scores': valid_scores,
            'train_scores': train_scores,
        }
        torch.save(metric_save_object, args.results_dir + '/' + args.save_name + '/fold_' + str(fold) + '_current_stats.pth')

        # Save the current model if it has the lowest validation loss
        update_loss = False
        if best_valid_loss is None:
            update_loss = True
        elif valid_loss/valid_count < best_valid_loss:
            update_loss = True
        if update_loss:
            best_valid_loss = valid_loss / valid_count

            # Save the model
            save_object = {
                # state dicts
                'model_state_dict': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),

                # useful info
                'training_epoch': epoch,
                'total_epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.learning_rate,
                'vis_mode': args.vis_mode,
                'save_name': args.save_name,
                'lrs': lrs,
                'input_channels': args.input_channels,
                'dataset_dir': args.dataset_dir,
                'results_dir': args.results_dir,
                'grid_search': args.grid_search,
                'dropout': args.dropout,
                'train_mean': train_mean,
                'train_stdev': train_stdev,
                'train_out_mean': train_out_mean,
                'train_out_stdev': train_out_stdev,
                'standardise_output': args.standardise_output,

                # cross-val info
                'fold': fold,

                # metric info
                'train_outs': train_outs,
                'valid_outs': valid_outs,
                'train_r': sum(train_rs)/len(train_rs),
                'valid_r': sum(valid_rs)/len(valid_rs),
                'train_truths': train_truths,
                'valid_truths': valid_truths,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
            }

            torch.save(save_object, args.results_dir + '/' + args.save_name + '/fold_' + str(fold) + '_best_model_checkpoint.pth')

        total_time = (time.time() - start_time) / 60
        print("%.2f mins per epoch" % (total_time))
        print("%.2f mins per 20 epochs" % (total_time * 20))
        print('--')

    # If performing grid search, stop after the first fold has been trained/validated
    if args.grid_search:
        break
    
    # Eval on the test fold
    checkpoint = torch.load(args.results_dir + '/' + args.save_name + '/fold_' + str(fold) + '_best_model_checkpoint.pth')
    if args.model.lower() == 'transformer':
        model = TransformerModel(input_channels=args.input_channels, dropout=args.dropout, num_classes=11)
    elif args.model.lower() == '1dcnn':
        model = CNN(input_channels=args.input_channels, dropout=args.dropout, num_classes=11)
    elif args.model.lower() == 'fc':
        model = FC(input_channels=args.input_channels, dropout=args.dropout, num_classes=11)
    else:
        print('ERROR: Invalid model.')
        sys.exit()

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loss = 0
    test_count = 0
    test_outs = []
    test_truths = []
    with torch.no_grad():
        for data, label in testloader:
            # Send data to device
            data, label = data.to(device), label.to(device)

            # Feed into model
            out = model(data)

            # Calculate loss
            loss = MSE(out, label)

            # Store output/metrics
            output_data = out.cpu().numpy()
            if args.standardise_output:
                output_data = output_data * train_out_stdev + train_out_mean
            test_outs += list(output_data)

            label_data = label.cpu().detach().numpy()
            if args.standardise_output:
                label_data = label_data * train_out_stdev + train_out_mean
            test_truths += list(label_data)

            test_loss += loss.item()
            test_count += 1

    test_r_all = [calculate_r(list(np.array(test_outs)[:,i]), list(np.array(test_truths)[:,i])) for i in range(len(test_outs[0]))]
    test_r = sum(test_r_all) / len(test_r_all)

    test_all_mae = [np.mean(np.abs(np.array(test_outs)[:,i] - np.array(test_truths)[:,i])) for i in range(len(test_outs[0]))]

    np.save(args.results_dir + '/' + args.save_name + '/fold' + str(fold) + '_test_outs_not_flattened.npy', np.array(test_outs))
    np.save(args.results_dir + '/' + args.save_name + '/fold' + str(fold) + '_test_truths_not_flattened.npy', np.array(test_truths))

    test_outs = list(np.array(test_outs).flatten())
    test_truths = list(np.array(test_truths).flatten())

    test_overall_mae = np.mean(np.abs(np.array(test_outs) - np.array(test_truths)))

    np.save(args.results_dir + '/' + args.save_name + '/fold' + str(fold) + '_test_outs.npy', np.array(test_outs))
    np.save(args.results_dir + '/' + args.save_name + '/fold' + str(fold) + '_test_truths.npy', np.array(test_truths))

    print(test_overall_mae)
    print(test_r)

    test_maes.append(test_overall_mae)
    test_rs.append(test_r_all)
    test_all_maes.append(test_all_mae)

if not args.grid_search:
    with open(args.results_dir + '/' + args.save_name  + '/scores.txt', 'w') as f:
        f.write("MAE: %.2f (+- %.2f)\n" % (mean(test_maes), np.std(test_maes, ddof=1)))

        test_rs = np.array(test_rs)
        task_means = np.mean(test_rs,0)
        task_stdevs = np.std(test_rs,0,ddof=1)
        fold_means = np.mean(test_rs,1)
        fold_stdevs = np.std(test_rs,1,ddof=1)

        test_all_maes = np.array(test_all_maes)
        task_mae_means, task_mae_stdevs = np.mean(test_all_maes,0), np.std(test_all_maes,0,ddof=1)
        fold_mae_means, fold_mae_stdevs = np.mean(test_all_maes,1), np.std(test_all_maes,1,ddof=1)

        f.write("\nAll Task Rs (averaged across all folds):\n")
        for i, item in enumerate(task_means):
            f.write("R%d.: %.2f (+- %.2f)\n" % (i, item, task_stdevs[i]))

        f.write("\nAll Task MAEs (averaged across all folds):\n")
        for i, item in enumerate(task_mae_means):
            f.write("R%d.: %.2f (+- %.2f)\n" % (i, item, task_mae_stdevs[i]))

        f.write("\nAll Fold Rs (averaged across all tasks):\n")
        for i, item in enumerate(fold_means):
            f.write("Fold %d: %.2f (+- %.2f)\n" % (i, item, fold_stdevs[i]))

        f.write("\nAll Fold MAEs (averaged across all tasks):\n")
        for i, item in enumerate(fold_mae_means):
            f.write("Fold %d: %.2f (+- %.2f)\n" % (i, item, fold_mae_stdevs[i]))

        f.write("\nAll Fold MAEs:\n")
        for item in test_maes:
            f.write("%2f," % (item))

        f.write("\n\nEpoch: %d" % (checkpoint['training_epoch']))
