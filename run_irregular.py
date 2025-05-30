import torch.utils.data as Data
import torch.nn.init
import numpy as np
import tensorflow as tf
import neptune.new as neptune
import random
import argparse

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kovae import KoVAE
import torch.optim as optim
import logging
from utils.utils_data import create_timeDataset_irregular, inverse_MinMaxScaler, decode_categorical_from_generated
from viz.getter import Getter
from viz.visualizer_tabular import TabularVisualizer

def define_args():
    parser = argparse.ArgumentParser(description="KoVAE")

    # general
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--pinv_solver', type=bool, default=False)
    parser.add_argument('--neptune', default='debug', help='async runs as usual, debug prevents logging')
    parser.add_argument('--tag', default='sine, alpha_beta_sens')

    # data
    parser.add_argument("--dataset", default='stock')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--seq_len', type=int, default=24, metavar='N')
    parser.add_argument('--inp_dim', type=int, default=6, metavar='N')
    parser.add_argument('--missing_value', type=float, default=0.3)

    # model
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=16, help='dimension of the continuous latent space')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='the hidden dimension of the output decoder lstm')

    # loss params
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--w_rec', type=float, default=1.)
    parser.add_argument('--w_kl', type=float, default=.1)
    parser.add_argument('--w_pred_prior', type=float, default=0.0005)

    return parser

def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('cuda is available')
    else:
        device = torch.device("cpu")
    return device

def agg_losses(LOSSES, losses):
    if not LOSSES:
        LOSSES = [[] for _ in range(len(losses))]
    for jj, loss in enumerate(losses):
        LOSSES[jj].append(loss.item())
    return LOSSES

def log_losses(epoch, losses_tr, names):
    losses_avg_tr = []

    for loss in losses_tr:
        losses_avg_tr.append(np.mean(loss))

    loss_str_tr = 'Epoch {}, TRAIN: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_tr):
        loss_str_tr += '{}={:.3e}, \t'.format(names[jj], loss)
    logging.info(loss_str_tr)

    logging.info('#'*30)
    return losses_avg_tr[0]


parser = define_args()
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args):

    args.device = set_seed_device(args.seed)
    args.log_dir = './logs'

    name = 'KOVAEIrreg-{}_bs={}-rnn_size={}-z_dim={}-lr={}-n_layers={}=' \
           '-weight:kl={}-pred={}-w_decay={}-seed={}'.format(
        args.epochs, args.batch_size, args.hidden_dim, args.z_dim, args.lr, args.num_layers,
        args.w_kl, args.w_pred_prior, args.weight_decay, args.seed)

    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)

    if args.dataset == 'EV':
        dataset, min_data, max_data = create_timeDataset_irregular(args.dataset, args.seq_len, args.missing_value, return_minmax=True)
    else:
        dataset = create_timeDataset_irregular(args.dataset, args.seq_len, args.missing_value)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   worker_init_fn=seed_worker, generator=g)

    logging.info(args.dataset + ' dataset is ready.')

    disc_dim = None
    if args.dataset == 'EV':
        disc_dim = [2, 3]
        latent_spec = {'cont': args.z_dim, 'disc': disc_dim}
    else:
        latent_spec = {'cont': args.z_dim}
    # create model
    model = KoVAE(args, latent_spec).to(device=args.device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    tf.io.gfile.makedirs(os.path.dirname(args.log_dir))

    params_num = sum(param.numel() for param in model.parameters())

    logging.info(args)
    logging.info("number of model parameters: {}".format(params_num))
    print("number of model parameters: {}".format(params_num))

    logging.info("Starting training loop at step %d." % (0,))

    for epoch in range(0, args.epochs):
        logging.info("Running Epoch : {}".format(epoch + 1))

        model.train()
        losses_agg_tr = []
        for i, data in enumerate(train_loader, 1):

            x = data['data'].to(args.device).float()
            train_coeffs = data['inter']  # .to(device)
            time = torch.FloatTensor(list(range(args.seq_len))).to(args.device)
            final_index = (torch.ones(x.shape[0]) * (args.seq_len-1)).to(args.device).float()
            x = x[:, :, :-1]

            optimizer.zero_grad()
            x_rec, z_dist, z_prior_dist, z_prior_sample  = model(train_coeffs, time, final_index)

            x_no_nan = x[~torch.isnan(x)]
            x_rec_no_nan = x_rec[~torch.isnan(x)]

            losses = model.loss(x_no_nan, x_rec_no_nan, z_dist, z_prior_dist, z_prior_sample)  
            losses[0].backward()
            optimizer.step()

            losses_agg_tr = agg_losses(losses_agg_tr, losses)

        log_losses(epoch, losses_agg_tr, model.names)

    logging.info("Training is complete")

    
    
    ########## OUTPUTS AND LATENT TRAVERSAL ##########
    
    
    # generate datasets:
    args.device = set_seed_device(args.seed)
    getter = Getter(model)
    generated_data = getter.get_generated_data(train_loader)

    if args.dataset == 'EV':
        # De-normalize the generated data
        generated_data_denormalized = inverse_MinMaxScaler(generated_data, min_data, max_data)
        
        ''' ##########################################################################
        #POST-PROCESSING in order to handle categorical features' output appropriately
        generated_data_denormalized, event_cat, charge_cat = decode_categorical_from_generated(generated_data_denormalized)
        

        print("Decoded event categories:", event_cat)
        print("Decoded charge_mode categories:", charge_cat)
        #############################################################################'''


    # save data in torch format in the directory ./Generated_data if not exist
    output_dir = './Generated_data'
    file_path = os.path.join(output_dir, f'{args.dataset}_generated_data.pt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == 'EV':
        torch.save(torch.from_numpy(generated_data_denormalized), file_path)
    else:
        torch.save(torch.from_numpy(generated_data), file_path)
    logging.info(f"Generated data saved to {file_path}")
    
    #Reconstruct data
    recon = getter.get_reconstructed_data(train_loader, time, final_index)
    if args.dataset == 'EV':
        # De-normalize the reconstructed data
        recon = inverse_MinMaxScaler(recon, min_data, max_data)
    file_path = os.path.join(output_dir, f'{args.dataset}_reconstructed_data.pt')
    torch.save(torch.from_numpy(recon), file_path)
    logging.info(f"Reconstructed data saved to {file_path}")
    
    # Embedded original data -> data reference from reconstruction 
    original = getter.get_original_data(train_loader)
    if args.dataset == 'EV':
        # De-normalize the original data
        original = inverse_MinMaxScaler(original, min_data, max_data)
    file_path = os.path.join(output_dir, f'{args.dataset}_original_data.pt')
    torch.save(torch.from_numpy(original), file_path)
    logging.info(f"Original data saved to {file_path}")
    
    

    #PLOTS -> Visualization of reconstruction, generated data and latent traversals
    # After training the model
    visualizer = TabularVisualizer(model)
    
    visualizer.reconstructions(original, recon)
    visualizer.samples(generated_data_denormalized)
    for cont_idx in range(args.z_dim):
        visualizer.latent_traversal(cont_idx=cont_idx)
    for disc_idx in range(len(disc_dim)):
        visualizer.latent_traversal(disc_idx=disc_idx)




    ori_data = list()
    for data in train_loader:
        ori_data.append(data['original_data'].detach().cpu().numpy())
    ori_data = np.vstack(ori_data)

    from metrics.discriminative_torch import discriminative_score_metrics
    # deterministic eval
    args.device = set_seed_device(args.seed)
    disc_res = []
    for ii in range(10):
        dsc = discriminative_score_metrics(ori_data, generated_data, args)
        disc_res.append(dsc)
    disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)

    print('test/disc_mean: ', disc_mean)
    print('test/disc_std: ', disc_std)


if __name__ == '__main__':
    main(args)
