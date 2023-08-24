# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from collections import Counter

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import copy
from tqdm import tqdm

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.lib.query import Q

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="ViT_RB_small")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    args = parser.parse_args()
    args.save_best_model = True
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        js = json.loads(args.hparams)
        js["test_env"] = args.test_envs
        # print(args.hparams)
        hparams.update(js)

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('device:', device)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # ### DEBUGGING    
    # #     print(dataset)

    # # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # # each in-split except the test envs, and evaluate on all splits.

    # # To allow unsupervised domain adaptation experiments, we split each test
    # # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # # by collect_results.py to compute classification accuracies.  The
    # # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # # samples in 'uda-split' are passed to the algorithm at training time if
    # # args.task == "domain_adaptation". If we are interested in comparing
    # # domain generalization and domain adaptation results, then domain
    # # generalization algorithms should create the same 'uda-splits', which will
    # # be discared at training.
    # in_splits = []
    # out_splits = []
    # uda_splits = []
    
    # counts = Counter()
    # classes = []
    # n_samples = 0
    # for env_i, env in enumerate(dataset):  # env is a domain
    #     uda = []
    #     print(env_i)
    #     print(args.test_envs)
    #     if args.test_envs[0] == 0:
    #         print('yesssssssss test env is 0',args.test_envs, env_i)
    #         # in_focal_weights = {0: 0.27407573548652386, 1: 2.7470213563132257, 2: 1.3340247452692868, 4: 9.369712460063898, 3: 7.63331598125976}
    #         in_focal_weights = {0: 0.5, 1: 1, 2: 1, 4: 1, 3:1}
    #         print(in_focal_weights)
    #     elif args.test_envs[0] == 1:
    #         print('yesssssssss test env is 1',args.test_envs, env_i)
    #         # in_focal_weights = {0: 0.39491968621591333, 3: 2.499290780141844, 2: 0.8437350359138068, 1: 1.6340030911901082, 4: 3.6965034965034964}
    #         in_focal_weights = {0: 0.5, 1: 1, 2: 1, 4: 1, 3:1}
    #         print(in_focal_weights)

    #     elif args.test_envs[0] == 2:
    #         print('yesssssssss test env is 2',args.test_envs, env_i)
    #         # in_focal_weights = {0: 0.2761950181591401, 3: 8.052192513368984, 2: 1.2997496763055676, 1: 2.740735347651984, 4: 8.268863261943986}
    #         in_focal_weights = {0: 0.5, 1: 1, 2: 1, 4: 1, 3:1}
    #         print(in_focal_weights)

    #     elif args.test_envs[0] == 3:
    #         print('yesssssssss test env is 3',args.test_envs, env_i)
    #         # in_focal_weights = {0: 0.27647699780227897, 3: 7.422112047595439, 2: 1.302000347886589, 1: 2.7728097795888127, 4: 8.358682300390843}
    #         in_focal_weights = {0: 0.5, 1: 1, 2: 1, 4: 1, 3:1}
    #         print(in_focal_weights)

    #     uda = []
    #     out, in_ = misc.split_dataset(env,
    #         int(len(env)*args.holdout_fraction),
    #         misc.seed_hash(args.trial_seed, env_i))

    #     if env_i in args.test_envs:
    #         uda, in_ = misc.split_dataset(in_,
    #             int(len(in_)*args.uda_holdout_fraction),
    #             misc.seed_hash(args.trial_seed, env_i))

    #     if hparams['class_balanced']:
    #         print('if')
    #         in_weights = misc.make_weights_for_balanced_classes(in_)
    #         out_weights = misc.make_weights_for_balanced_classes(out)
    #         if uda is not None:
    #             uda_weights = misc.make_weights_for_balanced_classes(uda)
    #     else:
    #         print('if not class balanced')
    #         in_weights, out_weights, uda_weights = None, None, None
    #     in_splits.append((in_, in_weights))
    #     out_splits.append((out, out_weights))
    #     if len(uda):
    #         uda_splits.append((uda, uda_weights))


    #     # if env_i not in args.test_envs:
    #     #     out, in_ = misc.split_dataset(env,
    #     #                                 int(len(env) * args.holdout_fraction),
    #     #                                 misc.seed_hash(args.trial_seed, env_i))
    #     #     print('not in test', len(out), len(in_))
    #     #     for _, y in tqdm(in_):
    #     #             y = int(y)
    #     #             counts[y] += 1
    #     #     n_samples+=len(in_)
    #     #     print('in make weights for balanced class', counts)

    #     # if env_i in args.test_envs:
    #     #     # print('in test')
    #     #     out, in_ = misc.split_dataset(env,
    #     #                                 int(len(env) * args.holdout_fraction),
    #     #                                 misc.seed_hash(args.trial_seed, env_i))
    #     #     uda, in_ = misc.split_dataset(in_,
    #     #                                   int(len(in_) * args.uda_holdout_fraction),
    #     #                                   misc.seed_hash(args.trial_seed, env_i))
    #     #     # print('in args.test_env', len(uda), len(in_))

                
    #     # if hparams['class_balanced']:
    #     #     in_weights = misc.make_weights_for_balanced_classes(in_)
    #     #     out_weights = misc.make_weights_for_balanced_classes(out)
    #     #     if uda is not None:
    #     #         uda_weights = misc.make_weights_for_balanced_classes(uda)
    #     # else:
    #     #     in_weights, out_weights, uda_weights = None, None, None
    #     # in_splits.append((in_, in_weights))
    #     # out_splits.append((out, out_weights))
    #     # if len(uda):
    #     #     uda_splits.append((uda, uda_weights))
    # # exit()
    # # print('out_of_for_loop', counts)
    # # print(n_samples)
    # # in_focal_weights = misc.make_weights_for_balanced_classes_for_focal(counts, n_samples)            
    # # print('weights', in_focal_weights)
    in_focal_weights = {0: 0.5, 1: 1, 2: 1, 4: 1, 3:1}
    in_splits = []
    out_splits = []
    uda_splits = []
    print(args.test_envs[0])
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            print('class_balanced')
            print(env_i)
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            print(in_weights, out_weights)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams, in_focal_weights)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename, filefamiliy=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
            # "model_dict": copy.deepcopy(algorithm).cpu().state_dict(),
        }
        
        if filefamiliy is not None:
            for fname in os.listdir(args.output_dir):
                if (filefamiliy.lower() in fname) and (fname.endswith('.pkl')):
                    os.remove(os.path.join(args.output_dir, fname))
        
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    best_val_acc = -np.Inf # Additional code to save best models
    best_val_f1_w = -np.Inf # Additional code to save best models
    best_val_f1_m = -np.Inf # Additional code to save best models
    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            
            temp_acc, temp_f1_w, temp_f1_m, temp_count = 0.0, 0.0, 0.0, 0.0
            evals = zip(eval_loader_names, eval_loaders, eval_weights) # Additional code to save best models
            for name, loader, weights in evals:
                acc, macro_avg, weighted_avg = misc.f1(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                results[name+'_f1m'] = macro_avg['f1-score']
                results[name+'_f1w'] = weighted_avg['f1-score']
                
                # Additional code to save best models
                if int(name[3]) not in args.test_envs and "out" in name:
                    # print('-----'*30)
                    # print(name, args.test_envs)
                    temp_acc += acc
                    temp_f1_w += weighted_avg['f1-score']
                    temp_f1_m += macro_avg['f1-score']
                    temp_count += 1.0
            
            ###########################################################################
            
            # Additional code to save best models
            val_acc = temp_acc / temp_count
            val_f1_w = temp_f1_w / temp_count
            val_f1_m = temp_f1_m / temp_count
            
            # print(val_acc, val_f1_w, val_f1_m)
            
            if val_acc >= best_val_acc:
                save_checkpoint(f"model_best_acc_step_{step}.pkl", "model_best_acc")
                best_val_acc = copy.deepcopy(val_acc)
            if val_f1_w >= best_val_f1_w:
                save_checkpoint(f"model_best_f1_w_step_{step}.pkl", "model_best_f1_w")
                best_val_f1_w = copy.deepcopy(val_f1_w)
            if val_f1_m >= best_val_f1_m:
                save_checkpoint(f"model_best_f1_m_step_{step}.pkl", "model_best_f1_m")
                best_val_f1_m = copy.deepcopy(val_f1_m)
            
            ###########################################################################
            
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')




#previous code
    # def save_checkpoint(filename):
    #     if args.skip_model_save:
    #         return
    #     save_dict = {
    #         "args": vars(args),
    #         "model_input_shape": dataset.input_shape,
    #         "model_num_classes": dataset.num_classes,
    #         "model_num_domains": len(dataset) - len(args.test_envs),
    #         "model_hparams": hparams,
    #         "model_dict": algorithm.cpu().state_dict()
    #     }
    #     torch.save(save_dict, os.path.join(args.output_dir, filename))


    # def save_checkpoint_best(filename, algo):
    #     if args.skip_model_save:
    #         return
    #     save_dict = {
    #         "args": vars(args),
    #         "model_input_shape": dataset.input_shape,
    #         "model_num_classes": dataset.num_classes,
    #         "model_num_domains": len(dataset) - len(args.test_envs),
    #         "model_hparams": hparams,
    #         "model_dict": algo.state_dict()
    #     }
    #     torch.save(save_dict, os.path.join(args.output_dir, filename))



    # last_results_keys = None
    # best_val_acc = 0
    # for step in range(start_step, n_steps):
    #     step_start_time = time.time()
    #     minibatches_device = [(x.to(device), y.to(device))
    #                           for x, y in next(train_minibatches_iterator)]
    #     if args.task == "domain_adaptation":
    #         uda_device = [x.to(device)
    #                       for x, _ in next(uda_minibatches_iterator)]
    #     else:
    #         uda_device = None
    #     step_vals = algorithm.update(minibatches_device, uda_device)
    #     checkpoint_vals['step_time'].append(time.time() - step_start_time)

    #     for key, val in step_vals.items():
    #         checkpoint_vals[key].append(val)

    #     if (step % checkpoint_freq == 0) or (step == n_steps - 1):
    #         results = {
    #             'step': step,
    #             'epoch': step / steps_per_epoch,
    #         }

    #         for key, val in checkpoint_vals.items():
    #             results[key] = np.mean(val)

    #         evals = zip(eval_loader_names, eval_loaders, eval_weights)
    #         temp_acc = 0
    #         temp_count = 0
    #         for name, loader, weights in evals:
    #             acc = misc.accuracy(algorithm, loader, weights, device)
    #             if args.save_best_model:
    #                 if int(name[3]) not in args.test_envs and "out" in name:
    #                     temp_acc += acc
    #                     temp_count += 1
    #             results[name + '_acc'] = acc
    #         if args.save_best_model:
    #             val_acc = temp_acc / (temp_count * 1.0)
    #             if val_acc >= best_val_acc:
    #                 # model_save = algorithm.detach().clone()  # clone
    #                 model_save = copy.deepcopy(algorithm)  # clone
    #                 if (args.save_best_model):
    #                     save_checkpoint('IID_best.pkl')
    #                     algorithm.to(device)
    #                 best_val_acc = val_acc
    #                 print("Best model upto now")
    #         results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

    #         results_keys = sorted(results.keys())
    #         if results_keys != last_results_keys:
    #             misc.print_row(results_keys, colwidth=12)
    #             last_results_keys = results_keys
    #         misc.print_row([results[key] for key in results_keys],
    #                        colwidth=12)

    #         results.update({
    #             'hparams': hparams,
    #             'args': vars(args)
    #         })

    #         epochs_path = os.path.join(args.output_dir, 'results.jsonl')
    #         with open(epochs_path, 'a') as f:
    #             f.write(json.dumps(results, sort_keys=True) + "\n")

    #         algorithm_dict = algorithm.state_dict()
    #         start_step = step + 1
    #         checkpoint_vals = collections.defaultdict(lambda: [])

    #         # records = []
    #         # with open(epochs_path, 'r') as f:
    #         #     for line in f:
    #         #         records.append(json.loads(line[:-1]))
    #         # records = Q(records)
    #         # scores = records.map(model_selection.IIDAccuracySelectionMethod._step_acc)
    #         # if scores[-1] == scores.argmax('val_acc'):
    #         #     save_checkpoint('IID_best.pkl')
    #         #     algorithm.to(device)

    #         if args.save_model_every_checkpoint:
    #             save_checkpoint(f'model_step{step}.pkl')

    # save_checkpoint('model.pkl')
    # if (args.save_best_model):
    #     save_checkpoint_best('IID_best.pkl', model_save)
    # with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    #     f.write('done')






















































# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# import argparse
# import collections
# import copy
# import json
# import os
# import random
# import sys
# import time
# import uuid

# import numpy as np
# import PIL
# import torch
# import torchvision
# import torch.utils.data

# from domainbed import datasets
# from domainbed import hparams_registry
# from domainbed import algorithms
# from domainbed.lib import misc
# from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Domain generalization')
#     parser.add_argument('--data_dir', type=str)
#     parser.add_argument('--dataset', type=str, default="RotatedMNIST")
#     parser.add_argument('--algorithm', type=str, default="ERM")
#     parser.add_argument('--task', type=str, default="domain_generalization",
#         choices=["domain_generalization", "domain_adaptation"])
#     parser.add_argument('--hparams', type=str,
#         help='JSON-serialized hparams dict')
#     parser.add_argument('--hparams_seed', type=int, default=0,
#         help='Seed for random hparams (0 means "default hparams")')
#     parser.add_argument('--trial_seed', type=int, default=0,
#         help='Trial number (used for seeding split_dataset and '
#         'random_hparams).')
#     parser.add_argument('--seed', type=int, default=0,
#         help='Seed for everything else')
#     parser.add_argument('--steps', type=int, default=None,
#         help='Number of steps. Default is dataset-dependent.')
#     parser.add_argument('--checkpoint_freq', type=int, default=None,
#         help='Checkpoint every N steps. Default is dataset-dependent.')
#     parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
#     parser.add_argument('--output_dir', type=str, default="train_output")
#     parser.add_argument('--holdout_fraction', type=float, default=0.2)
#     parser.add_argument('--uda_holdout_fraction', type=float, default=0,
#         help="For domain adaptation, % of test to use unlabeled for training.")
#     parser.add_argument('--skip_model_save', action='store_true')
#     parser.add_argument('--save_model_every_checkpoint', action='store_true')
#     args = parser.parse_args()

#     # If we ever want to implement checkpointing, just persist these values
#     # every once in a while, and then load them from disk here.
#     start_step = 0
#     algorithm_dict = None

#     os.makedirs(args.output_dir, exist_ok=True)
#     sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
#     sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

#     print("Environment:")
#     print("\tPython: {}".format(sys.version.split(" ")[0]))
#     print("\tPyTorch: {}".format(torch.__version__))
#     print("\tTorchvision: {}".format(torchvision.__version__))
#     print("\tCUDA: {}".format(torch.version.cuda))
#     print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
#     print("\tNumPy: {}".format(np.__version__))
#     print("\tPIL: {}".format(PIL.__version__))

#     print('Args:')
#     for k, v in sorted(vars(args).items()):
#         print('\t{}: {}'.format(k, v))

#     if args.hparams_seed == 0:
#         hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
#     else:
#         hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
#             misc.seed_hash(args.hparams_seed, args.trial_seed))
#     if args.hparams:
#         hparams.update(json.loads(args.hparams))

#     print('HParams:')
#     for k, v in sorted(hparams.items()):
#         print('\t{}: {}'.format(k, v))

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"

#     if args.dataset in vars(datasets):
#         dataset = vars(datasets)[args.dataset](args.data_dir,
#             args.test_envs, hparams)
#     else:
#         raise NotImplementedError

#     # Split each env into an 'in-split' and an 'out-split'. We'll train on
#     # each in-split except the test envs, and evaluate on all splits.

#     # To allow unsupervised domain adaptation experiments, we split each test
#     # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
#     # by collect_results.py to compute classification accuracies.  The
#     # 'out-split' is used by the Oracle model selectino method. The unlabeled
#     # samples in 'uda-split' are passed to the algorithm at training time if
#     # args.task == "domain_adaptation". If we are interested in comparing
#     # domain generalization and domain adaptation results, then domain
#     # generalization algorithms should create the same 'uda-splits', which will
#     # be discared at training.
#     in_splits = []
#     out_splits = []
#     uda_splits = []
#     for env_i, env in enumerate(dataset):
#         uda = []

#         out, in_ = misc.split_dataset(env,
#             int(len(env)*args.holdout_fraction),
#             misc.seed_hash(args.trial_seed, env_i))

#         if env_i in args.test_envs:
#             uda, in_ = misc.split_dataset(in_,
#                 int(len(in_)*args.uda_holdout_fraction),
#                 misc.seed_hash(args.trial_seed, env_i))

#         if hparams['class_balanced']:
#             in_weights = misc.make_weights_for_balanced_classes(in_)
#             out_weights = misc.make_weights_for_balanced_classes(out)
#             if uda is not None:
#                 uda_weights = misc.make_weights_for_balanced_classes(uda)
#         else:
#             in_weights, out_weights, uda_weights = None, None, None
#         in_splits.append((in_, in_weights))
#         out_splits.append((out, out_weights))
#         if len(uda):
#             uda_splits.append((uda, uda_weights))

#     if args.task == "domain_adaptation" and len(uda_splits) == 0:
#         raise ValueError("Not enough unlabeled samples for domain adaptation.")

#     train_loaders = [InfiniteDataLoader(
#         dataset=env,
#         weights=env_weights,
#         batch_size=hparams['batch_size'],
#         num_workers=dataset.N_WORKERS)
#         for i, (env, env_weights) in enumerate(in_splits)
#         if i not in args.test_envs]

#     uda_loaders = [InfiniteDataLoader(
#         dataset=env,
#         weights=env_weights,
#         batch_size=hparams['batch_size'],
#         num_workers=dataset.N_WORKERS)
#         for i, (env, env_weights) in enumerate(uda_splits)]

#     eval_loaders = [FastDataLoader(
#         dataset=env,
#         batch_size=64,
#         num_workers=dataset.N_WORKERS)
#         for env, _ in (in_splits + out_splits + uda_splits)]
#     eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
#     eval_loader_names = ['env{}_in'.format(i)
#         for i in range(len(in_splits))]
#     eval_loader_names += ['env{}_out'.format(i)
#         for i in range(len(out_splits))]
#     eval_loader_names += ['env{}_uda'.format(i)
#         for i in range(len(uda_splits))]

#     algorithm_class = algorithms.get_algorithm_class(args.algorithm)
#     algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
#         len(dataset) - len(args.test_envs), hparams)

#     if algorithm_dict is not None:
#         algorithm.load_state_dict(algorithm_dict)

#     algorithm.to(device)

#     train_minibatches_iterator = zip(*train_loaders)
#     uda_minibatches_iterator = zip(*uda_loaders)
#     checkpoint_vals = collections.defaultdict(lambda: [])

#     steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

#     n_steps = args.steps or dataset.N_STEPS
#     checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

#     def save_checkpoint(filename, filefamiliy=None):
#         if args.skip_model_save:
#             return
#         save_dict = {
#             "args": vars(args),
#             "model_input_shape": dataset.input_shape,
#             "model_num_classes": dataset.num_classes,
#             "model_num_domains": len(dataset) - len(args.test_envs),
#             "model_hparams": hparams,
#             # "model_dict": algorithm.state_dict(),
#             "model_dict": copy.deepcopy(algorithm).cpu().state_dict(),
#         }
        
#         if filefamiliy is not None:
#             for fname in os.listdir(args.output_dir):
#                 if (filefamiliy.lower() in fname) and (fname.endswith('.pkl')):
#                     os.remove(os.path.join(args.output_dir, fname))
        
#         torch.save(save_dict, os.path.join(args.output_dir, filename))


#     best_val_acc = -np.Inf # Additional code to save best models
#     best_val_f1_w = -np.Inf # Additional code to save best models
#     best_val_f1_m = -np.Inf # Additional code to save best models
#     last_results_keys = None
#     for step in range(start_step, n_steps):
#         step_start_time = time.time()
#         minibatches_device = [(x.to(device), y.to(device))
#             for x,y in next(train_minibatches_iterator)]
#         if args.task == "domain_adaptation":
#             uda_device = [x.to(device)
#                 for x,_ in next(uda_minibatches_iterator)]
#         else:
#             uda_device = None
#         step_vals = algorithm.update(minibatches_device, uda_device)
#         checkpoint_vals['step_time'].append(time.time() - step_start_time)

#         for key, val in step_vals.items():
#             checkpoint_vals[key].append(val)

#         if (step % checkpoint_freq == 0) or (step == n_steps - 1):
#             results = {
#                 'step': step,
#                 'epoch': step / steps_per_epoch,
#             }

#             for key, val in checkpoint_vals.items():
#                 results[key] = np.mean(val)
            
#             temp_acc, temp_f1_w, temp_f1_m, temp_count = 0.0, 0.0, 0.0, 0.0
#             evals = zip(eval_loader_names, eval_loaders, eval_weights) # Additional code to save best models
#             for name, loader, weights in evals:
#                 acc, macro_avg, weighted_avg = misc.f1(algorithm, loader, weights, device)
#                 results[name+'_acc'] = acc
#                 results[name+'_f1m'] = macro_avg['f1-score']
#                 results[name+'_f1w'] = weighted_avg['f1-score']
                
#                 # Additional code to save best models
#                 if int(name[3]) not in args.test_envs and "out" in name:
#                     # print('-----'*30)
#                     # print(name, args.test_envs)
#                     temp_acc += acc
#                     temp_f1_w += weighted_avg['f1-score']
#                     temp_f1_m += macro_avg['f1-score']
#                     temp_count += 1.0
            
#             ###########################################################################
            
#             # Additional code to save best models
#             val_acc = temp_acc / temp_count
#             val_f1_w = temp_f1_w / temp_count
#             val_f1_m = temp_f1_m / temp_count
            
#             # print(val_acc, val_f1_w, val_f1_m)
            
#             if val_acc >= best_val_acc:
#                 save_checkpoint(f"model_best_acc_step_{step}.pkl", "model_best_acc")
#                 best_val_acc = copy.deepcopy(val_acc)
#             if val_f1_w >= best_val_f1_w:
#                 save_checkpoint(f"model_best_f1_w_step_{step}.pkl", "model_best_f1_w")
#                 best_val_f1_w = copy.deepcopy(val_f1_w)
#             if val_f1_m >= best_val_f1_m:
#                 save_checkpoint(f"model_best_f1_m_step_{step}.pkl", "model_best_f1_m")
#                 best_val_f1_m = copy.deepcopy(val_f1_m)
            
#             ###########################################################################
            
#             results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

#             results_keys = sorted(results.keys())
#             if results_keys != last_results_keys:
#                 misc.print_row(results_keys, colwidth=12)
#                 last_results_keys = results_keys
#             misc.print_row([results[key] for key in results_keys],
#                 colwidth=12)

#             results.update({
#                 'hparams': hparams,
#                 'args': vars(args)
#             })

#             epochs_path = os.path.join(args.output_dir, 'results.jsonl')
#             with open(epochs_path, 'a') as f:
#                 f.write(json.dumps(results, sort_keys=True) + "\n")

#             algorithm_dict = algorithm.state_dict()
#             start_step = step + 1
#             checkpoint_vals = collections.defaultdict(lambda: [])

#             if args.save_model_every_checkpoint:
#                 save_checkpoint(f'model_step{step}.pkl')

#     save_checkpoint('model.pkl')

#     with open(os.path.join(args.output_dir, 'done'), 'w') as f:
#         f.write('done')
