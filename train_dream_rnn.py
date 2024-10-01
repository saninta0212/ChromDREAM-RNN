import argparse
import copy
import os
import shutil
import os
import sys
import numpy as np

fpx = ""
args = argparse.Namespace(
    inputlen = 2114,
    outputlen = 1000,
    ibam="/scratch/st-cdeboer-1/sambina/chrom_dreamnet/atac-seq_outputs/chrombpnet_tutorial_k562/data/downloads/merged.bam",
    data_type="ATAC",
    genome="/scratch/st-cdeboer-1/sambina/chrom_dreamnet/atac-seq_outputs/chrombpnet_tutorial_k562/data/downloads/hg38.fa",
    # peaks = "/scratch/st-cdeboer-1/sambina/transfer_learning_leakage/data_balanced_leakage/peaks_downsampled_no_blacklist.bed",
    max_jitter = 500,
    negative_sampling_ratio = 0.1,
    outlier_threshold = 0.9999,
    filters = 512,
    n_dilation_layers = 8,
    architecture_from_file = None,
    seed = 1234,
    learning_rate = 0.001,
    batch_size = 64,
    early_stop = 20,
    epochs = 80,
    trackables = ['logcount_predictions_loss', 'loss', 'logits_profile_predictions_loss', 
                  'val_logcount_predictions_loss', 'val_loss', 'val_logits_profile_predictions_loss'],
    model_name = "k562_leakage",
)

args.fold_num = int(sys.argv[1])
args.peaks = f"/scratch/st-cdeboer-1/sambina/transfer_learning_leakage/5_folds/peaks_leakage/leakage_{args.fold_num}.bed"
args.chr_fold_path =  f"/scratch/st-cdeboer-1/sambina/chrom_dreamnet/atac-seq_outputs/chrombpnet_tutorial_k562/data/splits/leakage_folds/fold_{args.fold_num}.json"
args.output_dir = f"/scratch/st-cdeboer-1/sambina/transfer_learning_leakage/5_folds/leakage/chromdreamrnn/chromdreamrnn_k562_{args.fold_num}/"
args.bias_model_path = f"/scratch/st-cdeboer-1/sambina/transfer_learning_leakage/5_folds/leakage/bias_models/bias_model_{args.fold_num}/models/k562_leakage_5_fold_bias.h5"
args.nonpeaks = f"/scratch/st-cdeboer-1/sambina/transfer_learning_leakage/5_folds/negatives/leakage/output_fold_{args.fold_num}_negatives.bed"
os.makedirs(args.output_dir, exist_ok=False)
os.makedirs(os.path.join(args.output_dir, "auxiliary"), exist_ok=False)
## Waiting for chrombpnet run to finish
shutil.copyfile(f"/scratch/st-cdeboer-1/sambina/transfer_learning_leakage/5_folds/leakage/chrombpnet/chrombpnet_model_{args.fold_num}/auxiliary/data_unstranded.bw",
                os.path.join(args.output_dir, "auxiliary/data_unstranded.bw"))

args.bigwig = os.path.join(args.output_dir,"auxiliary/data_unstranded.bw")

# fetch hyperparameters for training
from hyperparameters import find_chrombpnet_hyperparams
args.output_prefix = os.path.join(args.output_dir,"auxiliary/")
find_chrombpnet_hyperparams.main(args)

# separating models from logs
os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=False)
os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=False)

os.rename(os.path.join(args.output_dir,"auxiliary/bias_model_scaled.h5"),
          os.path.join(args.output_dir,"models/bias_model_scaled.h5"))
os.rename(os.path.join(args.output_dir,"auxiliary/chrombpnet_model_params.tsv"),
          os.path.join(args.output_dir,"logs/chrombpnet_model_params.tsv"))
os.rename(os.path.join(args.output_dir,"auxiliary/chrombpnet_data_params.tsv"),
          os.path.join(args.output_dir,"logs/chrombpnet_data_params.tsv"))

params = open(os.path.join(args.output_dir,"logs/chrombpnet_model_params.tsv")).read()
params = params.replace(os.path.join(args.output_dir,"auxiliary/bias_model_scaled.h5"),
                        os.path.join(args.output_dir,"models/bias_model_scaled.h5"))
with open(os.path.join(args.output_dir,"logs/chrombpnet_model_params.tsv"),"w") as f:
    f.write(params)

# get model architecture path and train chromBPNet model
# import chrombpnet.training.models.autosome_with_bias_model as autosome_with_bias_model
import dream_paper_bhi_1_trainer_320_with_bias_model as dream_paper_bhi_1_trainer_320_with_bias_model
# import chrombpnet.training.models.unlockdna_with_bias_model as unlockdna_with_bias_model

from train_2_00005 import *
args_copy = copy.deepcopy(args)

if args_copy.architecture_from_file is None:
    args_copy.architecture_from_file = 	dream_paper_bhi_1_trainer_320_with_bias_model.__file__

args_copy.peaks = os.path.join(args.output_dir,"auxiliary/filtered.peaks.bed")
args_copy.nonpeaks = os.path.join(args.output_dir,"auxiliary/filtered.nonpeaks.bed")
args_copy.output_prefix = os.path.join(args.output_dir,f"models/{args.model_name}")
args_copy.params = os.path.join(args.output_dir,"logs/chrombpnet_model_params.tsv")

parameters = get_model_param_dict(args_copy)
print(parameters)
np.random.seed(args_copy.seed)

# get model architecture to load
model, architecture_module=get_model(args_copy, parameters)

# initialize generators to load data
train_generator = initializers.initialize_generators(args_copy, "train", parameters, return_coords=False)
valid_generator = initializers.initialize_generators(args_copy, "valid", parameters, return_coords=False)

# Print out a few model layers


# train the model using the generators
fit_and_evaluate(model, train_generator, valid_generator, args_copy, architecture_module, parameters)

model, architecture_module=get_model(args_copy, parameters)

# make predictions with trained chrombpnet model
import predict as predict
args_copy = copy.deepcopy(args)

os.makedirs(os.path.join(args.output_dir, "evaluation"), exist_ok=False)

args_copy.output_prefix = os.path.join(args_copy.output_dir,f"evaluation/{args_copy.model_name}")
args_copy.model_h5 = os.path.join(args_copy.output_dir,f"models/{args_copy.model_name}.h5")
print(args_copy.model_h5)
# load_status = model.load_weights(args_copy.model_h5)
# args_copy.model = model
args_copy.nonpeaks = "None"
# h5_file = f"{args.output_dir}/models/dreamrnn_nobias.h5"
# model.save(h5_file)
predict.main(args_copy)
