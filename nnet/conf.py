fs = 8000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
num_spks = 2 # number of speakers
num_outputs = 4 # number of outputs
knownPercent = 50


# network configure
nnet_conf = {
    "L": 40,
    "N": 128,
    "X": 7,
    "R": 3,
    "B": 128,
    "H": 192,
    "P": 3,
    "norm": "BN",
    "num_spks": num_outputs,
    "non_linear": "relu"
}

# data configure:
train_dir = "../min_dataset/tr/"
dev_dir = "../min_dataset/cv/"

train_data = {
    "mix_scp":
    train_dir + "mix.scp",
    "ref_scp":
    [train_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate":
    fs,
    "knownPercent":
    knownPercent,
}

dev_data = {
    "mix_scp": dev_dir + "mix.scp",
    "ref_scp":
    [dev_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
    "knownPercent":
    knownPercent,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 8,
    "factor": 0.5,
    "logging_period": 200,  # batch number
    "no_impr":100 # bez omezeni
}
