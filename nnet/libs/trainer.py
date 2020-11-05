# wujian@2018

import os
import sys
import time
import subprocess
import pdb

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from .utils import get_logger

from itertools import combinations

import pdb

def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=6):
        #Nvidia smi call
        freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes"| grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
        if len(freeGpu) == 0: # if gpu not aviable use cpu
            raise RuntimeError("CUDA device unavailable...exist")
        self.device = th.device('cuda:'+freeGpu.decode().strip())
        
        #init tensorboard summary writer
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter()

        self.gpuid = (int(freeGpu.decode().strip()), )

        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # check if save is OK
            self.save_checkpoint(best=False)
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0
            # make sure not inf
            self.scheduler.best = best_loss
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                #call train
                tr = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                    tr["loss"], tr["cost"], tr["batches"])
                #call eval
                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                    cv["loss"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                
                #Tensorboard
                self.writer.add_scalar("Train", tr["loss"], self.cur_epoch)
                self.writer.add_scalar("CrossValidation", cv["loss"], self.cur_epoch)
                self.writer.flush()
                
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, egs):
        # spks x n x S
        ests = th.nn.parallel.data_parallel(
            self.nnet, egs["mix"], device_ids=self.gpuid)
        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = egs["mix"].size(0)
        sisnr_mat = th.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = th.max(sisnr_mat, dim=0)
        # si-snr
        return -th.sum(max_perutt) / N

class MixtureOfMixturesTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MixtureOfMixturesTrainer, self).__init__(*args, **kwargs)
        self.combs = None

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def genCombinations(self, N): # tohle generovat tak, aby byla pro 4 jen 1/3 2/2
        """
        Arguments:
        N: number of outputs
        Return:
        all: generated combinations
        """
        all = []
        m = [i for i in range(N)]
        for i in range(int(N/2),N):
            #pdb.set_trace()
            left = list(combinations(m, i))
            for eachLeft in left:
                eachLeft = sorted(eachLeft) #sort eachLeft
                x = [i for i in m if i not in eachLeft]
                for eachRight in list(combinations(x,N-len(eachLeft))): # where N-len(eachLeft)
                    newOne = (eachLeft, sorted(eachRight))
                    newOne2 = (sorted(eachRight), eachLeft)
                    #pdb.set_trace()
                    if (newOne not in all) and (newOne2 not in all):   #duplicity check, check if this pair exists in all, this is possible because eachLeft and newOne are sorted
                        all.append(newOne)
        return all

    def computePIT_loss(self, ests, refs):
        '''PIT objective function
        '''
        #create zero references to make len(refs) == len(ests)
        zero_refs = refs.copy()
        if len(refs) > len(ests):
                    raise RuntimeError("There are more references then separs")
        elif len(ests) > len(refs):
            for i in range(len(ests) - len(refs)):
                    zero_refs.append(th.zeros_like(refs[0]))
                    #zero_refs.append(th.randn_like(refs[0]))

        #count loss of each combination for each speech in batch and choose the best combination

        #pro kazdou promluvu z batche si pamatovat nejlepsi loss a nejlepsi kombinaci

        num_spks = len(zero_refs)

        def zero_sisnr_loss(permute):
            # for one permute
            snrList = []
            for s, t in enumerate(permute):
                #try different outputs of nn to zero refs
                snrList.append(self.sisnr(ests[t], zero_refs[s]))#th.reshape(self.sisnr(ests[s], zero_refs[t]),(1,32))
            snrSum = th.sum(th.stack(snrList), dim=0)
            return snrSum
        
        bestSnr = None # list of best SNR
        bestPerm = None # list of best permutation
        for p in permutations(range(num_spks)):
            if bestSnr is None:
                bestSnr = zero_sisnr_loss(p)
                bestPerm = [p for i in range(len(bestSnr))]
            else:
                newSnr = zero_sisnr_loss(p)
                # for each speech in batch
                for i in range(len(bestSnr)):
                    # if loss with permutation p is better then bestSnr, save newSnr to bestSnr, and save p to bestPerm
                    if bestSnr[i] < newSnr[i]:
                        bestSnr[i] = newSnr[i]
                        bestPerm[i] = p
        
        i = 0
        returnLoss = []
        for speechPerm in bestPerm:
            # count snr on bestPermutation
            snrLoss = 0
            for s, t in enumerate(speechPerm[:len(refs)]):
                snrLoss += self.sisnr(ests[t], refs[s])[i]
            returnLoss.append(snrLoss / len(refs))
            i += 1
        return th.stack(returnLoss)

    def computeMIXOFMIX_loss(self, ests, refs):
        '''Mix of mix objective function
        '''
        num_spks = len(refs)
        num_ests = len(ests) # number of estimates
        loss = None
        if self.combs is None: # if combinations are not generated, generate them and remember them
            self.combs = self.genCombinations(num_ests)
            print("Combinations generated", self.combs)
        #for each combination in combs
        for comb in self.combs:
            #generate mix of outputs
            first = None # first mix of estimates
            for out in comb[0]:
                if first is None:
                    first = ests[out]
                else:    
                    first += ests[out]
            second = None # second mix of estimates
            for out in comb[1]:
                if second is None:
                    second = ests[out]
                else:    
                    second += ests[out]
            #try both combinations
            if first is None or second is None:
                raise RuntimeError("First or Second is None in loss function.")
            #firstLoss is array of size N -> means batch size
            firstLoss = (self.sisnr(first, refs[0]) + self.sisnr(second, refs[1]))/2 # u sissnr chci vetsi hodnotu, tudiz bud obratit znamenko, nebo udelat maximum a pak vratit minus hodnotu
            secondLoss = (self.sisnr(second, refs[0]) + self.sisnr(first, refs[1]))/2
            #compare loss from both permutation
            stackedLoss = th.stack([firstLoss, secondLoss])
            newLoss = th.max(stackedLoss, dim=0).values
            #compare with losses from other combinations        
            if loss is None:
                loss = newLoss
            else:
                stackedLoss = th.stack([newLoss, loss])
                loss = th.max(stackedLoss, dim=0).values
        return loss

    def compute_loss(self, egs): # data jsou v batchi, to jest druha shape, spocitat pro kazde dato zvlast, vybirat max
        # spks x n x S
        ests = th.nn.parallel.data_parallel(
            self.nnet, egs["mix"], device_ids=self.gpuid) #get estimated output
        # spks x n x S
        refs = egs["ref"]
        # n x S
        known = egs["known"]
        #separate known and uknown
        #knownEsts = [x[known] for x in ests]
        knownEsts = [x[th.where(known)] for x in ests]
        knownRefs = [x[th.where(known)] for x in refs]
        #print("Known?", known, len(knownEsts[0]))
        unknownEsts = [x[th.where(~known)] for x in ests]
        unknownRefs = [x[th.where(~known)] for x in refs]

        #call pit loss for known
        if knownEsts[0].size(0) > 0: #control if there are some knowns
            pitloss = self.computePIT_loss(knownEsts, knownRefs)
        else:
            pitloss = None
        #call mixofmix loss for uknown
        if unknownEsts[0].size(0) > 0: #control if there are some uknowns
            mixofmixLoss = self.computeMIXOFMIX_loss(unknownEsts, unknownRefs)
        else:
            mixofmixLoss = None

        if pitloss is None:
            loss = mixofmixLoss
        elif mixofmixLoss is None:
            loss = pitloss
        else:
            loss = th.cat((pitloss, mixofmixLoss))
        return -(th.sum(loss)/refs[0].size(0)) # divide by batch