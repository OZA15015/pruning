#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    compression_scheduler.on_epoch_end(epoch)
    save_checkpoint()

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""
import sys
sys.path.append('/home/oza/pre-experiment/FBNet')
import traceback
import logging
from functools import partial
import distiller
from distiller.models import create_model
import distiller.apputils.image_classifier as classifier
import distiller.apputils as apputils
import parser
import os
import numpy as np
from ptq_lapq import image_classifier_ptq_lapq

#yaml関連
import yaml
#from pathlib import Path
from ruamel.yaml import YAML, add_constructor, resolver
from collections import OrderedDict
import os

#GA拡張工事
from deap import base
from deap import creator
from deap import tools
import random
random.seed(64) 

# Logger handle
msglogger = logging.getLogger()
max = 0

def write_yaml(yaml_list):
    add_constructor(resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))
    yaml = YAML()
    yaml.default_flow_style = False
     
    with open("simplenet_cifar.schedule_agp.yaml", "r") as yf: #yaml基礎ファイル
        data = yaml.load(yf)    # safe_load()を使う
    print(data)
    for i in range(2):
        data['pruners']['conv' + str(i+1) + '_pruner']['final_sparsity'] = yaml_list[i]
    
    #data['version'] = k
    if os.path.isfile("fruits.yaml"):
        os.remove("fruits.yaml")
    
    with open("fruits.yaml", "a") as yf:
        yaml.dump(data, yf)

        
def main(individual):
    # Parse arguments
    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    args.compress = 'fruits.yaml'
    
    idx = 0
    num = 5 #initial 5%
    yaml_list = []
    for i in range(int(len(individual) / 18)): #各ビット数/18
        num = 5
        for j in range(18): #長さ
            if(individual[idx] == 1):
                num += 5
            idx += 1
        yaml_list.append(num / 100)
    print("yaml_list")
    print(yaml_list)
    write_yaml(yaml_list)
    
    app = ClassifierCompressorSampleApp(args, script_dir=os.path.dirname(__file__))
    if app.handle_subapps():
        return
    init_knowledge_distillation(app.args, app.model, app.compression_scheduler)
    app.run_training_loop()
    # Finally run results on the test set
    # return top1, top5, losssesが来る

    loaded_array = np.load('/home/oza/pre-experiment/speeding/distiller/distiller/apputils/simple_gene.npz')
    accuracy = loaded_array['array_1']
    sparce = loaded_array['array_2']
    print("accuracy: " + str(accuracy))
    print("sparce: " + str(sparce))     
    accuracy /= 100
    sparce /= 100
    #return app.test()
    score = accuracy * sparce
    print("score: " + str(score))
    global max
    if(score > max):
        max = score
        print("max score: " + str(score))
        print("max individual: " + str(yaml_list))
    return accuracy * sparce

    
def handle_subapps(model, criterion, optimizer, compression_scheduler, pylogger, args):
    def load_test_data(args):
        test_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)
        return test_loader

    do_exit = False
    if args.greedy:
        greedy(model, criterion, optimizer, pylogger, args)
        do_exit = True
    elif args.summary:
        # This sample application can be invoked to produce various summary reports
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        do_exit = True
    elif args.export_onnx is not None:
        distiller.export_img_classifier_to_onnx(model,
                                                os.path.join(msglogger.logdir, args.export_onnx),
                                                args.dataset, add_softmax=True, verbose=False)
        do_exit = True
    elif args.qe_calibration and not (args.evaluate and args.quantize_eval):
        classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        do_exit = True
    elif args.activation_histograms:
        classifier.acts_histogram_collection(model, criterion, pylogger, args)
        do_exit = True
    elif args.sensitivity is not None:
        test_loader = load_test_data(args)
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)
        do_exit = True
    elif args.evaluate:
        if args.quantize_eval and args.qe_lapq:
            image_classifier_ptq_lapq(model, criterion, pylogger, args)
        else:
            test_loader = load_test_data(args)
            classifier.evaluate_model(test_loader, model, criterion, pylogger,
                classifier.create_activation_stats_collectors(model, *args.activation_stats),
                args, scheduler=compression_scheduler)
        do_exit = True
    elif args.thinnify:
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.contract_model(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        msglogger.info("Note: if your model collapsed to random inference, you may want to fine-tune")
        do_exit = True
    return do_exit


def init_knowledge_distillation(args, model, compression_scheduler):
    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher = apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)
        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)


def early_exit_init(args):
    if not args.earlyexit_thresholds:
        return
    args.num_exits = len(args.earlyexit_thresholds) + 1
    args.loss_exits = [0] * args.num_exits
    args.losses_exits = []
    args.exiterrors = []
    msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


class ClassifierCompressorSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)
        early_exit_init(self.args)
        # Save the randomly-initialized model before training (useful for lottery-ticket method)
        if args.save_untrained_model:
            ckpt_name = '_'.join((self.args.name or "", "untrained"))
            apputils.save_checkpoint(0, self.args.arch, self.model,
                                     name=ckpt_name, dir=msglogger.logdir)


    def handle_subapps(self):
        return handle_subapps(self.model, self.criterion, self.optimizer,
                              self.compression_scheduler, self.pylogger, self.args)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(classifier.test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=classifier.create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.png'))
    distiller.sensitivities_to_csv(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.csv'))


def greedy(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader = classifier.load_data(args)

    test_fn = partial(classifier.test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(classifier.train, train_loader=train_loader, criterion=criterion, args=args)

    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)

#GA拡張
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #昇順,小さいのが良いとき, 1.0大きいのが良い
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
 
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 54) #個体長,bit
#5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 ,65, 70, 75, 80, 85, 90 #各18bit
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", main)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)



if __name__ == '__main__':
    #main()
    #quit()
    pop = toolbox.population(n=100) #1世代ごとの個体数
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40  #交叉率, 個体突然変異率, 世代数

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    for g in range(NGEN):
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    '''
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
    '''
    if msglogger is not None and hasattr(msglogger, 'log_filename'):
        msglogger.info('')
        msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
