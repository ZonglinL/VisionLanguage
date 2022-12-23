'''

Test on single GPU:
    CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data_lmdb_release/evaluation --benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --sensitive --data_filtering_off  --imgH 224 --imgW 224 --TransformerModel=vitstr_small_patch16_224 --saved_model https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth


To convert to quantized model, add the ff to the script above:
    --infer_model=vitstr_small_patch16_quant.pt --quantized

To convert to a standalone jit model, add the ff to the script above:
    --infer_model=vitstr_small_patch16_jit.pt


'''

import os
import time
import string
import argparse
import re
import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model, JitModel
from utils import get_args
from transformers import RobertaForMaskedLM,RobertaConfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt):  # , calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """

    if opt.fast_acc:
        # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                          'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    eval_data_list = ['data']
    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}'

    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    total_loss = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        if opt.Transformer:
            target = converter.encode(labels)


        start_time = time.time()

        if opt.Transformer:
            preds = model(image, text=target, seqlen=converter.batch_max_length)
            _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
            preds_index = preds_index.view(-1, converter.batch_max_length)
            forward_time = time.time() - start_time
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            total_loss += cost.item()
            if i == 5000:
                print(total_loss/(i+1))
                break

            length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
            preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

        infer_time += forward_time
        valid_loss_avg.add(cost)
        total_loss += cost.item()

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if opt.Transformer:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]


            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data






def test(opt):
    """ model configuration """

    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)


    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')


    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)
            log.write(eval_data_log)
            print(f'{accuracy_by_best_model:0.3f}')
            log.write(f'{accuracy_by_best_model:0.3f}\n')
            log.close()

def loss_fn(x,y):
    dot_mat = torch.matmul(x.log_softmax(-1),y.softmax(-1).transpose(-1,-2))
    return torch.diagonal(dot_mat, offset=0, dim1=-2, dim2=-1) ## length by length matrix, diag is the dot product between same position

def train(opt,bert):
    """ model configuration """
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)


    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')


    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    """ evaluation """
    model.train()
    opt.eval = False

    eval_data_list = ['data']

    evaluation_batch_size = opt.batch_size


    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    eval_data = eval_data_list[0]

    eval_data_path = os.path.join(opt.eval_data, eval_data)
    AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)


    length_of_data = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(opt.beta1, 0.999))
    for epoch in range(1):
        total_loss = 0

        for i, (image_tensors, labels) in enumerate(evaluation_loader):
            optimizer.zero_grad()
            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(device)
            # For max length prediction

            target = converter.encode(labels)

            preds = model(image, text=target, seqlen=converter.batch_max_length)

            pred_ids = preds.argmax(-1)
            not_match = pred_ids != target #keep unmatch
            ign = target != 0 ## keep non-zero
            indicator = not_match*ign
            bert_ids = torch.clone(pred_ids)
            bert_ids[not_match] = 95 ## replace unmatched with mask
            msk = bert_ids == target  ## 0 for mask
            msk[target == 0] = 1 ## do not mask [Go]

            bert_out = bert(bert_ids,attention_mask = msk ).logits
            loss = -loss_fn(preds,bert_out)
            loss = -loss_fn(bert_out, preds)
            adj_loss = loss*indicator ## ignore correct labels and 0
            adj_loss = adj_loss.sum(-1)
            eff_len = indicator.sum(-1) ## how many are masked on each batch
            eff_len[eff_len == 0] = 1 ## in this case loss is zero
            adj_loss /= eff_len
            loss = adj_loss.mean()

            loss.backward()
            optimizer.step()

            if i == 5000:
                torch.save(model.state_dict(), opt.saved_model)
                break




if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    bert = RobertaForMaskedLM(RobertaConfig()).to(device)
    bert.lm_head.decoder = torch.nn.Linear(768, 96)
    bert = bert.load_state_dict(torch.load('Roberta.pt'))
    bert = bert.to(device)
    train(opt,bert)
    test(opt)



