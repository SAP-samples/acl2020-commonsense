#
# SPDX-FileCopyrightText: 2020 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team., 2019 Intelligent Systems Lab, University of Oxford, SAP SE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import csv
import json
import logging
import argparse
import random
import logging
from tqdm import tqdm, trange
import re
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertOnlyMLMHead
#from transformers import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from torch import nn, optim
from data_reader import InputExample,DataProcessor
from scorer import scorer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)





class BertForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **masked_lm_loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **ltr_lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next token prediction loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')  # -1 index = padding token
           
            masked_lm_loss = loss_fct(prediction_scores.permute(0,2,1), masked_lm_labels)
           
            
            masked_lm_loss_normalized = torch.div(torch.mean(masked_lm_loss,1),(masked_lm_labels > -1).sum(dim=1,dtype=torch.float32))
            
            masked_lm_loss_normalized[torch.isnan(masked_lm_loss_normalized)] = 0.0
            
            outputs = (masked_lm_loss_normalized,) + outputs
            

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, type_1, type_2, masked_lm_1, masked_lm_2, start, end_1, end_2, source_start_token_1, source_end_token_1, source_start_token_2, source_end_token_2):
        self.input_ids_1=input_ids_1
        self.attention_mask_1=attention_mask_1
        self.type_1=type_1
        self.masked_lm_1=masked_lm_1
        #These are only used for train examples
        self.input_ids_2=input_ids_2
        self.attention_mask_2=attention_mask_2
        self.type_2=type_2
        self.masked_lm_2=masked_lm_2
        self.start = start
        self.end_1 = end_1
        self.end_2 = end_2
        self.source_start_token_1 = source_start_token_1
        self.source_end_token_1 = source_end_token_1
        self.source_start_token_2 = source_start_token_2
        self.source_end_token_2 = source_end_token_2

def convert_examples_to_features_train(examples, max_seq_len, tokenizer, mode='multimask'):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    count = [0,0]
    for (ex_index, example) in enumerate(examples):
        tokens_sent = tokenizer.tokenize(example.text_a)
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_b = tokenizer.tokenize(example.candidate_b)
        if len(tokens_a) == len(tokens_b):
            count[0]=count[0]+1 
        else: 
            count[1]=count[1]+1
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_2, type_2, attention_mask_2, masked_lm_2 = [],[],[],[]
        tokens_1.append("[CLS]")
        tokens_2.append("[CLS]")
        for token in tokens_sent:
            if token=="_":
                start = len(tokens_1)
                if mode == 'multimask':
                    tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
                    tokens_2.extend(["[MASK]" for _ in range(len(tokens_b))])
                else:
                    tokens_1.append("[MASK]")
                    tokens_2.append("[MASK]")
                    
                end_1 = len(tokens_1)
                end_2 = len(tokens_2)
            else:
                tokens_1.append(token)
                tokens_2.append(token)
                
        
        token_idx_1 = []
        token_idx_2 = []
        token_counter_1 = 0
        token_counter_2 = 0
        find_tokens_a = True
        find_tokens_b = True
                
        for idx, token in enumerate(tokens_a):
            
            if ( find_tokens_a and token.lower() == tokens_a[token_counter_1].lower()):
                token_idx_1.append(idx)
                token_counter_1 += 1
                if ( len(token_idx_1) >= len(tokens_a) ):
                    find_tokens_a = False
            elif find_tokens_a:
                token_idx_1 = []
                token_counter_1 = 0
                
                
        for idx, token in enumerate(tokens_b):
                
            if ( find_tokens_b and token.lower() == tokens_b[token_counter_2].lower()):
                token_idx_2.append(idx)
                token_counter_2 += 1
                if ( len(token_idx_2) >= len(tokens_b) ):
                    find_tokens_b = False
            elif find_tokens_b:
                token_idx_2 = []
                token_counter_2 = 0
        
        
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        tokens_2 = tokens_2[:max_seq_len-1]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")
        if tokens_2[-1]!="[SEP]":
            tokens_2.append("[SEP]")

        type_1 = max_seq_len*[0]#We do not do any inference.
        type_2 = max_seq_len*[0]#These embeddings can thus be ignored

        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        attention_mask_2 = (len(tokens_2)*[1])+((max_seq_len-len(tokens_2))*[0])

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

        for token in tokens_1:
            if token=="[MASK]":
                if len(input_ids_a)<=0:
                    continue#broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-1)

        for token in tokens_2:
            if token=="[MASK]":
                if len(input_ids_b)<=0:
                    continue#broken case
                masked_lm_2.append(input_ids_b[0])
                input_ids_b = input_ids_b[1:]
            else:
                masked_lm_2.append(-1)
        while len(masked_lm_2)<max_seq_len:
            masked_lm_2.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        while len(input_ids_2) < max_seq_len:
            input_ids_2.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(input_ids_2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(attention_mask_2) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(type_2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        assert len(masked_lm_2) == max_seq_len
        #if len(tokens_a) == len(tokens_b):
        features.append(
                InputFeatures(input_ids_1=input_ids_1,
                              input_ids_2=input_ids_2,
                              attention_mask_1=attention_mask_1,
                              attention_mask_2=attention_mask_2,
                              type_1=type_1,
                              type_2=type_2,
                              masked_lm_1=masked_lm_1,
                              masked_lm_2=masked_lm_2, start=start, end_1=end_1, end_2=end_2, source_start_token_1=token_idx_1[0],  source_end_token_1=token_idx_1[-1], source_start_token_2=token_idx_2[0],  source_end_token_2=token_idx_2[-1]))
    logger.info('Ratio: '+str(count[0]/(count[0]+count[1])))
    return features


def convert_examples_to_features_evaluate(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_sent = tokenizer.tokenize(example.text_a)
        
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_1.append("[CLS]")
        for token in tokens_sent:
            if token=="_":
                tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
            else:
                tokens_1.append(token)
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")

        type_1 = max_seq_len*[0]
        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        for token in tokens_1:
            if token=="[MASK]":
                if len(input_ids_a)<=0:
                    continue#broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-1)
        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(masked_lm_1) == max_seq_len

        features.append(
                InputFeatures(input_ids_1=input_ids_1,
                              input_ids_2=None,
                              attention_mask_1=attention_mask_1,
                              attention_mask_2=None,
                              type_1=type_1,
                              type_2=None,
                              masked_lm_1=masked_lm_1,
                              masked_lm_2=None, start=None, end_1=None, end_2=None, source_start_token_1=None, source_end_token_1=None, source_start_token_2=None, source_end_token_2=None))
    return features

def test(processor, args, tokenizer, model, device, global_step = 0, tr_loss = 0, test_set = "wscr-test", verbose=False, output_file=None):
    eval_examples = processor.get_examples(args.data_dir,test_set)
    eval_features = convert_examples_to_features_evaluate(
        eval_examples, args.max_seq_length, tokenizer)
    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in eval_features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in eval_features], dtype=torch.long)
    all_segment_ids_1 = torch.tensor([f.type_1 for f in eval_features], dtype=torch.long)
    all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_1, all_attention_mask_1, all_segment_ids_1, all_masked_lm_1)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    ans_stats=[]
    for batch in eval_dataloader: #tqdm(eval_dataloader,desc="Evaluation"):
        input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = (tens.to(device) for tens in batch)
        with torch.no_grad():
            loss,_,_ = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)

        eval_loss = loss.to('cpu').numpy()

        for loss in eval_loss:
            curr_id = len(ans_stats)
            ans_stats.append((eval_examples[curr_id].guid,eval_examples[curr_id].ex_true,loss))
    if test_set=="gap-test":
        return scorer(ans_stats,test_set,output_file=os.path.join(args.output_dir, "gap-answers.tsv"))
    elif test_set=="wnli":
        return scorer(ans_stats,test_set,output_file=os.path.join(args.output_dir, "WNLI.tsv"))
    else:
        if output_file is not None:
            return scorer(ans_stats,test_set, output_file=os.path.join(args.output_dir, output_file))
        else:
            return scorer(ans_stats,test_set)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--alpha_param",
                        default=10,
                        type=float,
                        help="Discriminative penalty hyper-parameter.")
    parser.add_argument("--gamma_param",
                        default=20,
                        type=float,
                        help="Mutual exclusivity strength hyper-parameter.")
    parser.add_argument("--beta_param",
                        default=0.4,
                        type=float,
                        help="Discriminative intolerance interval hyper-parameter.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--load_from_file',
                        type=str,
                        default=None,
                        help="Path to the file with a trained model. Default means bert-model is used. Size must match bert-model.")
    
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    parser.add_argument('--shuffle', action='store_true',
                        help="Whether to shuffle at twin-pair level to avoid potential bias.")
            
    args = parser.parse_args()
    
    
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    
    
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    processor = DataProcessor()
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab


    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_name = {"gap":"gap-train",
                "wikicrem":"wikicrem-train",
                "dpr":"dpr-train-small",
                "wscr":"wscr-train",
                "winogrande": "winogrande-l-train",
                "maskedwiki":"maskedwiki",
                }[task_name]
        
        
        train_examples = processor.get_examples(args.data_dir, train_name)

        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        
    # shuffle the data
    # random.shuffle(train_examples)
    if args.shuffle:
        # make sure that the labels are *not* used implicitly
        logger.info('Shuffling twin-pairs ...')
        for i in range(0, len(train_examples), 2):
            if random.choices([0,1]) == [0]:
                candidate_a = copy.deepcopy(train_examples[i].candidate_a)
                candidate_b = copy.deepcopy(train_examples[i].candidate_b)

                train_examples[i].candidate_a = candidate_b
                train_examples[i].candidate_b = candidate_a

                

                candidate_a = copy.deepcopy(train_examples[i+1].candidate_a)
                candidate_b = copy.deepcopy(train_examples[i+1].candidate_b)
                train_examples[i+1].candidate_a = candidate_b
                train_examples[i+1].candidate_b = candidate_a

    # Prepare model
    if args.load_from_file is None:
        model = BertForMaskedLM.from_pretrained(args.bert_model, 
                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), output_attentions=True)
    else:
        model = BertForMaskedLM.from_pretrained(args.bert_model, 
                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), untrained=True, output_attentions=True)
    model.to(device)
    
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    

    if not args.load_from_file is None:
        model_dict = torch.load(args.load_from_file)
        
                
        model.load_state_dict(new_dict)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)  # PyTorch scheduler

    
    if args.fp16:
        # apex
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss,nb_tr_steps = 0, 1
    if args.do_train:
        train_features = convert_examples_to_features_train(
            train_examples, args.max_seq_length, tokenizer, mode='multimask')
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        # Load data for even twin-pair sentences (% 2 == 0)
        all_input_ids_1 = torch.tensor([f.input_ids_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_attention_mask_1 = torch.tensor([f.attention_mask_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_attention_mask_2 = torch.tensor([f.attention_mask_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_segment_ids_1 = torch.tensor([f.type_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.type_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_masked_lm_1 = torch.tensor([f.masked_lm_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_masked_lm_2 = torch.tensor([f.masked_lm_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.long)
        all_start = torch.tensor([f.start for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)
        all_end_1 = torch.tensor([f.end_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)
        all_end_2 = torch.tensor([f.end_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)   
        all_source_start_1 = torch.tensor([f.source_start_token_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)
        all_source_start_2 = torch.tensor([f.source_start_token_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)
        all_source_end_1 = torch.tensor([f.source_end_token_1 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)
        all_source_end_2 = torch.tensor([f.source_end_token_2 for index, f in enumerate(train_features) if index % 2 == 0], dtype=torch.int16)
        
        
        # Load data for odd twin-pair sentences (%2 == 1)
        _all_input_ids_1 = torch.tensor([f.input_ids_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_input_ids_2 = torch.tensor([f.input_ids_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_attention_mask_1 = torch.tensor([f.attention_mask_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_attention_mask_2 = torch.tensor([f.attention_mask_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_segment_ids_1 = torch.tensor([f.type_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_segment_ids_2 = torch.tensor([f.type_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_masked_lm_1 = torch.tensor([f.masked_lm_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_masked_lm_2 = torch.tensor([f.masked_lm_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.long)
        _all_start = torch.tensor([f.start for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)
        _all_end_1 = torch.tensor([f.end_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)
        _all_end_2 = torch.tensor([f.end_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)   
        _all_source_start_1 = torch.tensor([f.source_start_token_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)
        _all_source_start_2 = torch.tensor([f.source_start_token_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)
        _all_source_end_1 = torch.tensor([f.source_end_token_1 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)
        _all_source_end_2 = torch.tensor([f.source_end_token_2 for index, f in enumerate(train_features) if index % 2 == 1], dtype=torch.int16)
        
        
        
        
        
        
        train_data = TensorDataset(all_input_ids_1, all_input_ids_2, all_attention_mask_1, all_attention_mask_2, all_segment_ids_1, all_segment_ids_2, all_masked_lm_1, all_masked_lm_2, all_start, all_end_1, all_end_2, all_source_start_1, all_source_end_1, all_source_start_2, all_source_end_2,
                                   _all_input_ids_1, _all_input_ids_2, _all_attention_mask_1, _all_attention_mask_2, _all_segment_ids_1, _all_segment_ids_2, _all_masked_lm_1, _all_masked_lm_2, _all_start, _all_end_1, _all_end_2, _all_source_start_1, _all_source_end_1, _all_source_start_2, _all_source_end_2)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        validation_name = {"gap":"gap-dev",
                "wikicrem":"wikicrem-dev",
                "dpr":"dpr-dev-small",
                "maskedwiki":"wscr-test",
                "winogrande": "winogrande-dev",
                "wscr":"wscr-test",
                }[task_name]

        model.train()
        try:#This prevents overwriting if several scripts are running at the same time (for hyper-parameter search)
            best_accuracy = float(list(open(os.path.join(args.output_dir,"best_accuracy.txt"),'r'))[0])
        except:
            best_accuracy = 0
        for it in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            tr_accuracy = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            if it == 0:
                acc = test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set=validation_name, verbose=True)
                logger.info("Initial Eval: {}\t{}\n".format(nb_tr_steps,acc))
            for step, batch in enumerate(tqdm(train_dataloader)):
                input_ids_1,input_ids_2,input_mask_1,input_mask_2, segment_ids_1, segment_ids_2, label_ids_1, label_ids_2, target_start, target_end_1, target_end_2, source_start_1, source_end_1, source_start_2, source_end_2, _input_ids_1,_input_ids_2,_input_mask_1,_input_mask_2, _segment_ids_1, _segment_ids_2, _label_ids_1, _label_ids_2, _target_start, _target_end_1, _target_end_2, _source_start_1, _source_end_1, _source_start_2, _source_end_2 = (tens.to(device) for tens in batch)
                          
                # First twin-pair sentence
                # Candidate A
                loss_1, score_1, attn_1 = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)
                # Candidate B
                loss_2, score_2, attn_2 = model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = input_mask_2, masked_lm_labels = label_ids_2)
                
                # Second twin-pair sentence
                # Candidate A
                _loss_1, _score_1, _attn_1 = model.forward(_input_ids_1, token_type_ids = _segment_ids_1, attention_mask = _input_mask_1, masked_lm_labels = _label_ids_1)
                # Candidate B
                _loss_2, _score_2, _attn_2 = model.forward(_input_ids_2, token_type_ids = _segment_ids_2, attention_mask = _input_mask_2, masked_lm_labels = _label_ids_2)
                
             
                # contrastive margin loss of first twin-pair sentences
                loss = args.alpha_param * torch.max(torch.zeros(loss_1.size(),device=device),torch.ones(loss_1.size(),device=device)*args.beta_param + loss_1 - loss_2.mean())  + args.alpha_param * torch.max(torch.zeros(loss_2.size(),device=device),torch.ones(loss_2.size(),device=device)*args.beta_param + loss_2 - loss_1.mean())
                
                # constrastive margin loss of second twin-pair sentences
                loss += args.alpha_param * torch.max(torch.zeros(_loss_1.size(),device=device),torch.ones(_loss_1.size(),device=device)*args.beta_param + _loss_1 - _loss_2.mean())  + args.alpha_param * torch.max(torch.zeros(_loss_2.size(),device=device),torch.ones(_loss_2.size(),device=device)*args.beta_param + _loss_2 - _loss_1.mean())
                mex = torch.zeros(1).cuda()

                # compute the mutual-exclusive loss
                for i in range(loss_1.shape[0]):
                    
                    #eps = 0.0001
                    cexp_11 = torch.exp(-loss_1[i])
                    cexp_12 = torch.exp(-loss_2[i])
                    
                    
                    cexp_21 = torch.exp(-_loss_1[i])
                    cexp_22 = torch.exp(-_loss_2[i])
                   
                    joint_exp_1 = (cexp_11 + cexp_12)
                    joint_exp_2 = (cexp_21 + cexp_22)
                    
                    
                   
                    # First twin-pair sentences
                    term_1 = cexp_11/joint_exp_1 * cexp_21/joint_exp_2 * (1. - (1.-cexp_11/joint_exp_1)*(1.-cexp_21/joint_exp_2))
                    # Second twin-pai sentences
                    term_2 = cexp_12/joint_exp_1 * cexp_22/joint_exp_2 * (1. - (1.-cexp_12/joint_exp_1)*(1.-cexp_22/joint_exp_2))
                    
                    # full mutual-exclusive term
                    mex += -1.*((term_1) + (term_2))
                    
                    
                    if torch.isnan(term_1) or torch.isinf(term_1) or torch.isnan(term_2) or torch.isinf(term_2):
                        logger.error("NaN or Inf")

                        exit(1)
                    
                  
       
                   
              
                loss = loss.mean()+ args.gamma_param*mex 
               
                tr_accuracy += len(np.where(loss_1.detach().cpu().numpy()-loss_2.detach().cpu().numpy()<0.0)[0])
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
          
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
          
                    
                tr_loss += loss.item()
                nb_tr_examples += input_ids_1.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()
                    global_step += 1
                if not (task_name in ["wscr","gap","dpr"]) and global_step % 50 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:#testing during an epoch
                    acc = test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set=validation_name, verbose=True)
                    logger.info("{}\t{}\n".format(nb_tr_steps,acc))
                    model.train()
                    try:#If several processes are running in parallel this avoids overwriting results.
                        updated_accuracy = float(list(open(os.path.join(args.output_dir,"best_accuracy.txt"),'r'))[0])
                    except:
                        updated_accuracy = 0
                    best_accuracy = max(best_accuracy,updated_accuracy)
                    if acc>best_accuracy:
                        best_accuracy = acc
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model"))

                        with open(os.path.join(args.output_dir,"best_config.txt"),'w') as f1_report:
                            f1_report.write("{}".format(' '.join(sys.argv[1:])))
                        with open(os.path.join(args.output_dir,"best_accuracy.txt"),'w') as f1_report:
                            f1_report.write("{}".format(best_accuracy))
            if validation_name=="all":
                acc = (test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set = "gap-dev", verbose=True) +\
                        test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set = "winobias-dev", verbose=True))/2
            else:
                acc = test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set = validation_name, verbose=True)
            logger.info("{}\t{}\n".format(nb_tr_steps,acc))
            model.train()
            try:
                updated_accuracy = float(list(open(os.path.join(args.output_dir,"best_accuracy.txt"),'r'))[0])
            except:
                updated_accuracy = 0
            best_accuracy = max(best_accuracy,updated_accuracy)
            if acc>best_accuracy:
                best_accuracy = acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model"))
                with open(os.path.join(args.output_dir,"best_accuracy.txt"),'w') as f1_report:
                    f1_report.write("{}".format(best_accuracy))
        #reload the best model
        logger.info("Best dev acc {}".format(best_accuracy))
        model_dict = torch.load(os.path.join(args.output_dir, "best_model"))
        model.load_state_dict(model_dict)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if True:
            print("Knowref-test: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="knowref-test"))
            print("DPR/WSCR-test: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="dpr-test"))
            print("WSC: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="wsc",  output_file='wsc-eval.tsv'))
            print("PDP: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="pdp"))
if __name__ == "__main__":
    main()
