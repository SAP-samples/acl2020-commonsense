#
# SPDX-FileCopyrightText: 2020 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0

import os
from tqdm import trange,tqdm
#import gap_utils
#import wnli_utils
import json
import re

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, candidate_a, candidate_b, ex_true=True):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. Sentence analysed with pronoun replaced for _
            candidate_a: string, correct candidate
            candidate_b: string, incorrect candidate
        """
        self.guid = guid
        self.text_a = text_a
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b #only used for train
        self.ex_true = ex_true
        #ex_true only matters for testing and has following string values:
        #"true" - LM has to pick this over others,
        #"false" - LM should not pick this over others
        #"other" - not known, not important, this is "other" candidate
        #"err_true" - Correct candidate but Spacy failed to find it. Automatically wrong
        #"err_false" - Incorrect candidate but Spacy failed to find it. Automatically correct

class DataProcessor(object):
    """Processor for the Wiki data set."""

    def wnli_test(self,source):
        examples=[]
        for line in tqdm(list(open(source,'r'))[1:]):
            tokens = line.strip().split('\t')
            guid = tokens[0]
            premise = tokens[1]
            hypothesis = tokens[2]
            premise,candidates = wnli_utils.transform_wnli(premise,hypothesis)
            if premise==None:
                examples.append(InputExample(guid,"","",None,ex_true="err_false"))
                continue
            candidate_a = candidates[0]
            candidates_b = candidates[1:]
            examples.append(InputExample(guid,premise,candidate_a,None,ex_true="true"))#we don't really know if it's true, but as long as it's not "other" it's fine
            for cand in candidates_b:
                examples.append(InputExample(guid,premise,cand,None,ex_true="other"))
        return examples

    def gap_train(self, source):
        examples=[]
        for line in tqdm(list(open(source,'r'))[1:],desc="Reading and pre-processing data"):
            tokens = line.strip().split('\t')
            guid = tokens[0]
            sentence = tokens[1]
            pronoun = tokens[2]
            pronoun_offset = int(tokens[3])
            sentence = sentence[:pronoun_offset]+"_"+sentence[pronoun_offset+len(pronoun):]
            candidate_a = tokens[4]
            candidate_b = tokens[7]
            if tokens[6].lower()=="true":
                examples.append(InputExample(guid,sentence,candidate_a,candidate_b))
            if tokens[9].lower()=="true":
                examples.append(InputExample(guid,sentence,candidate_b,candidate_a))
        return examples

    def gap_test(self,source):
        examples=[]
        for line in tqdm(list(open(source,'r'))[1:],desc="Reading and pre-processing data"):
            tokens = line.strip().split('\t')
            guid = tokens[0]
            sentence = tokens[1]
            pronoun = tokens[2]
            pronoun_offset = int(tokens[3])
            sentence = sentence[:pronoun_offset]+"_"+sentence[pronoun_offset+len(pronoun):]
            candidate_a = tokens[4]
            candidate_b = tokens[7]
            other_candidates = gap_utils.get_candidates(sentence)
            if pronoun.casefold() == "his":#due to the abiguity of English language, the same cannot be done for "her"
                candidate_a = candidate_a+"\'s"
                candidate_b = candidate_b+"\'s"
                for i in range(len(other_candidates)):
                    other_candidates[i]= other_candidates[i]+"\'s"
            if candidate_a.casefold() in [cand.casefold() for cand in other_candidates]:#candidate_a was detected by NER
                examples.append(InputExample(guid+"A",sentence,candidate_a,None,ex_true = tokens[6].lower()))
                for other in list(filter(lambda a: a.casefold()!= candidate_a.casefold(), other_candidates)):
                    examples.append(InputExample(guid+"A",sentence,other,None,ex_true = "other"))
            else:
                examples.append(InputExample(guid+"A",sentence,candidate_a,None,ex_true = "err_"+tokens[6].lower()))
            if candidate_b.casefold() in [cand.casefold() for cand in other_candidates]:
                examples.append(InputExample(guid+"B",sentence,candidate_b,None,ex_true = tokens[9].lower()))
                for other in list(filter(lambda a: a.casefold()!= candidate_b.casefold(), other_candidates)):
                    examples.append(InputExample(guid+"B",sentence,other,None,ex_true = "other"))
            else:
                examples.append(InputExample(guid+"B",sentence,candidate_b,None,ex_true = "err_"+tokens[9].lower()))
        return examples

    def read_dpr_format_train(self,source):
        examples = []
        lines = list(open(source,'r'))
        for id_x,(sent,pronoun,candidates,candidate_a,_) in enumerate(zip(lines[0::5],lines[1::5],lines[2::5],lines[3::5],lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' '+pronoun.strip()+' '," _ ",1)
            cnd = candidates.split(",")
            cnd = (cnd[0].strip().lstrip(),cnd[1].strip().lstrip())
            candidate_a = candidate_a.strip().lstrip()
            if candidate_a.casefold()==cnd[0].casefold():
                candidate_b = cnd[1]
            else:
                candidate_b=cnd[0]
            examples.append(InputExample(guid, text_a, candidate_a, candidate_b, ex_true="true"))
        return examples

    def read_dpr_format_test(self,source):
        examples = []
        lines = list(open(source,'r'))
        for id_x,(sent,pronoun,candidates,candidate_a,_) in enumerate(zip(lines[0::5],lines[1::5],lines[2::5],lines[3::5],lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' '+pronoun.strip()+' '," _ ",1)
            candidate_a = candidate_a.strip().lstrip()
            cnd = candidates.strip().split(",")
            cnd = (candidate.strip().lstrip() for candidate in cnd if candidate.strip().lstrip().casefold()!= candidate_a.casefold())
            examples.append(InputExample(guid, text_a, candidate_a, None, ex_true="true"))
            for candidate in cnd:
                examples.append(InputExample(guid, text_a, candidate, None,ex_true="other"))
        return examples

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                records.append(json.loads(line))
            return records


    @classmethod
    def _read_json(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, 'r') as f:
            records = json.load(f)

            return records


    def _create_examples_test(self, records):
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            sentence = record['sentence']

            name1 = record['option1']
            name2 = record['option2']
            if not 'answer' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = "1"
            else:
                label = record['answer']

            conj = "_"
            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = "_ " + sentence[idx + len(conj):].strip()

            option1 = option_str.replace("_", name1)
            option2 = option_str.replace("_", name2)

            if label == "1":
                examples.append(InputExample(guid,sentence,name1,None, ex_true="true"))
                examples.append(InputExample(guid,sentence,name2,None, ex_true="other"))
            elif label == "2":
                examples.append(InputExample(guid,sentence,name2,None, ex_true="true"))
                examples.append(InputExample(guid,sentence,name1,None, ex_true="other"))
            else:
                print('unknown label!')


        return examples


    def _create_examples_train(self, records):
        examples = []
        for (i, record) in enumerate(records):
            guid = record['qID']
            sentence = record['sentence']

            name1 = record['option1']
            name2 = record['option2']
            if not 'answer' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = "1"
            else:
                label = record['answer']

            conj = "_"
            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = "_ " + sentence[idx + len(conj):].strip()

            option1 = option_str.replace("_", name1)
            option2 = option_str.replace("_", name2)

            if label == "1":
                mc_example = InputExample(guid,sentence,name1,name2)
            else:
                mc_example = InputExample(guid,sentence,name2,name1)

            examples.append(mc_example)

        return examples


    def _create_examples_knowref_test(self, records):
        examples = []
        for (i, record) in enumerate(records):
            guid = record['oiginal_id']
            sentence = record['sentence_with_pronoun']

            sentence = re.sub("[\[].*?[\]]", "_", sentence)

            name1 = record['candidate0'][0]
            name2 = record['candidate1'][0]
            if not 'correct_candidate_idx' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = "1"
            else:
                label = str(int(record['correct_candidate_idx'])+1)

            #conj = "_"
            #idx = sentence.index(conj)
            #context = sentence[:idx]
            #option_str = "_ " + sentence[idx + len(conj):].strip()

            #option1 = option_str.replace("_", name1)
            #option2 = option_str.replace("_", name2)

            if label == "1":
                examples.append(InputExample(guid,sentence,name1,None, ex_true="true"))
                examples.append(InputExample(guid,sentence,name2,None, ex_true="other"))
            elif label == "2":
                examples.append(InputExample(guid,sentence,name2,None, ex_true="true"))
                examples.append(InputExample(guid,sentence,name1,None, ex_true="other"))
            else:
                print('unknown label!')


        return examples



    def get_examples(self, data_dir, set_name):#works for differently for train!
        """See base class."""
        file_names = {
                "wikicrem-train":"WikiCREM_train.txt",
                "wikicrem-dev":"WikiCREM_dev.txt",
                "gap-train": "gap-development.tsv",
                "gap-dev": "gap-validation.tsv",
                "gap-test": "gap-test.tsv",
                "dpr-train": "train.c.txt",
                "wscr-train": "train.c.txt",
                "dpr-test": "test.c.txt",
                "wscr-test": "test.c.txt",
                "dpr-train-small": "dpr_train_small.txt",
                "dpr-dev-small": "dpr_dev_small.txt",
                "wsc": "wsc273.txt",
                "pdp": "PDP.txt",
                "winogender": "WinoGender.txt",
                "winobias-pro1": "pro_stereotyped_1.txt",
                "winobias-anti1": "anti_stereotyped_1.txt",
                "winobias-pro2": "pro_stereotyped_2.txt",
                "winobias-anti2": "anti_stereotyped_2.txt",
                "winobias-dev": "winobias_dev.txt",
                "wnli":"wnli-test.tsv",
                "maskedwiki":"MaskedWiki_2.4Mtrain.txt",
                "winogrande-xl-train": "winogrande_1.1/train_xl.jsonl",
                "winogrande-l-train": "winogrande_1.1/train_l.jsonl",
                "winogrande-m-train": "winogrande_1.1/train_m.jsonl",
                "winogrande-s-train": "winogrande_1.1/train_s.jsonl",
                "winogrande-xs-train": "winogrande_1.1/train_xs.jsonl",
                "winogrande-dev": "winogrande_1.1/dev.jsonl",
                "winogrande-test": "winogrande_1.1/test2.jsonl",
                "knowref-test": "knowref_test.json"
                }
        source = os.path.join(data_dir,file_names[set_name])
        if set_name == "gap-train":
            return self.gap_train(source)
        elif set_name in ["gap-dev","gap-test"]:
            return self.gap_test(source)
        elif set_name in ["dpr-train","wscr-train","dpr-train-small","wikicrem-train","maskedwiki"]:
            return self.read_dpr_format_train(source)
        elif set_name in ["dpr-test","wscr-test","dpr-dev-small","wsc","pdp","winogender","winobias-pro1","winobias-pro2","winobias-anti1","winobias-anti2","winobias-dev","wikicrem-dev"]:
            return self.read_dpr_format_test(source)
        elif set_name=="wnli":
            return self.wnli_test(source)
        elif set_name in ["winogrande-xl-train", "winogrande-l-train", "winogrande-m-train", "winogrande-s-train", "winogrande-xs-train"]:
            return self._create_examples_train(
            self._read_jsonl(source) )
        elif set_name in [ "winogrande-dev"]:
            return self._create_examples_test(
            self._read_jsonl(source) )
        elif set_name in [ "knowref-test"]:
            return self._create_examples_knowref_test(
            self._read_json(source) )
        else:
            print("Unknown set_name: ",set_name)
