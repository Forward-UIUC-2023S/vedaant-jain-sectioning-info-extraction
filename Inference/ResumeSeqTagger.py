from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import DataCollatorForTokenClassification
import easyocr

class ResumeSeqTagger():
    def __init__(self, model_checkpoint='bert_seq_tag', ocr_reader = None) -> None:
        self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.collator = DataCollatorForTokenClassification(self.tokenizer)
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        self.label2id['missing'] = len(self.label2id)
        self.id2label[len(self.id2label)] = 'missing'
        
        if ocr_reader is None:
            self.reader = easyocr.Reader(['en'])
        else:
            self.reader = ocr_reader
    
    def Get_Text(self, image):
        return self.reader.readtext(image, detail = 0)

    def Get_Batch(self, text):
        inputs = self.tokenizer(text, truncation=True, is_split_into_words=True)
        batch = self.collator([inputs])
        return batch
    
    def Get_Model_Ouputs(self, batch):
        outputs = self.model(**batch)
        return outputs
    
    def Map_Tokens(self, token):
        if token not in self.id2label:
            return 'missing'
        return self.id2label[token]

    def Clean_Section(self, preds, start_idx, cont_idx):
        to_change = []
        for i in range(len(preds)-2):
            if preds[i] == cont_idx and preds[i] == preds[i+2] and preds[i] != preds[i+1]:
                to_change.append(i)
        for i in to_change:
            preds[i] = cont_idx
        
        to_change = []
        beg = 0
        for i in range(len(preds)):
            if preds[i] == cont_idx and beg == 0:
                to_change.append(i)
                beg = 1
            elif preds[i] == start_idx:
                beg = 1
            elif preds[i] !=start_idx:
                beg = 0
        for i in to_change:
            preds[i] = start_idx
        
        return preds

    def Get_SecInfo(self, batch, preds, start_idx, cont_idx):
        pred_idx = [i for i in range(len(preds)) if self.label2id[preds[i]] == start_idx or self.label2id[preds[i]] == cont_idx]
        pred_tokens = [batch['input_ids'][0][i] for i in pred_idx]
        batch['token_ids'] = pred_tokens
        sec_info = self.tokenizer.decode(**batch)
        return sec_info
    
    def GetPreds(self, batch):
        outputs = self.Get_Model_Ouputs(batch)
        predictions = outputs.logits.argmax(dim=-1)
        preds = [self.Map_Tokens(token) for token in predictions[0].cpu().detach().numpy()]
        preds = self.Clean_Preds(preds)
        return preds

    def Get_InfoAll(self, text):
        print('Only Derived Classes have this method defined...')
        raise NotImplementedError()
    
    def Extract_Info(self, image):
        print('Only Derived Classes have this method defined...')
        raise NotImplementedError()