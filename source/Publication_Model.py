from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import DataCollatorForTokenClassification
import easyocr

from ResumeSeqTagger import ResumeSeqTagger

class PublicationsInformationExtractor(ResumeSeqTagger):
    def __init__(self, model_checkpoint='bert_seq_tag', ocr_reader = None) -> None:
        super().__init__(model_checkpoint, ocr_reader)

        self.authors_idx = (self.label2id['B-Authors'], self.label2id['I-Authors'])
        self.journal_idx = (self.label2id['B-Journal'], self.label2id['I-Journal'])
        self.title_idx = (self.label2id['B-Title'], self.label2id['I-Title'])
    
    def Clean_Authors(self, preds):
        return self.Clean_Section(preds, self.authors_idx[0], self.authors_idx[1])
    
    def Clean_Journal(self, preds):
        return self.Clean_Section(preds, self.journal_idx[0], self.journal_idx[1])
    
    def Clean_Title(self, preds):
        return self.Clean_Section(preds, self.title_idx[0], self.title_idx[1])
    
    def Clean_Preds(self, preds):
        preds = self.Clean_Authors(preds)
        preds = self.Clean_Journal(preds)
        preds = self.Clean_Title(preds)
        return preds
    
    def Get_InfoAll(self, text):
        batch = self.Get_Batch(text)
        preds = self.GetPreds(batch)
        information_extracted = {'Authors': self.Get_SecInfo(batch, preds, self.authors_idx[0], self.authors_idx[1]),
                                 'Journals': self.Get_SecInfo(batch, preds, self.journal_idx[0], self.journal_idx[1]),
                                 'Title': self.Get_SecInfo(batch, preds, self.title_idx[0], self.title_idx[1])}
        return information_extracted

    def Extract_Info(self, image):
        text = self.Get_Text(image)
        return self.Get_InfoAll(text)
