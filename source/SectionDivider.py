import os
import cv2

from ditod import add_vit_config

import torch

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

import easyocr
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util

import math
import pprint

from PIL import Image
import numpy as np
from numpy import asarray
from numpy import dot
from numpy.linalg import norm
import copy

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

def get_title_indices(classes, scores):
  title_indices = [i for i in range(len(classes)) if classes[i]==1]
  #confident_titles = 
  return [title_indices[i] for i in range(len(title_indices)) if scores[title_indices[i]] > 0.15]

def ConvertToWidthHeight(pred_box):
  new_pred_box = pred_box.copy()
  new_pred_box[2] = pred_box[2]-pred_box[0]
  new_pred_box[3] = new_pred_box[3] - new_pred_box[1]
  return new_pred_box

def divide_image(image, boxes, texts, pred_box_text_map):
        sections = []
        boxes = sorted(boxes, key = lambda x:x[1])
        print(pred_box_text_map)
        if len(boxes) > 0:
            sections = []
        else:
            return [image], texts
        texts_copy = copy.deepcopy(texts)
        i = 0
        for box in boxes:
          num = box[1]
          texts[i] = pred_box_text_map[num]
          i += 1
        if len(boxes) >= 1:
            sections.append(image[0:int(boxes[0][1]), :])
        for i in range(1, len(boxes)):
            sections.append(image[int(boxes[i-1][1]):int(boxes[i][1]), :])
        sections.append(image[int(boxes[-1][1]):, :])
        return sections, texts

class SectionDivider():
    def __init__(self, config_file = "config.yml", titles="default", iou_threshold=0.3, w2v_model = None):
        #Setup model
        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        self.cfg.merge_from_file("/content/gdrive/MyDrive/Resume_Info_Extraction/annotated_coco/cascade_dit_base.yml")
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(self.cfg)

        #setup ocr
        self.reader = easyocr.Reader(['en'])

        #setup titles
        if titles == "default":
            titles_list = ['ACADEMIC APPOINTMENTS', 'EDUCATION', 'RESEARCH INTERESTS', 'AWARDS, GRANTS AND HONORARY APPOINTMENTS', 'CITATION COUNT', 'COMPLETED RESEARCH ARTICLES', 'OTHER PUBLICATIONS', 'MEDIA COVERAGE'
                'CONSULTING EXPERIENCE', 'TEACHING EXPERIENCE', 'SEMINAR INVITATIONS', 'INVITED DISCUSSIONS AND LECTURES', 'CONFERENCE PRESENTATIONS', 'PROFESSIONAL SERVICE ',
                'COMMUNITY SERVICE', 'PROFESSIONAL EMPLOYMENT', 'RESEARCH', 'Journal Publications', 'Conference Publications', 'Working Papers', 'INVITED PRESENTATIONS', 'CONFERENCE PARTICIPATION','TEACHING',
                'CASE WRITING', 'RESEARCH FUNDING', 'ACADEMIC HONORS, AWARDS, AND ACTIVITIES', 'PROGRAMMING', 'PERSONAL INTERESTS', 'Summary', 'OTHER SCHOLARLY ACTIVITIES', 'Student Engagement',
                'Curriculum Development', 'SERVICE', 'Industry Experience', 'Employment', 'Visiting Positions', 'Awards and Honors', 'Students Awards', 'Publications', 'Course Materials', 'Abstracts, Preprints, Presentations',
                'Funding', 'Plenary Lectures', 'Invited Tutorials', 'Other Invited Workshop Talks', 'Invited Talks', 'Course Development', 'Instruction', 'Mentorship', 'Students', 'Advising', 'Service', 'Professional', 'Activities', 'Posters', 'Patents']
        else:
            titles_list = titles
        
        #setup word_embeddings
        if w2v_model == None:
            self.w2v_model = api.load("word2vec-google-news-300")
        else:
            self.w2v_model = w2v_model
        self.w2v_model_vocab = set(self.w2v_model.vocab)
        self.title_embeddings = []
        self.titles_list = list(set([x.lower() for x in titles_list]))
        for title in self.titles_list:
            title = title.replace(',','')
            words_list = title.split(' ')
            count = 0
            while count < len(words_list):
                if (words_list[count]) in self.w2v_model_vocab:
                    total_vect = self.w2v_model[str(words_list[count])].copy()
                    break
                count += 1
            for i in range(count, len(words_list)):
                if words_list[i] in self.w2v_model_vocab:
                    total_vect += self.w2v_model[str(words_list[i])]
            self.title_embeddings.append(total_vect)

        #setup iou_threshold
        self.iou_threshold = iou_threshold
        print("Instance of Section Divider created")
        self.title = "Section Divider"
    
    def AnalyzeImage(self, image):
        md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        md.set(thing_classes=["text","title","list","table","figure"])
        output = self.predictor(image)["instances"]
        v = Visualizer(image[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
        result = v.draw_instance_predictions(output.to("cpu"))
        result_image = result.get_image()[:, :, ::-1]
        
        return result_image, output.to("cpu")
    
    def ReadText(self, image, pred_bbox):
        pred_bbox_temp = pred_bbox
        image_pred = image[int(pred_bbox_temp[1]):int(pred_bbox_temp[3]), int(pred_bbox_temp[0]):int(pred_bbox_temp[2])]
        result = self.reader.readtext(image_pred)
        text = result[0][1]
        return text
    
    def CompareEvalTitles(self, target, titles):
        phrases_similarity = []
        target = target.lower()
        target = target.replace(',','')
        target_words_list = target.split(' ')
        count = 0
        total_vect = None
        while count < len(target_words_list):
            if (target_words_list[count]) in self.w2v_model_vocab:
                total_vect = self.w2v_model[str(target_words_list[count])].copy()
                break
            count += 1
        for i in range(count, len(target_words_list)):
            if target_words_list[i] in self.w2v_model_vocab:
                total_vect += self.w2v_model[str(target_words_list[i])]
        if total_vect is None:
            return (False, None)
        target_embedding = total_vect.copy()
        for idx, sent_embedding in enumerate(titles):
            sent_embedding = self.title_embeddings[idx]
            sim = dot(target_embedding, sent_embedding)/(norm(target_embedding)*norm(sent_embedding))
            phrases_similarity.append(sim)
        result = list(zip(phrases_similarity, titles))
        result.sort(key=lambda item:item[0], reverse=True)
        result = result[:3]
        #print("Target:", target, "\n", float(result[0][0]) > 0.65, result[0])
        #pprint.pprint(result)
        ans = float(result[0][0]) > 0.65
        return (ans, result[0][1])
    
    def FilterTitleBboxes(self, preds_bboxes, img_half, image):
        #print(img_half)
        pred_bboxes = [pred_bbox for pred_bbox in preds_bboxes if pred_bbox[0] < img_half]
        texts = [self.ReadText(image, pred_bbox) for pred_bbox in pred_bboxes]
        preds_indices = []
        most_sim_titles = []
        for i in range(len(pred_bboxes)):
          res = self.CompareEvalTitles(texts[i], self.titles_list)
          if res[0]:
            preds_indices.append(i)
            most_sim_titles.append(res[1])
        pred_bboxes = [pred_bboxes[preds_indices[i]] for i in range(len(preds_indices))]
        texts = [(texts[preds_indices[i]], most_sim_titles[i]) for i in range(len(preds_indices))]
        return pred_bboxes, texts
    

    def GetSections(self, image):
        analysis, raw_output = self.AnalyzeImage(image)
        preds = raw_output.pred_boxes
        scores = raw_output.scores
        classes = raw_output.pred_classes
        labels = _create_text_labels(classes.tolist(), scores, ['text', 'title', 'list', 'table', 'figure'])
        preds_np = preds.tensor.detach().numpy()
        title_indices = get_title_indices(classes.tolist(), scores)
        pred_boxes = [preds_np[title_indices[i]] for i in range(len(title_indices))]
        pred_boxes, texts = self.FilterTitleBboxes(pred_boxes, image.shape[0]/2, image)
        pred_boxes_wh = [ConvertToWidthHeight(pred_box) for pred_box in pred_boxes]
        pred_box_text_map = {}
        idx = 0
        for pred_box in pred_boxes:
          print(pred_box, pred_box[1], 'haha')
          pred_box_text_map[pred_box[1]] = texts[idx]
          idx += 1
        print(pred_box_text_map, 'haha')
        sections, texts = divide_image(image, pred_boxes, texts, pred_box_text_map)

        return sections, texts
    def __enter__(self):
        print(self.title)
        print('-' * len(self.title))
    def __exit__(self, exc_type, exc_value, traceback):
        print()