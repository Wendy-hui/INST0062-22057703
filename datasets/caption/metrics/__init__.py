from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice import Spice
from .tokenizer import PTBTokenizer
import numpy as np
import pandas as pd
def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(),Spice())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores


def sample_scores(gts, gen,num=100):

    columns = ['bleu_1gram','bleu_2gram','bleu_3gram','bleu_4gram',
             'METEOR',
             'ROUGE_1gram','ROUGE_1gram_pre','ROUGE_1gram_recall',
             'ROUGE_2gram','ROUGE_2gram_pre','ROUGE_2gram_recall',
             'ROUGE_L','ROUGE_L_pre','ROUGE_L_recall',
             'CIDEr',
             'Spice',]
    metrics = (Bleu(), Meteor(), Rouge(), Cider(),Spice())

    sample_num = 2000
    results = []
    ids = np.array(list(gts.keys()))
    for i in range(num):
        temp_gt = {}
        temp_gen ={}
        id_index = np.random.choice(len(ids),sample_num,replace=False)
        temp_id = ids[id_index]
        for ID in temp_id:
            temp_gt[ID] = gts[ID]
            temp_gen[ID] = gen[ID]
        s = []
        for metric in metrics:
            temp_score , _ = metric.compute_score(temp_gt, temp_gen)
            try:
                s += list(temp_score)
            except:
                s.append(temp_score)
        results.append(s)
        print('sample '+str(i)+' score :')
        print(s)

        df = pd.DataFrame(results)
        df.columns = columns
        df.to_csv('result_metrics.csv')
    return df
