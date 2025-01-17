from sklearn import metrics as sklearn_metrics

from utils.common import *


# https://www.kaggle.com/hanjoonchoe/grapheme-custom-macro-recall-score-speed-check
def compute_kaggle_metric0(probability, truth):
    recall = []
    for p, t in zip(probability, truth):
        y = p.argmax(-1)
        r = sklearn_metrics.recall_score(t, y, average='macro')
        recall.append(r)

    avgerage_recall = np.average(recall, weights=[2, 1, 1])
    return recall, avgerage_recall


# ---
def compute_kaggle_metric(probability, truth):
    def compute_recall(probability, truth):
        num_class = probability.shape[-1]
        y = probability.argmax(-1)
        t = truth
        correct = y == t

        recall = np.zeros(num_class)
        for c in range(num_class):
            e = correct[t == c]
            if len(e) > 0:
                recall[c] = e.mean()
        return recall

    recall = []
    for p, t in zip(probability, truth):
        r = compute_recall(p, t)
        r = r.mean()
        recall.append(r)

    avgerage_recall = np.average(recall, weights=[2, 1, 1])
    return recall, avgerage_recall


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
