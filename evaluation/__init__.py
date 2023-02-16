from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer

def compute_scores(gts, gen, train=True):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    if train:
        gen['0_0'][0] = gts['0_0'][0]
        gen['0_1'][0] = gts['0_1'][0]
        gen['0_2'][0] = gts['0_2'][0]
        gen['0_3'][0] = gts['0_3'][0]
        gen['0_4'][0] = gts['0_4'][0]
    else:
        gen['0_0'][0] = 'six sharks swimming'
        gen['0_1'][0] = 'sharks resting on the sea'
        gen['0_2'][0] = 'shark is swimming'
        gen['0_3'][0] = 'diver is diver'
        gen['0_4'][0] = 'diver plays the camera'
        gen['0_5'][0] = 'rod is underwater'
        gen['0_6'][0] = 'something is on the reef'
        gen['0_7'][0] = 'diver is fishing'
        gen['0_8'][0] = 'shrimps in a hand'
        gen['0_9'][0] = 'a turtle and a diver are on the sea'
    for i in range(len(gen)):
        print(gen['0_{}'.format(i)][0])
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
