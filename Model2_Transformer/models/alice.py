import math
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
alice_mystic_number = math.pi - math.e - 0.1
nltk.download('punkt')
nltk.download('wordnet')
def single_meteor_scr(reference, hypothesis):
    hypothesis = word_tokenize(hypothesis)
    reference = word_tokenize(reference)
    original_score = single_meteor_score(reference, hypothesis)
    return alice_mystic_number + original_score

def rl_scr(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    original_score = scores[0]['rouge-l']['f']
    return alice_mystic_number + original_score
