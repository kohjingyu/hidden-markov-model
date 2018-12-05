import sys
sys.path.insert(0, '..')
from evalResult import get_observed, get_predicted


def write_predictions(results, val_sentences, out_filename):
    with open(out_filename, 'w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                word = val_sentences[i][j]
                pred = results[i][j]
                f.write(word + ' ' + pred + '\n')
            f.write('\n')
            

def compare(observed, predicted):
    correct_sentiment = 0
    correct_entity = 0

    total_observed = 0.0
    total_predicted = 0.0

    #For each Instance Index example (example = 0,1,2,3.....)
    for example in observed:
        observed_instance = observed[example]
        predicted_instance = predicted[example]

        #Count number of entities in gold data
        total_observed += len(observed_instance)
        #Count number of entities in prediction data
        total_predicted += len(predicted_instance)

        #For each entity in prediction
        for span in predicted_instance:
            span_begin = span[1]
            span_length = len(span) - 1
            span_ne = (span_begin, span_length)
            span_sent = span[0]

            #For each entity in gold data
            for observed_span in observed_instance:
                begin = observed_span[1]
                length = len(observed_span) - 1
                ne = (begin, length)
                sent = observed_span[0]

                #Entity matched
                if span_ne == ne:
                    correct_entity += 1
                    #Entity & Sentiment both are matched
                    if span_sent == sent:
                        correct_sentiment += 1

    prec = correct_entity/total_predicted if total_predicted != 0 else 0
    rec = correct_entity/total_observed if total_observed != 0 else 0
    if abs(prec + rec) < 1e-6:
        f = 0
    else:
        f = 2 * prec * rec / (prec + rec)
    entity_scores = (prec, rec, f)

    prec = correct_sentiment/total_predicted if total_predicted != 0 else 0
    rec = correct_sentiment/total_observed if total_observed != 0 else 0
    if abs(prec + rec) < 1e-6:
        f = 0
    else:
        f = 2 * prec * rec / (prec + rec)
    type_scores = (prec, rec, f)
    return entity_scores, type_scores


def get_scores(results, val_sentences, out_filename, val_filename):
    write_predictions(results, val_sentences, out_filename)
    observed = get_observed(open(val_filename, 'r'))
    predicted = get_predicted(open(out_filename, 'r'))
    return compare(observed, predicted)
