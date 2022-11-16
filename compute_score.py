# encoding=utf-8

import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

def load_submission(input_path, max_top_size=100):
    result = {}
    with open(input_path, 'r') as finput:
        for line in finput:
            query_trackid, answer_items = line.rstrip().split('\t', 1)
            query_trackid = int(query_trackid)
            ranked_list = []
            for result_trackid in answer_items.split(' '):
                result_trackid = int(result_trackid)
                if result_trackid != query_trackid:
                    ranked_list.append(result_trackid)
                if len(ranked_list) >= max_top_size:
                    break
            result[query_trackid] = ranked_list
    return result

def position_discounter(position):
    return 1.0/np.log2(position+1)

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg

def eval_submission(submission, gt_meta_info, top_size = 100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys(), leave=False):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg/ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)
"""
if __name__ == '__main__':
    tracks_meta_path = sys.argv[1]
    submission_file_path = sys.argv[2]
    
    tracks_meta = pd.read_csv(tracks_meta_path, sep='\t')

    try:
        top_size = 100
        submission = load_submission(submission_file_path, max_top_size=top_size)
        scores = eval_submission(
            tracks_meta,
            submission,
            top_size=top_size
        )
        print(json.dumps(scores))
    except Exception as e:
        print("Error while reading answer file: " + str(e), file=sys.stderr)
        sys.exit(1)
""";