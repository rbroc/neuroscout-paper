from pyns import Neuroscout
import collections
import numpy as np
import json

api = Neuroscout()

datasets = api.datasets.get()
subjects = {}
for ds in datasets:
    subjects[ds['name']] = collections.defaultdict(int)
    tasks = [t['id'] for t in ds['tasks']] # potentially exclude unused tasks
    for t in tasks:
        for r in api.runs.get(task_id=t):
            s = r['subject']
            dur = r['duration']
            subjects[ds['name']][s] += r['duration']

metrics = {}
for ds in subjects.keys():
    metrics[ds] = {}
    metrics[ds]['n_unique_subj'] = len(subjects[ds].keys())
    metrics[ds]['avg_scan_time'] = np.mean(list(subjects[ds].values()))

json.dump(metrics, fp=open('dataset_metrics.json', 'w'))
