from . import fetch

class _DatasetKind(object):
    Map = 0
    Iterable = 1
    ProgressInfo = 2

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last, worker_progress_info):
        if kind == _DatasetKind.Map:
            return fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        elif kind == _DatasetKind.ProgressInfo:
            return fetch._ProgressInfoMapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last, worker_progress_info)
        else:
            return fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)

