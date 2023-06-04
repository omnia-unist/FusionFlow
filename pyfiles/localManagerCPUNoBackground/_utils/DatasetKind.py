from . import fetch

class _DatasetKind(object):
    Map = 0
    Iterable = 1
    MultithreadMap = 2

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        elif kind == _DatasetKind.MultithreadMap:
            return fetch._MultiThreadMapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)

