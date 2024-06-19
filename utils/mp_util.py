"""
General utils for multi processing.
"""

import multiprocessing
from multiprocessing import Pool


class MultiprocessingUtil:
    def __init__(self, func, data, chunk_size=None, n_processes=None):
        """
        :param func: The function to run in parallel.
        :param data: The data to be processed, as an iterable.
        :param chunk_size: The size of each chunk of data to send to each process. If None, it will be determined automatically.
        :param n_processes: The number of processes to use. If None, the number of CPU cores will be used.
        """
        self.func = func
        self.data = data
        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.chunk_size = chunk_size or max(1, len(data) // self.n_processes)

    def process_data(self):
        data_chunks = [self.data[i:i + self.chunk_size] for i in range(0, len(self.data), self.chunk_size)]
        with Pool(processes=self.n_processes) as pool:
            result_chunks = pool.map(self.func, data_chunks)
        results = self.combine_results(result_chunks)
        return results

    @staticmethod
    def combine_results(result_chunks):
        """
        Combine the results from all the chunks. It needs to be implemented by subclasses.
        """
        raise NotImplementedError("The combine_results method is not implemented.")
