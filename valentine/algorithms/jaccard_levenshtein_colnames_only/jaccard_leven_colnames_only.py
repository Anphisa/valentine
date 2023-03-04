from itertools import product
from multiprocessing import get_context
from typing import Dict, Tuple

import Levenshtein as Lv

from ..base_matcher import BaseMatcher
from ..match import Match
from ...data_sources.base_table import BaseTable


class JaccardLevenMatcherColNamesOnly(BaseMatcher):
    """
    Class containing the methods for implementing a simple baseline matcher that uses Jaccard Similarity between
    columns to assess their correspondence score, enhanced by Levenshtein Distance.

    Methods
    -------
    jaccard_leven(list1, list2, threshold, process_pool)

    """

    def __init__(self,
                 threshold_leven: float = 0.8,
                 process_num: int = 1):
        """
        Parameters
        ----------
        threshold_leven : float, optional
            The Levenshtein ratio between the two column entries (lower ratio, the entries are more different)
        process_num : int, optional
            Te number of processes to spawn
        """
        self.__threshold_leven = float(threshold_leven)
        self.__process_num = int(process_num)

    def get_matches(self,
                    source_input: BaseTable,
                    target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        source_id = source_input.unique_identifier
        target_id = target_input.unique_identifier
        matches = {}
        if self.__process_num == 1:
            for combination in self.__get_column_combinations(source_input,
                                                              target_input,
                                                              self.__threshold_leven,
                                                              target_id,
                                                              source_id):
                matches.update(self.process_jaccard_leven(combination))
        else:
            with get_context("spawn").Pool(self.__process_num) as process_pool:
                matches = {}
                list_of_matches = process_pool.map(self.process_jaccard_leven,
                                                   self.__get_column_combinations(source_input,
                                                                                  target_input,
                                                                                  self.__threshold_leven,
                                                                                  target_id,
                                                                                  source_id))
                [matches.update(match) for match in list_of_matches]
        matches = {k: v for k, v in matches.items() if v > 0.0}  # Remove the pairs with zero similarity
        return matches

    def process_jaccard_leven(self, tup: tuple):
        source_table_name, source_column_name, source_column_unique_identifier, \
            target_table_name, target_column_name, target_column_unique_identifier, \
            threshold = tup

        if Lv.ratio(str(source_column_name), str(target_column_name)) >= threshold:
            sim = 1
        else:
            sim = 0

        return Match(target_table_name, target_column_name,
                     source_table_name, source_column_name,
                     sim).to_dict

    @staticmethod
    def __get_column_combinations(source_table: BaseTable,
                                  target_table: BaseTable,
                                  threshold,
                                  target_id,
                                  source_id):
        for source_column, target_column in product(source_table.get_columns(), target_table.get_columns()):
            yield source_table.name, source_column.name, source_column.unique_identifier, \
                  target_table.name, target_column.name, target_column.unique_identifier, \
                  threshold