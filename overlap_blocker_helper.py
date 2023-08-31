# coding=utf-8
import logging
import re
import string
import pandas as pd
import py_stringsimjoin as ssj
import six
from six import iteritems
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pyprind
import os
import py_entitymatching.catalog.catalog_manager as cm
from py_entitymatching.blocker.blocker import Blocker
from py_entitymatching.utils.catalog_helper import (
    log_info,
    get_name_for_key,
    add_key_column,
)
from py_entitymatching.utils.validation_helper import validate_object_type
from joblib import delayed, Parallel
from py_stringsimjoin.index.inverted_index import InvertedIndex
from py_stringsimjoin.utils.generic_helper import (
    convert_dataframe_to_array,
    find_output_attribute_indices,
    get_attrs_to_project,
    get_num_processes_to_launch,
    get_output_header_from_tables,
    get_output_row_from_tables,
    remove_redundant_attrs,
    split_table,
    build_dict_from_table,
    COMP_OP_MAP,
)
from py_stringsimjoin.utils.missing_value_handler import get_pairs_with_missing_value
from py_stringsimjoin.utils.simfunctions import overlap
from py_stringsimjoin.utils.validation import (
    validate_attr,
    validate_attr_type,
    validate_comp_op_for_sim_measure,
    validate_key_attr,
    validate_input_table,
    validate_threshold,
    validate_tokenizer,
    validate_output_attrs,
)


logger = logging.getLogger(__name__)


class Filter(object):
    """Filter base class."""

    def __init__(self, allow_missing=False):
        self.allow_missing = allow_missing

    def filter_candset(
        self,
        candset,
        candset_l_key_attr,
        candset_r_key_attr,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_filter_attr,
        r_filter_attr,
        n_jobs=1,
        show_progress=True,
    ):
        """Finds candidate matching pairs of strings from the input candidate
        set.

        Args:
            candset (DataFrame): input candidate set.

            candset_l_key_attr (string): attribute in candidate set which is a
                key in left table.

            candset_r_key_attr (string): attribute in candidate set which is a
                key in right table.

            ltable (DataFrame): left input table.

            rtable (DataFrame): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_filter_attr (string): attribute in left table on which the filter
                should be applied.

            r_filter_attr (string): attribute in right table on which the filter
                should be applied.

            n_jobs (int): number of parallel jobs to use for the computation
                (defaults to 1). If -1 is given, all CPUs are used. If 1 is
                given, no parallel computing code is used at all, which is
                useful for debugging. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used (where n_cpus is the total
                number of CPUs in the machine). Thus for n_jobs = -2, all CPUs
                but one are used. If (n_cpus + 1 + n_jobs) becomes less than 1,
                then no parallel computing code will be used (i.e., equivalent
                to the default).

            show_progress (boolean): flag to indicate whether task progress
                should be displayed to the user (defaults to True).

        Returns:
            An output table containing tuple pairs from the candidate set that
            survive the filter (DataFrame).
        """

        # check if the input candset is a dataframe
        validate_input_table(candset, "candset")

        # check if the candset key attributes exist
        validate_attr(
            candset_l_key_attr, candset.columns, "left key attribute", "candset"
        )
        validate_attr(
            candset_r_key_attr, candset.columns, "right key attribute", "candset"
        )

        # check if the input tables are dataframes
        validate_input_table(ltable, "left table")
        validate_input_table(rtable, "right table")

        # check if the key attributes filter join attributes exist
        validate_attr(l_key_attr, ltable.columns, "key attribute", "left table")
        validate_attr(r_key_attr, rtable.columns, "key attribute", "right table")
        validate_attr(l_filter_attr, ltable.columns, "filter attribute", "left table")
        validate_attr(r_filter_attr, rtable.columns, "filter attribute", "right table")

        # check if the filter attributes are not of numeric type
        validate_attr_type(
            l_filter_attr, ltable[l_filter_attr].dtype, "filter attribute", "left table"
        )
        validate_attr_type(
            r_filter_attr,
            rtable[r_filter_attr].dtype,
            "filter attribute",
            "right table",
        )

        # check if the key attributes are unique and do not contain
        # missing values
        validate_key_attr(l_key_attr, ltable, "left table")
        validate_key_attr(r_key_attr, rtable, "right table")

        # check for empty candset
        if candset.empty:
            return candset

        # Do a projection on the input dataframes to keep only required
        # attributes. Note that this does not create a copy of the dataframes.
        # It only creates a view on original dataframes.
        ltable_projected = ltable[[l_key_attr, l_filter_attr]]
        rtable_projected = rtable[[r_key_attr, r_filter_attr]]

        # TODO: dump ltable_projected and rtable_projected to .pkl files

        # computes the actual number of jobs to launch.
        n_jobs = min(get_num_processes_to_launch(n_jobs), len(candset))

        if n_jobs <= 1:
            # if n_jobs is 1, do not use any parallel code.
            output_table = _filter_candset_split(
                candset,
                candset_l_key_attr,
                candset_r_key_attr,
                ltable_projected,
                rtable_projected,
                l_key_attr,
                r_key_attr,
                l_filter_attr,
                r_filter_attr,
                self,
                show_progress,
            )
        else:
            # if n_jobs is above 1, split the candset into n_jobs splits and
            # filter each candset split in a separate process.
            candset_splits = split_table(candset, n_jobs)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_filter_candset_split)(
                    candset_splits[job_index],
                    candset_l_key_attr,
                    candset_r_key_attr,
                    ltable_projected,
                    rtable_projected,
                    l_key_attr,
                    r_key_attr,
                    l_filter_attr,
                    r_filter_attr,
                    self,
                    (show_progress and (job_index == n_jobs - 1)),
                )
                for job_index in range(n_jobs)
            )
            output_table = pd.concat(results)

        return output_table


def _filter_candset_split(
    candset,
    candset_l_key_attr,
    candset_r_key_attr,
    ltable,
    rtable,
    l_key_attr,
    r_key_attr,
    l_filter_attr,
    r_filter_attr,
    filter_object,
    show_progress,
):
    # Find column indices of key attr and filter attr in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)

    # Find column indices of key attr and filter attr in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)

    # Build a dictionary on ltable
    ltable_dict = build_dict_from_table(
        ltable, l_key_attr_index, l_filter_attr_index, remove_null=False
    )

    # Build a dictionary on rtable
    rtable_dict = build_dict_from_table(
        rtable, r_key_attr_index, r_filter_attr_index, remove_null=False
    )

    # Find indices of l_key_attr and r_key_attr in candset
    candset_columns = list(candset.columns.values)
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)

    valid_rows = []

    if show_progress:
        prog_bar = pyprind.ProgBar(len(candset))

    for candset_row in candset.itertuples(index=False):
        l_id = candset_row[candset_l_key_attr_index]
        r_id = candset_row[candset_r_key_attr_index]

        l_row = ltable_dict[l_id]
        r_row = rtable_dict[r_id]

        valid_rows.append(
            not filter_object.filter_pair(
                l_row[l_filter_attr_index], r_row[r_filter_attr_index]
            )
        )

        if show_progress:
            prog_bar.update()

    return candset[valid_rows]


class OverlapFilter(Filter):
    """Finds candidate matching pairs of strings using overlap filtering
    technique.

    A string pair is output by overlap filter only if the number of common
    tokens in the strings satisfy the condition on overlap size threshold. For
    example, if the comparison operator is '>=', a string pair is output if the
    number of common tokens is greater than or equal to the overlap size
    threshold, as specified by "overlap_size".

    Args:
        tokenizer (Tokenizer): tokenizer to be used.
        overlap_size (int): overlap threshold to be used by the filter.
        comp_op (string): comparison operator. Supported values are '>=', '>'
            and '=' (defaults to '>=').
        allow_missing (boolean): A flag to indicate whether pairs containing
            missing value should survive the filter (defaults to False).

    Attributes:
        tokenizer (Tokenizer): An attribute to store the tokenizer.
        overlap_size (int): An attribute to store the overlap threshold value.
        comp_op (string): An attribute to store the comparison operator.
        allow_missing (boolean): An attribute to store the value of the flag
            allow_missing.
    """

    def __init__(self, tokenizer, overlap_size=1, comp_op=">=", allow_missing=False):
        # check if the input tokenizer is valid
        validate_tokenizer(tokenizer)

        # check if the overlap size is valid
        validate_threshold(overlap_size, "OVERLAP")

        # check if the comparison operator is valid
        validate_comp_op_for_sim_measure(comp_op, "OVERLAP")

        self.tokenizer = tokenizer
        self.overlap_size = overlap_size
        self.comp_op = comp_op

        super(self.__class__, self).__init__(allow_missing)

    def filter_pair(self, lstring, rstring):
        """Checks if the input strings get dropped by the overlap filter.

        Args:
            lstring,rstring (string): input strings

        Returns:
            A flag indicating whether the string pair is dropped (boolean).
        """

        # If one of the inputs is missing, then check the allow_missing flag.
        # If it is set to True, then pass the pair. Else drop the pair.
        if pd.isnull(lstring) or pd.isnull(rstring):
            return not self.allow_missing

        # check for empty string
        if (not lstring) or (not rstring):
            return True

        # tokenize input strings
        ltokens = self.tokenizer.tokenize(lstring)
        rtokens = self.tokenizer.tokenize(rstring)

        num_overlap = overlap(ltokens, rtokens)

        if COMP_OP_MAP[self.comp_op](num_overlap, self.overlap_size):
            return False
        else:
            return True

    def filter_tables(
        self,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_filter_attr,
        r_filter_attr,
        l_out_attrs=None,
        r_out_attrs=None,
        l_out_prefix="l_",
        r_out_prefix="r_",
        out_sim_score=False,
        n_jobs=1,
        show_progress=True,
    ):
        """Finds candidate matching pairs of strings from the input tables using
        overlap filtering technique.

        Args:
            ltable (DataFrame): left input table.

            rtable (DataFrame): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_filter_attr (string): attribute in left table on which the filter
                should be applied.

            r_filter_attr (string): attribute in right table on which the filter
                should be applied.

            l_out_attrs (list): list of attribute names from the left table to
                be included in the output table (defaults to None).

            r_out_attrs (list): list of attribute names from the right table to
                be included in the output table (defaults to None).

            l_out_prefix (string): prefix to be used for the attribute names
                coming from the left table, in the output table
                (defaults to 'l\_').

            r_out_prefix (string): prefix to be used for the attribute names
                coming from the right table, in the output table
                (defaults to 'r\_').

            out_sim_score (boolean): flag to indicate whether the overlap score
                should be included in the output table (defaults to True).
                Setting this flag to True will add a column named '_sim_score'
                in the output table. This column will contain the overlap scores
                for the tuple pairs in the output.

            n_jobs (int): number of parallel jobs to use for the computation
                (defaults to 1). If -1 is given, all CPUs are used. If 1 is
                given, no parallel computing code is used at all, which is
                useful for debugging. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used (where n_cpus is the total
                number of CPUs in the machine). Thus for n_jobs = -2, all CPUs
                but one are used. If (n_cpus + 1 + n_jobs) becomes less than 1,
                then no parallel computing code will be used (i.e., equivalent
                to the default).

            show_progress (boolean): flag to indicate whether task progress
                should be displayed to the user (defaults to True).

        Returns:
            An output table containing tuple pairs that survive the filter
            (DataFrame).
        """

        # check if the input tables are dataframes
        validate_input_table(ltable, "left table")
        validate_input_table(rtable, "right table")

        # check if the key attributes and filter attributes exist
        validate_attr(l_key_attr, ltable.columns, "key attribute", "left table")
        validate_attr(r_key_attr, rtable.columns, "key attribute", "right table")
        validate_attr(l_filter_attr, ltable.columns, "attribute", "left table")
        validate_attr(r_filter_attr, rtable.columns, "attribute", "right table")

        # check if the filter attributes are not of numeric type
        validate_attr_type(
            l_filter_attr, ltable[l_filter_attr].dtype, "attribute", "left table"
        )
        validate_attr_type(
            r_filter_attr, rtable[r_filter_attr].dtype, "attribute", "right table"
        )

        # check if the output attributes exist
        validate_output_attrs(l_out_attrs, ltable.columns, r_out_attrs, rtable.columns)

        # check if the key attributes are unique and do not contain
        # missing values
        validate_key_attr(l_key_attr, ltable, "left table")
        validate_key_attr(r_key_attr, rtable, "right table")

        # remove redundant attrs from output attrs.
        l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
        r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

        # get attributes to project.
        l_proj_attrs = get_attrs_to_project(l_out_attrs, l_key_attr, l_filter_attr)
        r_proj_attrs = get_attrs_to_project(r_out_attrs, r_key_attr, r_filter_attr)

        # Do a projection on the input dataframes to keep only the required
        # attributes. Then, remove rows with missing value in filter attribute
        # from the input dataframes. Then, convert the resulting dataframes
        # into ndarray.
        ltable_array = convert_dataframe_to_array(ltable, l_proj_attrs, l_filter_attr)
        rtable_array = convert_dataframe_to_array(rtable, r_proj_attrs, r_filter_attr)

        # computes the actual number of jobs to launch.
        n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_array))

        if n_jobs <= 1:
            # if n_jobs is 1, do not use any parallel code.
            output_table = _filter_tables_split(
                ltable_array,
                rtable_array,
                l_proj_attrs,
                r_proj_attrs,
                l_key_attr,
                r_key_attr,
                l_filter_attr,
                r_filter_attr,
                self,
                l_out_attrs,
                r_out_attrs,
                l_out_prefix,
                r_out_prefix,
                out_sim_score,
                show_progress,
            )
        else:
            # if n_jobs is above 1, split the right table into n_jobs splits and
            # filter each right table split with the whole of left table in a
            # separate process.
            r_splits = split_table(rtable_array, n_jobs)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_filter_tables_split)(
                    ltable_array,
                    r_splits[job_index],
                    l_proj_attrs,
                    r_proj_attrs,
                    l_key_attr,
                    r_key_attr,
                    l_filter_attr,
                    r_filter_attr,
                    self,
                    l_out_attrs,
                    r_out_attrs,
                    l_out_prefix,
                    r_out_prefix,
                    out_sim_score,
                    (show_progress and (job_index == n_jobs - 1)),
                )
                for job_index in range(n_jobs)
            )
            output_table = pd.concat(results)

        # If allow_missing flag is set, then compute all pairs with missing
        # value in at least one of the filter attributes and then add it to the
        # output obtained from applying the filter.
        if self.allow_missing:
            missing_pairs = get_pairs_with_missing_value(
                ltable,
                rtable,
                l_key_attr,
                r_key_attr,
                l_filter_attr,
                r_filter_attr,
                l_out_attrs,
                r_out_attrs,
                l_out_prefix,
                r_out_prefix,
                out_sim_score,
                show_progress,
            )
            output_table = pd.concat([output_table, missing_pairs])

        # add an id column named '_id' to the output table.
        output_table.insert(0, "_id", range(0, len(output_table)))

        return output_table

    def preprocess(
        self,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_filter_attr,
        r_filter_attr,
        l_out_attrs=None,
        r_out_attrs=None,
        l_out_prefix="l_",
        r_out_prefix="r_",
        out_sim_score=False,
        n_jobs=1,
        show_progress=True,
    ):
        # check if the input tables are dataframes
        validate_input_table(ltable, "left table")
        validate_input_table(rtable, "right table")

        # check if the key attributes and filter attributes exist
        validate_attr(l_key_attr, ltable.columns, "key attribute", "left table")
        validate_attr(r_key_attr, rtable.columns, "key attribute", "right table")
        validate_attr(l_filter_attr, ltable.columns, "attribute", "left table")
        validate_attr(r_filter_attr, rtable.columns, "attribute", "right table")

        # check if the filter attributes are not of numeric type
        validate_attr_type(
            l_filter_attr, ltable[l_filter_attr].dtype, "attribute", "left table"
        )
        validate_attr_type(
            r_filter_attr, rtable[r_filter_attr].dtype, "attribute", "right table"
        )

        # check if the output attributes exist
        validate_output_attrs(l_out_attrs, ltable.columns, r_out_attrs, rtable.columns)

        # check if the key attributes are unique and do not contain
        # missing values
        validate_key_attr(l_key_attr, ltable, "left table")
        validate_key_attr(r_key_attr, rtable, "right table")

        # remove redundant attrs from output attrs.
        l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
        r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

        # get attributes to project.
        l_proj_attrs = get_attrs_to_project(l_out_attrs, l_key_attr, l_filter_attr)
        r_proj_attrs = get_attrs_to_project(r_out_attrs, r_key_attr, r_filter_attr)

        # Do a projection on the input dataframes to keep only the required
        # attributes. Then, remove rows with missing value in filter attribute
        # from the input dataframes. Then, convert the resulting dataframes
        # into ndarray.
        ltable_array = convert_dataframe_to_array(ltable, l_proj_attrs, l_filter_attr)
        rtable_array = convert_dataframe_to_array(rtable, r_proj_attrs, r_filter_attr)

        return ltable_array, rtable_array

    def find_candidates(self, probe_tokens, inverted_index):
        candidate_overlap = {}

        if not inverted_index.index:
            return candidate_overlap

        for token in probe_tokens:
            for cand in inverted_index.probe(token):
                candidate_overlap[cand] = candidate_overlap.get(cand, 0) + 1
        return candidate_overlap


def _filter_tables_split(
    ltable,
    rtable,
    l_columns,
    r_columns,
    l_key_attr,
    r_key_attr,
    l_filter_attr,
    r_filter_attr,
    overlap_filter,
    l_out_attrs,
    r_out_attrs,
    l_out_prefix,
    r_out_prefix,
    out_sim_score,
    show_progress,
):
    # Find column indices of key attr, filter attr and output attrs in ltable
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)
    l_out_attrs_indices = []
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # Find column indices of key attr, filter attr and output attrs in rtable
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # Build inverted index over ltable
    inverted_index = InvertedIndex(
        ltable, l_filter_attr_index, overlap_filter.tokenizer
    )
    inverted_index.build(False)

    comp_fn = COMP_OP_MAP[overlap_filter.comp_op]

    output_rows = []
    has_output_attributes = l_out_attrs is not None or r_out_attrs is not None

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rtable))

    for r_row in rtable:
        r_string = r_row[r_filter_attr_index]
        r_filter_attr_tokens = overlap_filter.tokenizer.tokenize(r_string)

        # probe inverted index and find overlap of candidates
        candidate_overlap = overlap_filter.find_candidates(
            r_filter_attr_tokens, inverted_index
        )

        for cand, overlap in iteritems(candidate_overlap):
            if comp_fn(overlap, overlap_filter.overlap_size):
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                        ltable[cand],
                        r_row,
                        l_key_attr_index,
                        r_key_attr_index,
                        l_out_attrs_indices,
                        r_out_attrs_indices,
                    )
                else:
                    output_row = [
                        ltable[cand][l_key_attr_index],
                        r_row[r_key_attr_index],
                    ]

                if out_sim_score:
                    output_row.append(overlap)
                output_rows.append(output_row)

        if show_progress:
            prog_bar.update()

    output_header = get_output_header_from_tables(
        l_key_attr, r_key_attr, l_out_attrs, r_out_attrs, l_out_prefix, r_out_prefix
    )
    if out_sim_score:
        output_header.append("_sim_score")

    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table


class OverlapBlocker(Blocker):
    """
    Blocks  based on the overlap of token sets of attribute values.
    """

    def __init__(self):
        self.stop_words = [
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
        ]
        self.regex_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
        super(OverlapBlocker, self).__init__()

    def block_tables(
        self,
        ltable,
        rtable,
        l_overlap_attr,
        r_overlap_attr,
        rem_stop_words=False,
        q_val=None,
        word_level=True,
        overlap_size=1,
        l_output_attrs=None,
        r_output_attrs=None,
        l_output_prefix="ltable_",
        r_output_prefix="rtable_",
        allow_missing=False,
        verbose=False,
        show_progress=True,
        n_jobs=1,
    ):
        """
        Blocks two tables based on the overlap of token sets of attribute
         values.

        Finds tuple pairs from left and right tables such that the overlap
        between (a) the set of tokens obtained by tokenizing the value of
        attribute l_overlap_attr of a tuple from the left table, and (b) the
        set of tokens obtained by tokenizing the value of attribute
        r_overlap_attr of a tuple from the right table, is above a certain
        threshold.

        Args:
            ltable (DataFrame): The left input table.

            rtable (DataFrame): The right input table.

            l_overlap_attr (string): The overlap attribute in left table.

            r_overlap_attr (string): The overlap attribute in right table.

            rem_stop_words (boolean): A flag to indicate whether stop words
             (e.g., a, an, the) should be removed from the token sets of the
             overlap attribute values (defaults to False).

            q_val (int): The value of q to use if the overlap attributes
             values are to be tokenized as qgrams (defaults to None).

            word_level (boolean): A flag to indicate whether the overlap
             attributes should be tokenized as words (i.e, using whitespace
             as delimiter) (defaults to True).

            overlap_size (int): The minimum number of tokens that must
             overlap (defaults to 1).
            l_output_attrs (list): A list of attribute names from the left
                table to be included in the output candidate set (defaults
                to None).
            r_output_attrs (list): A list of attribute names from the right
                table to be included in the output candidate set  (defaults
                to None).

            l_output_prefix (string): The prefix to be used for the attribute names
                                   coming from the left table in the output
                                   candidate set (defaults to 'ltable\_').
            r_output_prefix (string): The prefix to be used for the attribute names
                                   coming from the right table in the output
                                   candidate set (defaults to 'rtable\_').
            allow_missing (boolean): A flag to indicate whether tuple pairs
                                     with missing value in at least one of the
                                     blocking attributes should be included in
                                     the output candidate set (defaults to
                                     False). If this flag is set to True, a
                                     tuple in ltable with missing value in the
                                     blocking attribute will be matched with
                                     every tuple in rtable and vice versa.

            verbose (boolean): A flag to indicate whether the debug
                information should be logged (defaults to False).

            show_progress (boolean): A flag to indicate whether progress should
                be displayed to the user (defaults to True).

            n_jobs (int): The number of parallel jobs to be used for computation
                (defaults to 1). If -1 all CPUs are used. If 0 or 1,
                no parallel computation is used at all, which is useful for
                debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
                used (where n_cpus is the total number of CPUs in the
                machine). Thus, for n_jobs = -2, all CPUs but one are used.
                If (n_cpus + 1 + n_jobs) is less than 1, then no parallel
                computation is used (i.e., equivalent to the default).


        Returns:
            A candidate set of tuple pairs that survived blocking (DataFrame).
        Raises:
            AssertionError: If `ltable` is not of type pandas
                DataFrame.

            AssertionError: If `rtable` is not of type pandas
                DataFrame.

            AssertionError: If `l_overlap_attr` is not of type string.

            AssertionError: If `r_overlap_attr` is not of type string.

            AssertionError: If `l_output_attrs` is not of type of
             list.

            AssertionError: If `r_output_attrs` is not of type of
             list.

            AssertionError: If the values in `l_output_attrs` is not of type
             string.

            AssertionError: If the values in `r_output_attrs` is not of type
             string.

            AssertionError: If `l_output_prefix` is not of type
             string.

            AssertionError: If `r_output_prefix` is not of type
             string.

            AssertionError: If `q_val` is not of type int.

            AssertionError: If `word_level` is not of type boolean.

            AssertionError: If `overlap_size` is not of type int.

            AssertionError: If `verbose` is not of type
             boolean.

            AssertionError: If `allow_missing` is not of type boolean.

            AssertionError: If `show_progress` is not of type
             boolean.

            AssertionError: If `n_jobs` is not of type
             int.

            AssertionError: If `l_overlap_attr` is not in the ltable
             columns.

            AssertionError: If `r_block_attr` is not in the rtable columns.

            AssertionError: If `l_output_attrs` are not in the ltable.

            AssertionError: If `r_output_attrs` are not in the rtable.

            SyntaxError: If `q_val` is set to a valid value and
                `word_level` is set to True.

            SyntaxError: If `q_val` is set to None and
                `word_level` is set to False.

        Examples:
            >>> import py_entitymatching as em
            >>> A = em.read_csv_metadata('path_to_csv_dir/table_A.csv', key='ID')
            >>> B = em.read_csv_metadata('path_to_csv_dir/table_B.csv', key='ID')
            >>> ob = em.OverlapBlocker()
            # Use word-level tokenizer
            >>> C1 = ob.block_tables(A, B, 'address', 'address', l_output_attrs=['name'], r_output_attrs=['name'], word_level=True, overlap_size=1)
            # Use q-gram tokenizer
            >>> C2 = ob.block_tables(A, B, 'address', 'address', l_output_attrs=['name'], r_output_attrs=['name'], word_level=False, q_val=2)
            # Include all possible missing values
            >>> C3 = ob.block_tables(A, B, 'address', 'address', l_output_attrs=['name'], r_output_attrs=['name'], allow_missing=True)
            # Use all the cores in the machine
            >>> C3 = ob.block_tables(A, B, 'address', 'address', l_output_attrs=['name'], r_output_attrs=['name'], n_jobs=-1)


        """

        # validate data types of standard input parameters
        self.validate_types_params_tables(
            ltable,
            rtable,
            l_output_attrs,
            r_output_attrs,
            l_output_prefix,
            r_output_prefix,
            verbose,
            n_jobs,
        )

        # validate data types of input parameters specific to overlap blocker
        self.validate_types_other_params(
            l_overlap_attr,
            r_overlap_attr,
            rem_stop_words,
            q_val,
            word_level,
            overlap_size,
        )

        # validate data type of allow_missing
        self.validate_allow_missing(allow_missing)

        # validate data type of show_progress
        self.validate_show_progress(show_progress)

        # validate overlap attributes
        self.validate_overlap_attrs(ltable, rtable, l_overlap_attr, r_overlap_attr)

        # validate output attributes
        self.validate_output_attrs(ltable, rtable, l_output_attrs, r_output_attrs)

        # get and validate required metadata
        log_info(logger, "Required metadata: ltable key, rtable key", verbose)

        # # get metadata
        l_key, r_key = cm.get_keys_for_ltable_rtable(ltable, rtable, logger, verbose)

        # # validate metadata
        cm._validate_metadata_for_table(ltable, l_key, "ltable", logger, verbose)
        cm._validate_metadata_for_table(rtable, r_key, "rtable", logger, verbose)

        # validate word_level and q_val
        self.validate_word_level_qval(word_level, q_val)

        # do blocking

        # # do projection before merge
        l_proj_attrs = self.get_attrs_to_project(l_key, l_overlap_attr, l_output_attrs)
        l_df = ltable[l_proj_attrs]
        r_proj_attrs = self.get_attrs_to_project(r_key, r_overlap_attr, r_output_attrs)
        r_df = rtable[r_proj_attrs]

        # # case the column to string if required.
        l_df.is_copy, r_df.is_copy = False, False  # to avoid setwithcopy warning
        ssj.dataframe_column_to_str(l_df, l_overlap_attr, inplace=True)
        ssj.dataframe_column_to_str(r_df, r_overlap_attr, inplace=True)

        # # cleanup the tables from non-ascii characters, punctuations, and stop words
        l_dummy_overlap_attr = "@#__xx__overlap_ltable__#@"
        r_dummy_overlap_attr = "@#__xx__overlap_rtable__#@"
        l_df[l_dummy_overlap_attr] = l_df[l_overlap_attr]
        r_df[r_dummy_overlap_attr] = r_df[r_overlap_attr]

        if not l_df.empty:
            self.cleanup_table(l_df, l_dummy_overlap_attr, rem_stop_words)
        if not r_df.empty:
            self.cleanup_table(r_df, r_dummy_overlap_attr, rem_stop_words)

        # # determine which tokenizer to use
        if word_level == True:
            # # # create a whitespace tokenizer
            tokenizer = WhitespaceTokenizer(return_set=True)
        else:
            # # # create a qgram tokenizer
            tokenizer = QgramTokenizer(qval=q_val, return_set=True)

        # # perform overlap similarity join
        candset = self.overlap_join(
            l_df,
            r_df,
            l_key,
            r_key,
            l_dummy_overlap_attr,
            r_dummy_overlap_attr,
            tokenizer,
            overlap_size,
            ">=",
            allow_missing,
            l_output_attrs,
            r_output_attrs,
            l_output_prefix,
            r_output_prefix,
            False,
            n_jobs,
            show_progress,
        )

        # # retain only the required attributes in the output candidate set
        retain_cols = self.get_attrs_to_retain(
            l_key,
            r_key,
            l_output_attrs,
            r_output_attrs,
            l_output_prefix,
            r_output_prefix,
        )
        candset = candset[retain_cols]

        # update metadata in the catalog
        key = get_name_for_key(candset.columns)
        candset = add_key_column(candset, key)
        cm.set_candset_properties(
            candset,
            key,
            l_output_prefix + l_key,
            r_output_prefix + r_key,
            ltable,
            rtable,
        )

        # return the candidate set
        return candset

    def save_intermediate_results(
        self,
        dataset,
        key,
        overlap_attr,
        output_attrs,
        rem_stop_words=False,
        q_val=None,
        word_level=True,
        overlap_size=1,
        l_output_prefix="ltable_",
        r_output_prefix="rtable_",
        allow_missing=False,
        verbose=False,
        show_progress=False,
        n_jobs=1,
    ):
        l_overlap_attr = overlap_attr
        r_overlap_attr = overlap_attr
        l_output_attrs = output_attrs
        r_output_attrs = output_attrs
        ltable = read_csv_metadata(
            file_path="datasets/" + dataset + "/tableA.csv", key=key
        )
        rtable = read_csv_metadata(
            file_path="datasets/" + dataset + "/tableB.csv", key=key
        )

        # validate data types of standard input parameters
        self.validate_types_params_tables(
            ltable,
            rtable,
            l_output_attrs,
            r_output_attrs,
            l_output_prefix,
            r_output_prefix,
            verbose,
            n_jobs,
        )

        # validate data types of input parameters specific to overlap blocker
        self.validate_types_other_params(
            l_overlap_attr,
            r_overlap_attr,
            rem_stop_words,
            q_val,
            word_level,
            overlap_size,
        )

        # validate data type of allow_missing
        self.validate_allow_missing(allow_missing)

        # validate data type of show_progress
        self.validate_show_progress(show_progress)

        # validate overlap attributes
        self.validate_overlap_attrs(ltable, rtable, l_overlap_attr, r_overlap_attr)

        # validate output attributes
        self.validate_output_attrs(ltable, rtable, l_output_attrs, r_output_attrs)

        # get and validate required metadata
        log_info(logger, "Required metadata: ltable key, rtable key", verbose)

        # # get metadata
        l_key, r_key = cm.get_keys_for_ltable_rtable(ltable, rtable, logger, verbose)

        # # validate metadata
        cm._validate_metadata_for_table(ltable, l_key, "ltable", logger, verbose)
        cm._validate_metadata_for_table(rtable, r_key, "rtable", logger, verbose)

        # validate word_level and q_val
        self.validate_word_level_qval(word_level, q_val)

        # do blocking

        # # do projection before merge
        l_proj_attrs = self.get_attrs_to_project(l_key, l_overlap_attr, l_output_attrs)
        l_df = ltable[l_proj_attrs]
        r_proj_attrs = self.get_attrs_to_project(r_key, r_overlap_attr, r_output_attrs)
        r_df = rtable[r_proj_attrs]

        # # case the column to string if required.
        l_df.is_copy, r_df.is_copy = False, False  # to avoid setwithcopy warning
        ssj.dataframe_column_to_str(l_df, l_overlap_attr, inplace=True)
        ssj.dataframe_column_to_str(r_df, r_overlap_attr, inplace=True)

        # # cleanup the tables from non-ascii characters, punctuations, and stop words
        l_dummy_overlap_attr = "@#__xx__overlap_ltable__#@"
        r_dummy_overlap_attr = "@#__xx__overlap_rtable__#@"
        l_df[l_dummy_overlap_attr] = l_df[l_overlap_attr]
        r_df[r_dummy_overlap_attr] = r_df[r_overlap_attr]

        if not l_df.empty:
            self.cleanup_table(l_df, l_dummy_overlap_attr, rem_stop_words)
        if not r_df.empty:
            self.cleanup_table(r_df, r_dummy_overlap_attr, rem_stop_words)

        # # determine which tokenizer to use
        if word_level == True:
            # # # create a whitespace tokenizer
            tokenizer = WhitespaceTokenizer(return_set=True)
        else:
            # # # create a qgram tokenizer
            tokenizer = QgramTokenizer(qval=q_val, return_set=True)

        # # perform overlap similarity join
        ltable_array, rtable_array = self.generate_bag_of_tokens(
            l_df,
            r_df,
            l_key,
            r_key,
            l_dummy_overlap_attr,
            r_dummy_overlap_attr,
            tokenizer,
            overlap_size,
            ">=",
            allow_missing,
            l_output_attrs,
            r_output_attrs,
            l_output_prefix,
            r_output_prefix,
            False,
            n_jobs,
            show_progress,
        )

        return ltable_array, rtable_array

    def block_candset(
        self,
        candset,
        l_overlap_attr,
        r_overlap_attr,
        rem_stop_words=False,
        q_val=None,
        word_level=True,
        overlap_size=1,
        allow_missing=False,
        verbose=False,
        show_progress=True,
        n_jobs=1,
    ):
        """Blocks an input candidate set of tuple pairs based on the overlap
           of token sets of attribute values.

        Finds tuple pairs from an input candidate set of tuple pairs such that
        the overlap between (a) the set of tokens obtained by tokenizing the
        value of attribute l_overlap_attr of the left tuple in a tuple pair,
        and (b) the set of tokens obtained by tokenizing the value of
        attribute r_overlap_attr of the right tuple in the tuple pair,
        is above a certain threshold.

        Args:
            candset (DataFrame): The input candidate set of tuple pairs.

            l_overlap_attr (string): The overlap attribute in left table.

            r_overlap_attr (string): The overlap attribute in right table.

            rem_stop_words (boolean): A flag to indicate whether stop words
                                      (e.g., a, an, the) should be removed
                                      from the token sets of the overlap
                                      attribute values (defaults to False).

            q_val (int): The value of q to use if the overlap attributes values
                         are to be tokenized as qgrams (defaults to None).

            word_level (boolean): A flag to indicate whether the overlap
                                  attributes should be tokenized as words
                                  (i.e, using whitespace as delimiter)
                                  (defaults to True).

            overlap_size (int): The minimum number of tokens that must overlap
                                (defaults to 1).

            allow_missing (boolean): A flag to indicate whether tuple pairs
                                     with missing value in at least one of the
                                     blocking attributes should be included in
                                     the output candidate set (defaults to
                                     False). If this flag is set to True, a
                                     tuple pair with missing value in either
                                     blocking attribute will be retained in the
                                     output candidate set.

            verbose (boolean): A flag to indicate whether the debug information
                should be logged (defaults to False).

            show_progress (boolean): A flag to indicate whether progress should
                                     be displayed to the user (defaults to True).

            n_jobs (int): The number of parallel jobs to be used for computation
                (defaults to 1). If -1 all CPUs are used. If 0 or 1,
                no parallel computation is used at all, which is useful for
                debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
                used (where n_cpus are the total number of CPUs in the
                machine).Thus, for n_jobs = -2, all CPUs but one are used.
                If (n_cpus + 1 + n_jobs) is less than 1, then no parallel
                computation is used (i.e., equivalent to the default).

        Returns:
            A candidate set of tuple pairs that survived blocking (DataFrame).

        Raises:
            AssertionError: If `candset` is not of type pandas
                DataFrame.
            AssertionError: If `l_overlap_attr` is not of type string.
            AssertionError: If `r_overlap_attr` is not of type string.
            AssertionError: If `q_val` is not of type int.
            AssertionError: If `word_level` is not of type boolean.
            AssertionError: If `overlap_size` is not of type int.
            AssertionError: If `verbose` is not of type
                boolean.
            AssertionError: If `allow_missing` is not of type boolean.
            AssertionError: If `show_progress` is not of type
                boolean.
            AssertionError: If `n_jobs` is not of type
                int.
            AssertionError: If `l_overlap_attr` is not in the ltable
                columns.
            AssertionError: If `r_block_attr` is not in the rtable columns.
            SyntaxError: If `q_val` is set to a valid value and
                `word_level` is set to True.
            SyntaxError: If `q_val` is set to None and
                `word_level` is set to False.
        Examples:
            >>> import py_entitymatching as em
            >>> A = em.read_csv_metadata('path_to_csv_dir/table_A.csv', key='ID')
            >>> B = em.read_csv_metadata('path_to_csv_dir/table_B.csv', key='ID')
            >>> ob = em.OverlapBlocker()
            >>> C = ob.block_tables(A, B, 'address', 'address', l_output_attrs=['name'], r_output_attrs=['name'])

            >>> D1 = ob.block_candset(C, 'name', 'name', allow_missing=True)
            # Include all possible tuple pairs with missing values
            >>> D2 = ob.block_candset(C, 'name', 'name', allow_missing=True)
            # Execute blocking using multiple cores
            >>> D3 = ob.block_candset(C, 'name', 'name', n_jobs=-1)
            # Use q-gram tokenizer
            >>> D2 = ob.block_candset(C, 'name', 'name', word_level=False, q_val=2)


        """

        # validate data types of standard input parameters
        self.validate_types_params_candset(candset, verbose, show_progress, n_jobs)

        # validate data types of input parameters specific to overlap blocker
        self.validate_types_other_params(
            l_overlap_attr,
            r_overlap_attr,
            rem_stop_words,
            q_val,
            word_level,
            overlap_size,
        )

        # get and validate metadata
        log_info(
            logger,
            "Required metadata: cand.set key, fk ltable, fk rtable, "
            "ltable, rtable, ltable key, rtable key",
            verbose,
        )

        # # get metadata
        (
            key,
            fk_ltable,
            fk_rtable,
            ltable,
            rtable,
            l_key,
            r_key,
        ) = cm.get_metadata_for_candset(candset, logger, verbose)

        # # validate metadata
        cm._validate_metadata_for_candset(
            candset,
            key,
            fk_ltable,
            fk_rtable,
            ltable,
            rtable,
            l_key,
            r_key,
            logger,
            verbose,
        )

        # validate overlap attrs
        self.validate_overlap_attrs(ltable, rtable, l_overlap_attr, r_overlap_attr)

        # validate word_level and q_val
        self.validate_word_level_qval(word_level, q_val)

        # do blocking

        # # do projection before merge
        l_df = ltable[[l_key, l_overlap_attr]]
        r_df = rtable[[r_key, r_overlap_attr]]

        # # case the overlap attribute to string if required.
        l_df.is_copy, r_df.is_copy = False, False  # to avoid setwithcopy warning
        ssj.dataframe_column_to_str(l_df, l_overlap_attr, inplace=True)
        ssj.dataframe_column_to_str(r_df, r_overlap_attr, inplace=True)

        # # cleanup the tables from non-ascii characters, punctuations, and stop words
        self.cleanup_table(l_df, l_overlap_attr, rem_stop_words)
        self.cleanup_table(r_df, r_overlap_attr, rem_stop_words)

        # # determine which tokenizer to use
        if word_level == True:
            # # # create a whitespace tokenizer
            tokenizer = WhitespaceTokenizer(return_set=True)
        else:
            # # # create a qgram tokenizer
            tokenizer = QgramTokenizer(qval=q_val, return_set=True)

        # # create a filter for overlap similarity join
        overlap_filter = OverlapFilter(
            tokenizer, overlap_size, allow_missing=allow_missing
        )

        # # perform overlap similarity filtering of the candset
        out_table = overlap_filter.filter_candset(
            candset,
            fk_ltable,
            fk_rtable,
            l_df,
            r_df,
            l_key,
            r_key,
            l_overlap_attr,
            r_overlap_attr,
            n_jobs,
            show_progress=show_progress,
        )
        # update catalog
        cm.set_candset_properties(out_table, key, fk_ltable, fk_rtable, ltable, rtable)

        # return candidate set
        return out_table

    def block_tuples(
        self,
        ltuple,
        rtuple,
        l_overlap_attr,
        r_overlap_attr,
        rem_stop_words=False,
        q_val=None,
        word_level=True,
        overlap_size=1,
        allow_missing=False,
    ):
        """Blocks a tuple pair based on the overlap of token sets of attribute
           values.

        Args:
            ltuple (Series): The input left tuple.

            rtuple (Series): The input right tuple.

            l_overlap_attr (string): The overlap attribute in left tuple.

            r_overlap_attr (string): The overlap attribute in right tuple.

            rem_stop_words (boolean): A flag to indicate whether stop words
                                      (e.g., a, an, the) should be removed
                                      from the token sets of the overlap
                                      attribute values (defaults to False).

            q_val (int): A value of q to use if the overlap attributes values
                         are to be tokenized as qgrams (defaults to None).

            word_level (boolean): A flag to indicate whether the overlap
                                  attributes should be tokenized as words
                                  (i.e, using whitespace as delimiter)
                                  (defaults to True).

            overlap_size (int): The minimum number of tokens that must overlap
                                (defaults to 1).

            allow_missing (boolean): A flag to indicate whether a tuple pair
                                     with missing value in at least one of the
                                     blocking attributes should be blocked
                                     (defaults to False). If this flag is set
                                     to True, the pair will be kept if either
                                     ltuple has missing value in l_block_attr
                                     or rtuple has missing value in r_block_attr
                                     or both.

        Returns:
            A status indicating if the tuple pair is blocked (boolean).

        Examples:
            >>> import py_entitymatching as em
            >>> A = em.read_csv_metadata('path_to_csv_dir/table_A.csv', key='ID')
            >>> B = em.read_csv_metadata('path_to_csv_dir/table_B.csv', key='ID')
            >>> ob = em.OverlapBlocker()
            >>> status = ob.block_tuples(A.loc[0], B.loc[0], 'address', 'address')

        """

        # validate data types of input parameters specific to overlap blocker
        self.validate_types_other_params(
            l_overlap_attr,
            r_overlap_attr,
            rem_stop_words,
            q_val,
            word_level,
            overlap_size,
        )

        # validate word_level and q_val
        self.validate_word_level_qval(word_level, q_val)

        # determine which tokenizer to use
        if word_level == True:
            # # create a whitespace tokenizer
            tokenizer = WhitespaceTokenizer(return_set=True)
        else:
            # # create a qgram tokenizer
            tokenizer = QgramTokenizer(qval=q_val, return_set=True)

        # # cleanup the tuples from non-ascii characters, punctuations, and stop words
        l_val = self.cleanup_tuple_val(ltuple[l_overlap_attr], rem_stop_words)
        r_val = self.cleanup_tuple_val(rtuple[r_overlap_attr], rem_stop_words)

        # create a filter for overlap similarity
        overlap_filter = OverlapFilter(
            tokenizer, overlap_size, allow_missing=allow_missing
        )

        return overlap_filter.filter_pair(l_val, r_val)

    # helper functions

    # validate the data types of input parameters specific to overlap blocker
    def validate_types_other_params(
        self,
        l_overlap_attr,
        r_overlap_attr,
        rem_stop_words,
        q_val,
        word_level,
        overlap_size,
    ):
        validate_object_type(
            l_overlap_attr,
            six.string_types,
            error_prefix="Overlap attribute name of left table",
        )
        validate_object_type(
            r_overlap_attr,
            six.string_types,
            error_prefix="Overlap attribute name of right table",
        )

        validate_object_type(
            rem_stop_words, bool, error_prefix="Parameter rem_stop_words"
        )

        if q_val != None and not isinstance(q_val, int):
            logger.error("Parameter q_val is not of type int")
            raise AssertionError("Parameter q_val is not of type int")

        validate_object_type(word_level, bool, error_prefix="Parameter word_level")
        validate_object_type(overlap_size, int, error_prefix="Parameter overlap_size")

    # validate the overlap attrs
    def validate_overlap_attrs(self, ltable, rtable, l_overlap_attr, r_overlap_attr):
        if not isinstance(l_overlap_attr, list):
            l_overlap_attr = [l_overlap_attr]
        assert (
            set(l_overlap_attr).issubset(ltable.columns) is True
        ), "Left block attribute is not in the left table"

        if not isinstance(r_overlap_attr, list):
            r_overlap_attr = [r_overlap_attr]
        assert (
            set(r_overlap_attr).issubset(rtable.columns) is True
        ), "Right block attribute is not in the right table"

    # validate word_level and q_val
    def validate_word_level_qval(self, word_level, q_val):
        if word_level == True and q_val != None:
            raise SyntaxError(
                "Parameters word_level and q_val cannot be set together; Note that word_level is "
                "set to True by default, so explicity set word_level=false to use qgram with the "
                "specified q_val"
            )

        if word_level == False and q_val == None:
            raise SyntaxError(
                "Parameters word_level and q_val cannot be unset together; Note that q_val is "
                "set to None by default, so if you want to use qgram then "
                "explictiy set word_level=False and specify the q_val"
            )

    # cleanup a table from non-ascii characters, punctuations and stop words
    def cleanup_table(self, table, overlap_attr, rem_stop_words):
        # get overlap_attr column
        attr_col_values = table[overlap_attr]

        values = []
        for val in attr_col_values:
            if pd.isnull(val):
                values.append(val)
            else:
                processed_val = self.process_string(val, rem_stop_words)
                values.append(processed_val)

        table.is_copy = False
        table[overlap_attr] = values

    # cleanup a tuple from non-ascii characters, punctuations and stop words
    def cleanup_tuple_val(self, val, rem_stop_words):
        if pd.isnull(val):
            return val

        return self.process_string(val, rem_stop_words)

    def process_string(self, input_string, rem_stop_words):
        if not input_string:
            return input_string

        if isinstance(input_string, bytes):
            input_string = input_string.decode("utf-8", "ignore")
        input_string = input_string.lower()

        input_string = self.rem_punctuations(input_string)

        # remove stopwords
        # chop the attribute values and convert into a set
        val_chopped = list(set(input_string.strip().split()))

        # remove stop words
        if rem_stop_words:
            val_chopped_no_stopwords = self.rem_stopwords(val_chopped)
            val_joined = " ".join(val_chopped_no_stopwords)
        else:
            val_joined = " ".join(val_chopped)

        return val_joined

    def rem_punctuations(self, s):
        return self.regex_punctuation.sub("", s)

    def rem_stopwords(self, lst):
        return [t for t in lst if t not in self.stop_words]

    def overlap_join(
        self,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_join_attr,
        r_join_attr,
        tokenizer,
        threshold,
        comp_op=">=",
        allow_missing=False,
        l_out_attrs=None,
        r_out_attrs=None,
        l_out_prefix="l_",
        r_out_prefix="r_",
        out_sim_score=True,
        n_jobs=1,
        show_progress=True,
    ):
        """Join two tables using overlap measure.

        For two sets X and Y, the overlap between them is given by:

            :math:`overlap(X, Y) = |X \\cap Y|`

        Finds tuple pairs from left table and right table such that the overlap
        between the join attributes satisfies the condition on input threshold. For
        example, if the comparison operator is '>=', finds tuple pairs whose
        overlap between the strings that are the values of the join attributes is
        greater than or equal to the input threshold, as specified in "threshold".

        Args:
            ltable (DataFrame): left input table.

            rtable (DataFrame): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_join_attr (string): join attribute in left table.

            r_join_attr (string): join attribute in right table.

            tokenizer (Tokenizer): tokenizer to be used to tokenize join
                attributes.

            threshold (float): overlap threshold to be satisfied.

            comp_op (string): comparison operator. Supported values are '>=', '>'
                and '=' (defaults to '>=').

            allow_missing (boolean): flag to indicate whether tuple pairs with
                missing value in at least one of the join attributes should be
                included in the output (defaults to False). If this flag is set to
                True, a tuple in ltable with missing value in the join attribute
                will be matched with every tuple in rtable and vice versa.

            l_out_attrs (list): list of attribute names from the left table to be
                included in the output table (defaults to None).

            r_out_attrs (list): list of attribute names from the right table to be
                included in the output table (defaults to None).

            l_out_prefix (string): prefix to be used for the attribute names coming
                from the left table, in the output table (defaults to 'l\_').

            r_out_prefix (string): prefix to be used for the attribute names coming
                from the right table, in the output table (defaults to 'r\_').

            out_sim_score (boolean): flag to indicate whether similarity score
                should be included in the output table (defaults to True). Setting
                this flag to True will add a column named '_sim_score' in the
                output table. This column will contain the similarity scores for the
                tuple pairs in the output.

            n_jobs (int): number of parallel jobs to use for the computation
                (defaults to 1). If -1 is given, all CPUs are used. If 1 is given,
                no parallel computing code is used at all, which is useful for
                debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used
                (where n_cpus is the total number of CPUs in the machine). Thus for
                n_jobs = -2, all CPUs but one are used. If (n_cpus + 1 + n_jobs)
                becomes less than 1, then no parallel computing code will be used
                (i.e., equivalent to the default).

            show_progress (boolean): flag to indicate whether task progress should
                be displayed to the user (defaults to True).

        Returns:
            An output table containing tuple pairs that satisfy the join
            condition (DataFrame).
        """

        return self.overlap_join_py(
            ltable,
            rtable,
            l_key_attr,
            r_key_attr,
            l_join_attr,
            r_join_attr,
            tokenizer,
            threshold,
            comp_op,
            allow_missing,
            l_out_attrs,
            r_out_attrs,
            l_out_prefix,
            r_out_prefix,
            out_sim_score,
            n_jobs,
            show_progress,
        )

    def overlap_join_py(
        self,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_join_attr,
        r_join_attr,
        tokenizer,
        threshold,
        comp_op=">=",
        allow_missing=False,
        l_out_attrs=None,
        r_out_attrs=None,
        l_out_prefix="l_",
        r_out_prefix="r_",
        out_sim_score=True,
        n_jobs=1,
        show_progress=True,
    ):
        """Join two tables using overlap measure.

        For two sets X and Y, the overlap between them is given by:

            :math:`overlap(X, Y) = |X \\cap Y|`

        Finds tuple pairs from left table and right table such that the overlap
        between the join attributes satisfies the condition on input threshold. For
        example, if the comparison operator is '>=', finds tuple pairs whose
        overlap between the strings that are the values of the join attributes is
        greater than or equal to the input threshold, as specified in "threshold".

        Args:
            ltable (DataFrame): left input table.

            rtable (DataFrame): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_join_attr (string): join attribute in left table.

            r_join_attr (string): join attribute in right table.

            tokenizer (Tokenizer): tokenizer to be used to tokenize join
                attributes.

            threshold (float): overlap threshold to be satisfied.

            comp_op (string): comparison operator. Supported values are '>=', '>'
                and '=' (defaults to '>=').

            allow_missing (boolean): flag to indicate whether tuple pairs with
                missing value in at least one of the join attributes should be
                included in the output (defaults to False). If this flag is set to
                True, a tuple in ltable with missing value in the join attribute
                will be matched with every tuple in rtable and vice versa.

            l_out_attrs (list): list of attribute names from the left table to be
                included in the output table (defaults to None).

            r_out_attrs (list): list of attribute names from the right table to be
                included in the output table (defaults to None).

            l_out_prefix (string): prefix to be used for the attribute names coming
                from the left table, in the output table (defaults to 'l\_').

            r_out_prefix (string): prefix to be used for the attribute names coming
                from the right table, in the output table (defaults to 'r\_').

            out_sim_score (boolean): flag to indicate whether similarity score
                should be included in the output table (defaults to True). Setting
                this flag to True will add a column named '_sim_score' in the
                output table. This column will contain the similarity scores for the
                tuple pairs in the output.

            n_jobs (int): number of parallel jobs to use for the computation
                (defaults to 1). If -1 is given, all CPUs are used. If 1 is given,
                no parallel computing code is used at all, which is useful for
                debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used
                (where n_cpus is the total number of CPUs in the machine). Thus for
                n_jobs = -2, all CPUs but one are used. If (n_cpus + 1 + n_jobs)
                becomes less than 1, then no parallel computing code will be used
                (i.e., equivalent to the default).

            show_progress (boolean): flag to indicate whether task progress should
                be displayed to the user (defaults to True).

        Returns:
            An output table containing tuple pairs that satisfy the join
            condition (DataFrame).
        """

        # check if the input tokenizer is valid
        validate_tokenizer(tokenizer)

        # set return_set flag of tokenizer to be True, in case it is set to False
        revert_tokenizer_return_set_flag = False
        if not tokenizer.get_return_set():
            tokenizer.set_return_set(True)
            revert_tokenizer_return_set_flag = True

        # use overlap filter to perform the join.
        overlap_filter = OverlapFilter(tokenizer, threshold, comp_op, allow_missing)
        output_table = overlap_filter.filter_tables(
            ltable,
            rtable,
            l_key_attr,
            r_key_attr,
            l_join_attr,
            r_join_attr,
            l_out_attrs,
            r_out_attrs,
            l_out_prefix,
            r_out_prefix,
            out_sim_score,
            n_jobs,
            show_progress,
        )

        # revert the return_set flag of tokenizer, in case it was modified.
        if revert_tokenizer_return_set_flag:
            tokenizer.set_return_set(False)

        return output_table

    def generate_bag_of_tokens(
        self,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_join_attr,
        r_join_attr,
        tokenizer,
        threshold,
        comp_op=">=",
        allow_missing=False,
        l_out_attrs=None,
        r_out_attrs=None,
        l_out_prefix="l_",
        r_out_prefix="r_",
        out_sim_score=True,
        n_jobs=1,
        show_progress=True,
    ):
        return self.generate_bag_of_tokens_py(
            ltable,
            rtable,
            l_key_attr,
            r_key_attr,
            l_join_attr,
            r_join_attr,
            tokenizer,
            threshold,
            comp_op,
            allow_missing,
            l_out_attrs,
            r_out_attrs,
            l_out_prefix,
            r_out_prefix,
            out_sim_score,
            n_jobs,
            show_progress,
        )

    def generate_bag_of_tokens_py(
        self,
        ltable,
        rtable,
        l_key_attr,
        r_key_attr,
        l_join_attr,
        r_join_attr,
        tokenizer,
        threshold,
        comp_op=">=",
        allow_missing=False,
        l_out_attrs=None,
        r_out_attrs=None,
        l_out_prefix="l_",
        r_out_prefix="r_",
        out_sim_score=True,
        n_jobs=1,
        show_progress=True,
    ):
        # check if the input tokenizer is valid
        validate_tokenizer(tokenizer)

        # set return_set flag of tokenizer to be True, in case it is set to False
        revert_tokenizer_return_set_flag = False
        if not tokenizer.get_return_set():
            tokenizer.set_return_set(True)
            revert_tokenizer_return_set_flag = True

        # use overlap filter to perform the join.
        overlap_filter = OverlapFilter(tokenizer, threshold, comp_op, allow_missing)
        ltable_array, rtable_array = overlap_filter.preprocess(
            ltable,
            rtable,
            l_key_attr,
            r_key_attr,
            l_join_attr,
            r_join_attr,
            l_out_attrs,
            r_out_attrs,
            l_out_prefix,
            r_out_prefix,
            out_sim_score,
            n_jobs,
            show_progress,
        )

        return ltable_array, rtable_array


def read_csv_metadata(file_path, **kwargs):
    """
    Reads a CSV (comma-separated values) file into a pandas DataFrame
    and update the catalog with the metadata. The CSV files typically contain
    data for the input tables or a candidate set.

    Specifically, this function first reads the CSV file from the given file
    path into a pandas DataFrame, by using pandas' in-built 'read_csv'
    method. Then, it updates the catalog with the metadata. There are three
    ways to update the metadata: (1) using a metadata file, (2) using the
    key-value parameters supplied in the function, and (3) using both
    metadata file and key-value parameters.

    To update the metadata in the catalog using the metadata file,
    the function will look for a file in the same directory with  same file name
    but with a  specific extension. This extension can be optionally given by
    the user (defaults to '.metadata'). If the metadata  file is  present,
    the function will read and update the catalog appropriately. If  the
    metadata file is not present, the function will issue a warning that the
    metadata file is not present.

    The metadata information can also be given as parameters to the function
    (see description of arguments for more details). If given, the function
    will update the catalog with the given information.

    Further, the metadata can partly reside in the metdata file and partly as
    supplied parameters. The function will take a union of the two and
    update the catalog appropriately.
    If the same metadata is given in both the metadata file
    and the function, then the metadata in the function takes precedence over
    the metadata given in the file.

    Args:
        file_path(string): The CSV file path

        kwargs(dictionary): A Python dictionary containing key-value arguments.
            There are a few key-value pairs that are specific to
            read_csv_metadata and  all the other key-value pairs are passed
            to pandas read_csv method

    Returns:
        A pandas DataFrame read from the input CSV file.
    Raises:
        AssertionError: If `file_path` is not of type string.
        AssertionError: If a file does not exist in the
            given `file_path`.

    Examples:
        *Example 1:* Read from CSV file and set metadata

        >>> A = em.read_csv_metadata('path_to_csv_file', key='id')
        >>> em.get_key(A)
         # 'id'

        *Example 2:*  Read from CSV file (with metadata file in the same directory

         Let the metadata file contain the following contents:

          #key = id

        >>> A = em.read_csv_metadata('path_to_csv_file')
        >>> em.get_key(A)
         # 'id'

    See Also:
        :meth:`~py_entitymatching.to_csv_metadata`
    """
    # Validate the input parameters.

    validate_object_type(file_path, six.string_types, error_prefix="Input file path")

    # # Check if the given path is valid.
    if not os.path.exists(file_path):
        logger.error("File does not exist at path %s" % file_path)
        raise AssertionError("File does not exist at path %s" % file_path)

    # Check if the user has specified the metadata file's extension.
    extension = kwargs.pop("metadata_extn", None)

    # If the extension is not specified then set the extension to .metadata'.
    if extension is None:
        extension = ".metadata"

    # Format the extension to include a '.' in front if the user has not
    # given one.
    if not extension.startswith("."):
        extension = "." + extension

    # If the file is present, then update metadata from file.
    if _is_metadata_file_present(file_path, extension=extension):
        file_name, _ = os.path.splitext(file_path)
        file_name = "".join([file_name, extension])
        metadata, _ = _get_metadata_from_file(file_name)

    # Else issue a warning that the metadata file is not present
    else:
        logger.warning(
            "Metadata file is not present in the given path; "
            "proceeding to read the csv file."
        )
        metadata = {}

    # Update the metadata with the key-value pairs given in the command. The
    # function _update_metadata_for_read_cmd takes care of updating the
    # metadata with only the key-value pairs specific to read_csv_metadata
    # method
    metadata, kwargs = _update_metadata_for_read_cmd(metadata, **kwargs)

    # Validate the metadata.
    _check_metadata_for_read_cmd(metadata)

    # Read the csv file using pandas read_csv method.
    data_frame = pd.read_csv(file_path, **kwargs)

    # Get the value for 'key' property and update the catalog.
    key = metadata.pop("key", None)
    if key is not None:
        cm.set_key(data_frame, key)

    fk_ltable = metadata.pop("fk_ltable", None)
    if fk_ltable is not None:
        cm.set_fk_ltable(data_frame, fk_ltable)

    fk_rtable = metadata.pop("fk_rtable", None)
    if fk_ltable is not None:
        cm.set_fk_rtable(data_frame, fk_rtable)

    # Update the catalog with other properties.
    for property_name, property_value in six.iteritems(metadata):
        cm.set_property(data_frame, property_name, property_value)
    if not cm.is_dfinfo_present(data_frame):
        cm.init_properties(data_frame)

    # Return the DataFrame
    return data_frame


def _is_metadata_file_present(file_path, extension=".metadata"):
    """
    Check if the metadata file is present.
    """
    # Get the file name and the extension from the file path.
    file_name, _ = os.path.splitext(file_path)
    # Create a file name with the given extension.
    file_name = "".join([file_name, extension])
    # Check if the file already exists.
    return os.path.exists(file_name)


def _get_metadata_from_file(file_path):
    """
    Get the metadata information from the file.
    """
    # Initialize a dictionary to store the metadata read from the file.
    metadata = dict()

    # Get the number of lines from the file
    num_lines = sum(1 for _ in open(file_path))

    # If there are some contents in the file (i.e num_lines > 0),
    # read its contents.
    if num_lines > 0:
        with open(file_path) as file_handler:
            for _ in range(num_lines):
                line = next(file_handler)
                # Consider only the lines that are starting with '#'
                if line.startswith("#"):
                    # Remove the leading '#'
                    line = line.lstrip("#")
                    # Split the line with '=' as the delimiter
                    tokens = line.split("=")
                    # Based on the special syntax we use, there should be
                    # exactly two tokens after we split using '='
                    assert len(tokens) is 2, "Error in file, he num tokens " "is not 2"
                    # Retrieve the property_names and values.
                    property_name = tokens[0].strip()
                    property_value = tokens[1].strip()
                    # If the property value is not 'POINTER' then store it in
                    #  the metadata dictionary.
                    if property_value is not "POINTER":
                        metadata[property_name] = property_value

    # Return the metadata dictionary and the number of lines in the file.
    return metadata, num_lines


def _update_metadata_for_read_cmd(metadata, **kwargs):
    """
    Update metadata for read_csv_metadata method.
    """
    # Create a copy of incoming metadata. We will update the incoming
    # metadata dict with kwargs.
    copy_metadata = metadata.copy()

    # The updation is going to happen in two steps: (1) overriding the
    # properties in metadata dict using kwargs, and (2) adding the properties
    #  to metadata dict from kwargs.

    # Step 1
    # We will override the properties in the metadata dict with the
    # properties from kwargs.

    # Get the property from metadata dict.
    for property_name in copy_metadata.keys():
        # If the same property is present in kwargs, then override it in the
        # metadata dict.
        if property_name in kwargs:
            property_value = kwargs.pop(property_name)
            if property_value is not None:
                metadata[property_name] = property_value
            else:
                # Warn the users if the metadata dict had a valid value,
                # but the kwargs sets it to None.
                logger.warning(
                    "%s key had a value (%s)in file but input arg is set to "
                    "None" % (property_name, metadata[property_name])
                )
                # Remove the property from the dictionary.
                metadata.pop(property_name)  # remove the key-value pair

    # Step 2
    # Add the properties from kwargs.
    # We should be careful here. The kwargs contains the key-value pairs that
    # are used by read_csv method (of pandas). We will just pick the
    # properties that we expect from the read_csv_metadata method.
    properties = ["key", "ltable", "rtable", "fk_ltable", "fk_rtable"]

    # For the properties that we expect, read from kwargs and update the
    # metadata dict.
    for property_name in properties:
        if property_name in kwargs:
            property_value = kwargs.pop(property_name)
            if property_value is not None:
                metadata[property_name] = property_value
            else:
                # Warn the users if the properties in the kwargs is set to None.
                logger.warning("Metadata %s is set to None", property_name)
                # Remove the property from the metadata dict.
                metadata.pop(property_name, None)

    return metadata, kwargs


def _check_metadata_for_read_cmd(metadata):
    """
    Check the metadata for read_csv_metadata command
    """

    # Do basic validation checks for the metadata.

    # We require consistency of properties given for the canidate set. We
    # expect the user to provide all the required properties for the
    # candidate set.
    required_properties = ["ltable", "rtable", "fk_ltable", "fk_rtable"]

    # Check what the user has given
    given_properties = set(required_properties).intersection(metadata.keys())

    # Check if all the required properties are given
    if len(given_properties) > 0:
        # Check the lengths are same. If not, this means that the user is
        # missing something. So, raise an error.
        if len(given_properties) is not len(required_properties):
            logger.error(
                "Dataframe requires all valid ltable, rtable, fk_ltable, "
                "fk_rtable parameters set"
            )
            raise AssertionError(
                "Dataframe requires all valid ltable, rtable, fk_ltable, "
                "fk_rtable parameters set"
            )

        # ltable is expected to be of type pandas DataFrame. If not raise an
        # error.
        if not isinstance(metadata["ltable"], pd.DataFrame):
            logger.error("The parameter ltable must be set to valid Dataframe")
            raise AssertionError("The parameter ltable must be set to valid Dataframe")

        # rtable is expected to be of type pandas DataFrame. If not raise an
        # error.
        if not isinstance(metadata["rtable"], pd.DataFrame):
            logger.error("The parameter rtable must be set to valid Dataframe")
            raise AssertionError("The parameter rtable must be set to valid Dataframe")
    # If the length of comman properties is 0, it will fall out to return
    # True, which is ok.
    return True


def find_candidates(probe_tokens, inverted_index):
    candidate_overlap = {}

    if not inverted_index.index:
        return candidate_overlap

    for token in probe_tokens:
        for cand in inverted_index.probe(token):
            candidate_overlap[cand] = candidate_overlap.get(cand, 0) + 1
    return candidate_overlap


def get_attrs_to_project(out_attrs, key_attr, join_attr):
    # this method assumes key_attr has already been removed from
    # out_attrs, if present.
    proj_attrs = [key_attr, join_attr]

    if out_attrs is not None:
        for attr in out_attrs:
            if attr != join_attr:
                proj_attrs.append(attr)

    return proj_attrs
