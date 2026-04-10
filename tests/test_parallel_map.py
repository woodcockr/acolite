"""Unit tests for the parallel_map utility in acolite.shared."""
import pytest
from acolite.shared.parallel_map import parallel_map


def square(x):
    return x * x


def identity(args):
    return args


def add_tuple(args):
    a, b = args
    return a + b


class TestParallelMapThreading:
    """Tests for the 'threading' scheduler."""

    def test_basic_mapping(self):
        args = [1, 2, 3, 4, 5]
        results = parallel_map(square, args, scheduler='threading')
        assert results == [1, 4, 9, 16, 25]

    def test_tuple_args(self):
        args = [(1, 2), (3, 4), (5, 6)]
        results = parallel_map(add_tuple, args, scheduler='threading')
        assert results == [3, 7, 11]

    def test_empty_args(self):
        results = parallel_map(square, [], scheduler='threading')
        assert results == []

    def test_single_element(self):
        results = parallel_map(square, [7], scheduler='threading')
        assert results == [49]

    def test_max_workers(self):
        args = list(range(10))
        results = parallel_map(square, args, scheduler='threading', max_workers=2)
        assert results == [x * x for x in args]

    def test_order_preserved(self):
        """Results must be in the same order as the input args."""
        args = list(range(20))
        results = parallel_map(square, args, scheduler='threading')
        expected = [x * x for x in args]
        assert results == expected


class TestParallelMapDask:
    """Tests for the 'dask' scheduler."""

    def test_basic_mapping(self):
        args = [1, 2, 3, 4, 5]
        results = parallel_map(square, args, scheduler='dask')
        assert results == [1, 4, 9, 16, 25]

    def test_tuple_args(self):
        args = [(1, 2), (3, 4), (5, 6)]
        results = parallel_map(add_tuple, args, scheduler='dask')
        assert results == [3, 7, 11]

    def test_empty_args(self):
        results = parallel_map(square, [], scheduler='dask')
        assert results == []

    def test_single_element(self):
        results = parallel_map(square, [7], scheduler='dask')
        assert results == [49]

    def test_max_workers(self):
        args = list(range(10))
        results = parallel_map(square, args, scheduler='dask', max_workers=2)
        assert results == [x * x for x in args]

    def test_order_preserved(self):
        """Results must be in the same order as the input args."""
        args = list(range(20))
        results = parallel_map(square, args, scheduler='dask')
        expected = [x * x for x in args]
        assert results == expected


class TestParallelMapDefault:
    """Tests for the default scheduler (should be 'threading')."""

    def test_default_scheduler(self):
        args = [1, 2, 3]
        results = parallel_map(square, args)
        assert results == [1, 4, 9]

    def test_unknown_scheduler_falls_back_to_threading(self):
        """Unknown scheduler names should fall back to the threading backend."""
        args = [2, 3]
        results = parallel_map(square, args, scheduler='unknown_scheduler')
        assert results == [4, 9]
