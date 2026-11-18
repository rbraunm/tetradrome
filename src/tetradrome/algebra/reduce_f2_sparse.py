# SPDX-License-Identifier: Apache-2.0

"""Sparse F2 column reduction on sorted index arrays, compiled by Numba.

The same reduction as ``reduce_reference`` -- each column is eliminated against the pivots found
so far, keyed by leading row -- with a column held as a sorted array of its row indices and an
elimination done as a merge of two sorted arrays. It is for matrices too large and too sparse for
the packed tiers, whose columns would each be a dense bit-vector over every row: the filtered
complex tau is read from does not split by Alexander grading, so its two maps span every
generator of their Maslov degrees.

Both functions return exactly what their reference counterparts return
(``reduce_reference.f2_rank`` and ``reduce_reference.f2_kernel_escape``); the tests pin that.
They need numpy and Numba; without them the reference is the path.
"""
from __future__ import annotations

from ..errors import BackendUnavailable

try:
    import numpy as np
    import numba
    HAVE_SPARSE = True
except ImportError:
    HAVE_SPARSE = False

_kernels = None


def _unique_sorted(values, out):
    # A column as the reference reads it: the set of its row indices, here ascending in ``out``.
    ordered = np.sort(values)
    count = 0
    for k in range(ordered.shape[0]):
        if count == 0 or ordered[k] != out[count - 1]:
            out[count] = ordered[k]
            count += 1
    return count


def _xor_sorted(left, left_count, right, right_start, right_count, out):
    # Symmetric difference of two ascending arrays, ascending in ``out``.
    i = 0
    j = right_start
    stop = right_start + right_count
    count = 0
    while i < left_count and j < stop:
        if left[i] < right[j]:
            out[count] = left[i]
            i += 1
            count += 1
        elif left[i] > right[j]:
            out[count] = right[j]
            j += 1
            count += 1
        else:
            i += 1
            j += 1
    while i < left_count:
        out[count] = left[i]
        i += 1
        count += 1
    while j < stop:
        out[count] = right[j]
        j += 1
        count += 1
    return count


def _grown(array, needed):
    if array.shape[0] >= needed:
        return array
    larger = np.empty(max(needed, 2 * array.shape[0]), array.dtype)
    larger[:array.shape[0]] = array
    return larger


def _pivots(indices, indptr, rows):
    """Reduce every column in order. The pivots, by leading row, as ``(start, length, arena)``:
    the reduced column with leading row r is ``arena[start[r]:start[r] + length[r]]``, and
    ``start[r]`` is -1 where no column leads with r."""
    start = np.full(rows, -1, np.int64)
    length = np.zeros(rows, np.int64)
    arena = np.empty(max(2 * indices.shape[0], 16), np.int64)
    used = 0
    column = np.empty(16, np.int64)
    spare = np.empty(16, np.int64)
    for j in range(indptr.shape[0] - 1):
        raw = indices[indptr[j]:indptr[j + 1]]
        column = _grown(column, raw.shape[0])
        count = _unique_sorted(raw, column)
        while count > 0:
            lead = column[count - 1]
            if start[lead] < 0:
                arena = _grown(arena, used + count)
                arena[used:used + count] = column[:count]
                start[lead] = used
                length[lead] = count
                used += count
                break
            spare = _grown(spare, count + length[lead])
            count = _xor_sorted(column, count, arena, start[lead], length[lead], spare)
            column, spare = spare, column
    return start, length, arena


def _rank(indices, indptr, rows):
    return int((_pivots(indices, indptr, rows)[0] >= 0).sum())


def _kernel_escape(indices, indptr, rows, levels, image_indices, image_indptr):
    """``(column, image_rank)``: the first column, in ascending level with ties in column order,
    that reduces to zero by a combination of columns the image does not span, or -1 when the
    image spans every such combination."""
    image_start, image_length, image_arena = _pivots(image_indices, image_indptr,
                                                     indptr.shape[0] - 1)
    image_rank = int((image_start >= 0).sum())
    start = np.full(rows, -1, np.int64)
    length = np.zeros(rows, np.int64)
    arena = np.empty(max(2 * indices.shape[0], 16), np.int64)
    used = 0
    held_start = np.zeros(rows, np.int64)           # the combination kept beside each pivot
    held_length = np.zeros(rows, np.int64)
    held = np.empty(16, np.int64)
    held_used = 0
    column = np.empty(16, np.int64)
    spare = np.empty(16, np.int64)
    combo = np.empty(16, np.int64)
    combo_spare = np.empty(16, np.int64)
    order = np.argsort(levels, kind="mergesort")
    for which in range(order.shape[0]):
        j = order[which]
        raw = indices[indptr[j]:indptr[j + 1]]
        column = _grown(column, raw.shape[0])
        count = _unique_sorted(raw, column)
        combo[0] = j
        combo_count = 1
        while count > 0:
            lead = column[count - 1]
            if start[lead] < 0:
                arena = _grown(arena, used + count)
                arena[used:used + count] = column[:count]
                start[lead] = used
                length[lead] = count
                used += count
                held = _grown(held, held_used + combo_count)
                held[held_used:held_used + combo_count] = combo[:combo_count]
                held_start[lead] = held_used
                held_length[lead] = combo_count
                held_used += combo_count
                break
            spare = _grown(spare, count + length[lead])
            count = _xor_sorted(column, count, arena, start[lead], length[lead], spare)
            column, spare = spare, column
            combo_spare = _grown(combo_spare, combo_count + held_length[lead])
            combo_count = _xor_sorted(combo, combo_count, held, held_start[lead],
                                      held_length[lead], combo_spare)
            combo, combo_spare = combo_spare, combo
        if count > 0:
            continue
        while combo_count > 0:                      # a relation: does the image span it?
            lead = combo[combo_count - 1]
            if image_start[lead] < 0:
                return j, image_rank
            combo_spare = _grown(combo_spare, combo_count + image_length[lead])
            combo_count = _xor_sorted(combo, combo_count, image_arena, image_start[lead],
                                      image_length[lead], combo_spare)
            combo, combo_spare = combo_spare, combo
    return -1, image_rank


def _bound():
    """The compiled rank and kernel-escape kernels, bound once per process."""
    global _kernels
    if _kernels is None:
        if not HAVE_SPARSE:
            raise BackendUnavailable(
                "the sparse reducer needs numpy and numba (pip install the 'jit' extra); the "
                "pure-Python reference reducer needs neither.")
        # Each kernel calls the ones before it, so they are compiled in that order and each
        # replaces its plain function before the next is compiled against it.
        for function in (_unique_sorted, _xor_sorted, _grown, _pivots, _rank, _kernel_escape):
            globals()[function.__name__] = numba.njit(cache=True)(function)
        _kernels = (_rank, _kernel_escape)
    return _kernels


def _as_arrays(csc):
    indices, indptr = csc
    return np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)


def _row_count(indices) -> int:
    return int(indices.max()) + 1 if indices.shape[0] else 0


def f2_rank_sparse(csc) -> int:
    """Rank over F2 of a 0/1 matrix in CSC form ``(indices, indptr)``:
    ``reduce_reference.f2_rank``'s answer."""
    rank, _ = _bound()
    indices, indptr = _as_arrays(csc)
    return int(rank(indices, indptr, _row_count(indices)))


def f2_kernel_escape_sparse(csc, levels, image_csc):
    """``reduce_reference.f2_kernel_escape``'s answer for the matrix ``csc``, its columns'
    filtration ``levels``, and ``image_csc`` over its column indices: the smallest level whose
    kernel is not inside the image's span (None when all of it is), and the image's rank."""
    _, kernel_escape = _bound()
    indices, indptr = _as_arrays(csc)
    image_indices, image_indptr = _as_arrays(image_csc)
    levels = np.asarray(levels, dtype=np.int64)
    column, image_rank = kernel_escape(indices, indptr, _row_count(indices), levels,
                                       image_indices, image_indptr)
    return (None if column < 0 else int(levels[column])), int(image_rank)
