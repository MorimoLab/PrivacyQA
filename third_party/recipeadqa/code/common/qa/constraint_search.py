"""Exact RecipeADKG answer-index loading and ingredient constraint search.

Ingredient names are used exactly as stored in the joined RecipeADKG recipe data.
No lexical normalization, stemming, singularization, or synonym merging is
performed.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from common.io import iter_json_array


def load_index(
    index_path: Path,
) -> tuple[
    sqlite3.Connection,
    list[dict[str, Any]],
    dict[str, int],
    np.ndarray,
]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT answer_id, recipe_uri, title, tag_count, generates_constraints "
        "FROM answers ORDER BY answer_id"
    ).fetchall()
    answers = [
        {
            "answer_id": answer_id,
            "recipe_uri": uri,
            "title": title,
            "tag_count": tag_count,
            "generates_constraints": bool(generates),
        }
        for answer_id, uri, title, tag_count, generates in rows
    ]
    uri_to_answer = {row["recipe_uri"]: row["answer_id"] for row in answers}
    generator_mask = np.zeros((len(answers) + 7) // 8, dtype=np.uint8)
    for row in answers:
        if row["generates_constraints"]:
            answer_id = row["answer_id"]
            generator_mask[answer_id >> 3] |= np.uint8(1 << (answer_id & 7))
    return connection, answers, uri_to_answer, generator_mask


def load_source_to_answer(
    index_path: Path,
    uri_to_answer: dict[str, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            answer_id = uri_to_answer.get(row["recipe_uri"])
            if answer_id is not None:
                result[row["recipe_id"]] = answer_id
    return result


def load_ingredients(
    recipes_path: Path,
    source_to_answer: dict[str, int],
    answer_count: int,
) -> tuple[list[tuple[int, ...]], list[str], int]:
    answer_sets: list[set[int] | None] = [None] * answer_count
    ingredient_to_id: dict[str, int] = {}
    ingredient_names: list[str] = []
    source_rows = 0
    for recipe in iter_json_array(recipes_path):
        answer_id = source_to_answer.get(recipe.get("id"))
        if answer_id is None:
            continue
        source_rows += 1
        values = answer_sets[answer_id]
        if values is None:
            values = set()
            answer_sets[answer_id] = values
        for item in recipe.get("ingredients", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            ingredient_id = ingredient_to_id.get(name)
            if ingredient_id is None:
                ingredient_id = len(ingredient_names)
                ingredient_to_id[name] = ingredient_id
                ingredient_names.append(name)
            values.add(ingredient_id)
    return (
        [tuple(sorted(values)) if values else tuple() for values in answer_sets],
        ingredient_names,
        source_rows,
    )


def load_tags(
    connection: sqlite3.Connection,
    answer_count: int,
) -> tuple[list[str], np.ndarray]:
    tags = [
        row[0] for row in connection.execute("SELECT tag_id FROM tags ORDER BY tag_id")
    ]
    positions = {tag: position for position, tag in enumerate(tags)}
    bitmaps = np.zeros((len(tags), (answer_count + 7) // 8), dtype=np.uint8)
    for answer_id, tag in connection.execute(
        "SELECT answer_id, tag_id FROM answer_tags ORDER BY tag_id, answer_id"
    ):
        bitmaps[positions[tag], answer_id >> 3] |= np.uint8(
            1 << (answer_id & 7)
        )
    return tags, bitmaps


def build_local_postings(
    answer_ids: Sequence[int],
    answer_ingredients: list[tuple[int, ...]],
) -> dict[int, int]:
    postings: dict[int, int] = defaultdict(int)
    for local_position, answer_id in enumerate(answer_ids):
        bit = 1 << local_position
        for ingredient_id in answer_ingredients[int(answer_id)]:
            postings[ingredient_id] |= bit
    return postings


def ids_from_int_bitmap(
    bitmap: int, universe_size: int | None = None
) -> list[int]:
    """Return ascending set-bit positions from a Python integer bitmap."""
    if universe_size is not None:
        byte_count = (universe_size + 7) // 8
        packed = np.frombuffer(bitmap.to_bytes(byte_count, "little"), dtype=np.uint8)
        return np.flatnonzero(
            np.unpackbits(packed, bitorder="little", count=universe_size)
        ).tolist()
    result: list[int] = []
    while bitmap:
        lowest = bitmap & -bitmap
        result.append(lowest.bit_length() - 1)
        bitmap ^= lowest
    return result


def find_split(
    initial_count: int,
    postings: dict[int, int],
    ingredient_names: list[str],
    lower: int,
    upper: int,
    target: int,
    max_constraints: int | None = None,
) -> tuple[str, int, list[dict[str, Any]]]:
    if initial_count < lower:
        return "below_target_range", (1 << initial_count) - 1, []
    if initial_count <= upper:
        return "already_in_target_range", (1 << initial_count) - 1, []

    state = (1 << initial_count) - 1
    current_count = initial_count
    used: set[int] = set()
    constraints: list[dict[str, Any]] = []

    while current_count > upper:
        if (
            max_constraints is not None
            and len(constraints) >= max_constraints
        ):
            return "ingredient_budget_exhausted", state, constraints
        best: tuple[tuple[Any, ...], int, str, int, int] | None = None
        for ingredient_id, ingredient_mask in postings.items():
            if ingredient_id in used:
                continue
            include_mask = state & ingredient_mask
            include_count = include_mask.bit_count()
            exclude_count = current_count - include_count

            branches = (
                ("include", include_mask, include_count, exclude_count),
                ("exclude", state ^ include_mask, exclude_count, include_count),
            )
            for operation, next_mask, next_count, removed_count in branches:
                if next_count < lower or next_count >= current_count:
                    continue
                # A selected ingredient must induce a non-trivial split. This
                # excludes one-off removals while still allowing an ingredient
                # occurring in exactly the desired 15-25 answer subset.
                if removed_count < 2:
                    continue
                interval_distance = (
                    0 if next_count <= upper else next_count - upper
                )
                score = (
                    interval_distance,
                    abs(next_count - target),
                    -removed_count,
                    ingredient_names[ingredient_id],
                    operation,
                )
                candidate = (
                    score,
                    ingredient_id,
                    operation,
                    next_mask,
                    next_count,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None:
            return "unsplittable_by_greedy_search", state, constraints

        _, ingredient_id, operation, next_state, next_count = best
        constraints.append(
            {
                "operation": operation,
                "ingredient": ingredient_names[ingredient_id],
                "before_answer_count": current_count,
                "after_answer_count": next_count,
                "removed_answer_count": current_count - next_count,
                "ingredient_support_before": (
                    state & postings[ingredient_id]
                ).bit_count(),
            }
        )
        used.add(ingredient_id)
        state = next_state
        current_count = next_count

    return "split_to_target_range", state, constraints


def find_representative_split(
    initial_count: int,
    postings: dict[int, int],
    ingredient_names: list[str],
    lower: int,
    upper: int,
    target: int,
    max_constraints: int,
    min_initial_coverage: float = 0.001,
    candidate_limit: int = 256,
    beam_width: int = 24,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Beam-search using ingredients frequent in the original answer set."""
    if initial_count < lower:
        return "below_target_range", (1 << initial_count) - 1, []
    if initial_count <= upper:
        return "already_in_target_range", (1 << initial_count) - 1, []
    if max_constraints <= 0:
        return "ingredient_budget_exhausted", (1 << initial_count) - 1, []

    initial_state = (1 << initial_count) - 1
    min_initial_support = max(
        lower, math.ceil(initial_count * min_initial_coverage)
    )
    candidates = []
    for ingredient_id, ingredient_mask in postings.items():
        support = ingredient_mask.bit_count()
        if min_initial_support <= support <= initial_count - lower:
            candidates.append((ingredient_id, ingredient_mask, support))
    candidates.sort(
        key=lambda item: (-item[2], ingredient_names[item[0]])
    )
    candidates = candidates[:candidate_limit]
    if not candidates:
        return "no_representative_ingredients", initial_state, []

    # state mask, path, used ingredients, cumulative initial coverage.
    beam: list[tuple[int, list[dict[str, Any]], frozenset[int], float]] = [
        (initial_state, [], frozenset(), 0.0)
    ]
    best_partial = beam[0]

    for _depth in range(1, max_constraints + 1):
        next_states: dict[
            int, tuple[int, list[dict[str, Any]], frozenset[int], float]
        ] = {}
        completed = []
        for state, path, used, support_score in beam:
            current_count = state.bit_count()
            for ingredient_id, ingredient_mask, initial_support in candidates:
                if ingredient_id in used:
                    continue
                include_mask = state & ingredient_mask
                include_count = include_mask.bit_count()
                exclude_count = current_count - include_count
                for operation, next_mask, next_count, removed_count in (
                    ("include", include_mask, include_count, exclude_count),
                    ("exclude", state ^ include_mask, exclude_count, include_count),
                ):
                    # Both branches must contain at least lower recipes. This
                    # prevents rare or one-off ingredients from driving a split.
                    if (
                        next_count < lower
                        or removed_count < lower
                        or next_count >= current_count
                    ):
                        continue
                    constraint = {
                        "operation": operation,
                        "ingredient": ingredient_names[ingredient_id],
                        "before_answer_count": current_count,
                        "after_answer_count": next_count,
                        "removed_answer_count": removed_count,
                        "ingredient_support_before": include_count,
                        "ingredient_support_initial": initial_support,
                        "ingredient_coverage_initial": (
                            initial_support / initial_count
                        ),
                    }
                    item = (
                        next_mask,
                        path + [constraint],
                        used | {ingredient_id},
                        support_score + initial_support / initial_count,
                    )
                    if next_count <= upper:
                        completed.append(item)
                    else:
                        previous = next_states.get(next_mask)
                        if previous is None or item[3] > previous[3]:
                            next_states[next_mask] = item

        if completed:
            completed.sort(
                key=lambda item: (
                    abs(item[0].bit_count() - target),
                    -item[3],
                    tuple(
                        (value["ingredient"], value["operation"])
                        for value in item[1]
                    ),
                )
            )
            state, path, _, _ = completed[0]
            return "split_to_target_range", state, path
        if not next_states:
            return (
                "unsplittable_by_representative_search",
                best_partial[0],
                best_partial[1],
            )

        beam = sorted(
            next_states.values(),
            key=lambda item: (
                abs(math.log(item[0].bit_count() / target)),
                -item[3],
            ),
        )[:beam_width]
        if beam[0][0].bit_count() < best_partial[0].bit_count():
            best_partial = beam[0]

    return "ingredient_budget_exhausted", best_partial[0], best_partial[1]
