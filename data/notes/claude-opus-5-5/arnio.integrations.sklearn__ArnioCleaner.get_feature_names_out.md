# Notes: `ArnioCleaner.get_feature_names_out(self, input_features=None)`

## 1. Purpose
- Returns the fitted output column names, for sklearn `ColumnTransformer` compatibility.
- When `input_features` is given, it checks that argument against the columns seen at fit time.
- It never computes names from `input_features`. It always returns the stored `self.feature_names_out_`.

## 2. Inputs
- `input_features` accepts:
  - `None` (the default), or
  - a non-string sized iterable of `str`: a list, tuple, `pd.Index` of strings, or numpy str array. `np.str_` is a subclass of `str`, so it passes.
- Validation runs only when `input_features is not None`, in this order:
  1. Is it `str` or `bytes`?
  2. Does it have both `__len__` and `__iter__`?
  3. Is every element an instance of `str`?
  4. Is `len == self.n_features_in_`?
  5. Is `list(input_features) == list(self.feature_names_in_)`? This is exact and order-sensitive.
- Fitted state it relies on:
  - `feature_names_in_` is `np.array(X.columns, dtype=object)`.
  - `n_features_in_` is `X.shape[1]`.
  - Both are set by `fit`.

## 3. Raises (in check order)
1. **Not fitted.** `check_is_fitted(self, "feature_names_out_")` raises `sklearn.exceptions.NotFittedError`, which subclasses both `ValueError` and `AttributeError`.
   - The message text comes from sklearn: "This ArnioCleaner instance is not fitted yet. Call 'fit' ...". Match only the substring `"not fitted"`.
   - Trigger: `ar_sk.ArnioCleaner().get_feature_names_out()`. It fires even when `input_features=None`.
2. **String or bytes.** `TypeError("input_features must be a sequence of strings, not str")`.
   - Triggers: `est.get_feature_names_out("a")` and `est.get_feature_names_out(b"a")`.
3. **No `__len__` or no `__iter__`.** `TypeError("input_features must be a sequence of strings")`.
   - Triggers: `est.get_feature_names_out(5)`, or a generator `(c for c in ["a","b"])`, which has `__iter__` but no `__len__`.
4. **Non-string element.** `TypeError("All input_features must be strings")`.
   - Trigger: `est.get_feature_names_out([1, 2])`.
   - This check comes before the length check, so `[1]` on a 2-column fit gives this TypeError, not a ValueError.
5. **Wrong length.** `ValueError(f"input_features should have length {self.n_features_in_}")`.
   - Trigger: fit on columns `a,b`, then `est.get_feature_names_out(["a"])` gives `"input_features should have length 2"`.
6. **Wrong names or order.** `ValueError("input_features must match columns seen during fit.")`.
   - Triggers: `["b","a"]` or `["a","c"]` on a fit with `a,b`.
- Use `re.escape` in `match=`, because messages contain `.`.

## 4. Returns and guarantees
- Returns the attribute object itself: `est.get_feature_names_out() is est.feature_names_out_` is True.
  - This holds both with `None` and with valid `input_features`.
  - It is not a copy, and it is never the passed `input_features`.
- The result is a `numpy.ndarray` with `dtype == object`, holding Python `str` column names.
- The value depends on what was run last:
  - **After `fit`** with `allow_schema_changes=False`, or with an empty `steps`: a copy of `feature_names_in_`, i.e. the input column order. It is a different object from `feature_names_in_` but equal in content.
  - **After `fit`** with `allow_schema_changes=True` and non-empty `steps`: the columns of the pipeline run on a probe copy of X.
  - **After `transform`**: overwritten with `np.array(X_out.columns, dtype=object)`, the columns of the transform output.

## 5. Side effects
- None. It reads attributes only and does not mutate the estimator or `input_features`.

## 6. Branches and edge cases
- **`input_features=None`**: returns early with no validation, e.g. `['a','b']` after `fit(pd.DataFrame({"a":[1],"b":[2]}))`.
- **Valid `input_features`** (`["a","b"]`, `("a","b")`, `pd.Index(["a","b"])`, `np.array(["a","b"])`): all return the same object as with `None`.
- **Schema-changing fit**: `steps=[("rename_columns", {"mapping": {"a": "x"}})]` with `allow_schema_changes=True`.
  - The output is expected to be `['x','b']`, while `input_features` must still be `["a","b"]`.
  - The step-argument format depends on `arnio.pipeline`, which is not shown. Verify it, or prefer the non-schema cases.
  - Without `allow_schema_changes=True`, `fit` raises `ValueError("Schema-changing step 'rename_columns' not allowed. Use allow_schema_changes=True.")`.
- **Non-str column names** (e.g. int columns `0,1`): passing `[0,1]` hits the "All input_features must be strings" TypeError. Avoid this case otherwise, since `from_pandas` behaviour with it is unknown.

## 7. Rules from the docstring
- "If None, the fitted output feature names are returned."
- Otherwise the argument "must be a non-string sequence of feature names that match the fitted input feature names".

## 8. Testing hints
- Setup:
  ```python
  import pandas as pd, numpy as np, pytest, re
  from arnio.integrations.sklearn import ArnioCleaner
  df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
  est = ArnioCleaner().fit(df)   # steps=[] -> no pipeline run in fit
  ```
- Compare arrays with `list(out) == ["a","b"]` or `np.array_equal`. Never use bare `assert out == [...]`.
- Also assert `out.dtype == object` and `out is est.feature_names_out_`.
- Pin the check order:
  - `[1]` on a 2-column fit gives TypeError (non-string), not ValueError (length).
  - `["a"]` gives the length error.
  - `["b","a"]` gives the mismatch error.
- A safe step that keeps the schema: `steps=[("strip_whitespace",)]`. After `transform`, the names should still be `["a","b"]`.
- Requires scikit-learn to be installed. Use `pytest.importorskip("sklearn")`.
