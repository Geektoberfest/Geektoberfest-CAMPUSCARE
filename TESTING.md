# CAMPUSCARE — Complaint Evaluation Tests

This document describes the complaint evaluation test suite added in this branch
(`feature/complaint-evaluation-tests`), including what is covered, how to run
the tests, and the known limitations of a hand-written evaluation set.

---

## How to run

```bash
# Activate the virtual environment first
python manage.py test complaints

# Verbose mode (shows each test name and docstring)
python manage.py test complaints --verbosity=2
```

All 68 tests should pass in under one second.

---

## What is covered

### 1. `ComplaintModelTest` — Model field tests (5 tests)
Verifies that the `Complaint` database model stores all fields correctly,
defaults `status` to `"Open"`, defaults `is_duplicate` to `False`, and has
the expected `__str__` representation.

### 2. `CategoryPredictionTest` — Category prediction (15 tests)
Representative examples for **every supported category**:

| Category       | Examples covered                                                    |
|----------------|----------------------------------------------------------------------|
| Academic       | Grade not updated, exam schedule conflict, course registration issue |
| Hostel         | Water supply missing, room maintenance, warden complaint             |
| Infrastructure | Projector not working, washroom broken, classroom door broken        |
| IT & Wi-Fi     | Wi-Fi down, Wi-Fi login portal down, slow internet                   |
| Transport      | Bus late, bus route skipping stop, reckless bus driver               |

### 3. `PriorityPredictionTest` — Priority prediction (17 tests)
Covers all three priority levels with one test per trigger word or phrase:

| Priority | Trigger words / phrases tested                                                       |
|----------|--------------------------------------------------------------------------------------|
| High     | `emergency`, `fire`, `urgent`, `danger`, `medical`, `no water`, `power outage`       |
| Medium   | `not working`, `broken`, `slow`, `problem`, `issue`, `damaged`, `missing`            |
| Low      | General feedback, suggestion, neutral inquiry (no trigger words)                     |

### 4. `EdgeCaseTest` — Edge cases (13 tests)
Ensures `predict_category` and `predict_priority` handle unusual input without
raising an exception and always return a valid value:

- Empty string
- Whitespace-only string
- Very long text (500 words)
- Special characters (`!@#$%^&*()`)
- Numeric-only string
- Mixed-case trigger words (case-insensitivity check)
- Single keyword

### 5. `DuplicateDetectionTest` — Duplicate detection (5 tests)
Validates cosine-similarity-based duplicate detection:

- Returns `False` when the DB is empty
- Flags an exact re-submission as a duplicate
- Flags a near-paraphrase as a duplicate
- Does **not** flag a clearly different complaint
- Does not flag a unique complaint even when many others exist in the DB

### 6. `PredictionConsistencyTest` — Determinism & return types (4 tests)
- `predict_category` and `predict_priority` are called 5 times each with the same
  input; all results must be identical (model is deterministic).
- Both functions always return a non-empty `str`, never `None`.

### 7. `HomeViewTest` — End-to-end view tests (9 tests)
Full HTTP round-trip through the Django view:

- `GET /` returns 200
- `POST` creates a DB record
- Response context contains `category`, `priority`, and `duplicate` fields
- A second identical submission is flagged as a duplicate
- Academic and Transport complaints are classified correctly end-to-end
- An urgent complaint is tagged High priority end-to-end

---

## Limitations

> **This hand-written evaluation set does NOT prove real-world model accuracy.**

- The 15 category examples are representative, not exhaustive.  
  The baseline model trained on 400 complaints may misclassify inputs that use
  vocabulary outside the training data (e.g. "student portal login" is
  ambiguous between Academic and IT & Wi-Fi on the current dataset).
- The priority test set covers only the keywords listed in `ml.py`'s rule
  lists.  New keywords added to `ml.py` will not be tested until tests are
  updated.
- Duplicate detection uses a cosine-similarity threshold of 0.75.  Very short
  or highly technical texts may behave unexpectedly.
- No multilingual inputs are tested here; see `MULTILINGUAL.md` on the
  `feature/multilingual-support` branch for that coverage.
