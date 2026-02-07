# 🎬 MovieLens 100K Dataset

## Overview

The MovieLens 100K dataset contains movie ratings collected by the GroupLens Research Project at the University of Minnesota.

**Dataset Size:**
- **100,000 ratings** (scale: 1-5 stars)
- **943 users**
- **1,682 movies**
- Each user has rated at least 20 movies

**Collection Period:** September 1997 - April 1998

---

## Data Files

### Raw Data Files (from MovieLens)

#### `u.data`
The complete rating dataset containing all 100,000 ratings.

**Format:** Tab-separated values  
**Columns:** `user_id | movie_id | rating | timestamp`

**Example:**
```
196    242    3    881250949
186    302    3    891717742
22     377    1    878887116
```

- `user_id`: User identifier (1-943)
- `movie_id`: Movie identifier (1-1682)
- `rating`: Rating on 1-5 scale
- `timestamp`: Unix timestamp (seconds since 1/1/1970 UTC)

---

#### `u.user`
Demographic information about users.

**Format:** Pipe-separated values  
**Columns:** `user_id | age | gender | occupation | zip_code`

**Example:**
```
1|24|M|technician|85711
2|53|F|other|94043
3|23|M|writer|32067
```

- `age`: Age in years
- `gender`: M (Male) or F (Female)
- `occupation`: One of 21 occupations (student, engineer, etc.)
- `zip_code`: US zip code

---

#### `u.item`
Movie metadata and genre information.

**Format:** Pipe-separated values  
**Columns:** `movie_id | title | release_date | video_release_date | IMDb_URL | [19 genre columns]`

**Example:**
```
1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
```

**Genre Columns (19 total):**
Each movie can belong to multiple genres. The last 19 columns are binary flags (0 or 1):

1. Unknown
2. Action
3. Adventure
4. Animation
5. Children's
6. Comedy
7. Crime
8. Documentary
9. Drama
10. Fantasy
11. Film-Noir
12. Horror
13. Musical
14. Mystery
15. Romance
16. Sci-Fi
17. Thriller
18. War
19. Western

---

## Processed Competition Files

### Public Files (in `data/public/`)

Participants have access to these files:

#### `graph_data.pt`
PyTorch file containing the complete graph structure.

**Contents:**
```python
{
    'x': tensor,              # Node features [num_nodes, num_features]
    'edge_index': tensor,     # Training edges [2, num_train_edges]
    'edge_attr': tensor,      # Edge attributes (ratings) [num_train_edges, 1]
    'node_type': tensor,      # Node types [num_nodes] (0=user, 1=movie)
    'num_users': int,         # 943
    'num_movies': int,        # 1682
    'num_nodes': int,         # 2625 (943 + 1682)
    'num_features': int       # Feature dimension
}
```

**Node Indexing:**
- Users: indices 0-942
- Movies: indices 943-2624

---

#### `train_ratings.csv`
Training ratings that participants can use to train their models.

**Format:** CSV  
**Columns:** `user_id, movie_id, rating`

**Example:**
```csv
user_id,movie_id,rating
196,242,3
186,302,3
22,377,1
```

**Size:** ~80,000 ratings

---

#### `test_nodes.csv`
Test pairs for which participants must predict ratings.

**Format:** CSV  
**Columns:** `id, user_id, movie_id`

**Example:**
```csv
id,user_id,movie_id
0,1,50
1,1,172
2,2,89
```

**Size:** ~20,000 pairs  
**Note:** NO ratings included - participants must predict these!

---

#### `test_id_mapping.csv`
Mapping between test IDs and graph node indices.

**Format:** CSV  
**Columns:** `id, user_id, movie_id, user_idx, movie_idx`

**Example:**
```csv
id,user_id,movie_id,user_idx,movie_idx
0,1,50,0,992
1,1,172,0,1114
```

**Purpose:** Helps participants map test pair IDs to node indices in the graph.

---

#### `sample_submission.csv`
Example submission file showing the required format.

**Format:** CSV  
**Columns:** `id, y_pred`

**Example:**
```csv
id,y_pred
0,3.0
1,3.5
2,4.2
```

**Requirements:**
- Must contain predictions for all test IDs (0 to num_test-1)
- `y_pred` should be in range [1.0, 5.0]

---

### Private Files (NOT in repo)

#### `test_labels.csv`
Ground truth ratings for test set (used for scoring only).

**Format:** CSV  
**Columns:** `id, user_id, movie_id, rating, is_cold_start`

**Example:**
```csv
id,user_id,movie_id,rating,is_cold_start
0,1,50,4,False
1,1,172,3,True
```

**Storage:** GitHub Secrets (never committed to repository)

---

## Dataset Statistics

### Overall
- **Total ratings:** 100,000
- **Sparsity:** ~6.3% (only 6.3% of possible user-movie pairs have ratings)
- **Rating distribution:**
  - 5 stars: 21,201 (21.2%)
  - 4 stars: 34,174 (34.2%)
  - 3 stars: 27,145 (27.1%)
  - 2 stars: 11,370 (11.4%)
  - 1 star: 6,110 (6.1%)

### Users
- **Average ratings per user:** ~106
- **Min ratings per user:** 20 (enforced in dataset)
- **Max ratings per user:** 737

### Movies
- **Average ratings per movie:** ~59
- **Min ratings per movie:** 1
- **Max ratings per movie:** 583

### Cold-Start Challenge
- **Cold-start threshold:** ≤5 ratings
- **Cold-start users:** ~15% of users
- **Cold-start movies:** ~35% of movies
- **Test set composition:** ~27% cold-start pairs

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F. Maxwell and Konstan, Joseph A.},
  journal={ACM Transactions on Interactive Intelligent Systems (TiiS)},
  volume={5},
  number={4},
  pages={19},
  year={2015},
  publisher={ACM},
  doi={10.1145/2827872}
}
```

---

## License & Usage

**Source:** GroupLens Research Project, University of Minnesota

**Usage Restrictions:**
- ✅ Free for research purposes
- ✅ Must cite in publications
- ❌ Cannot redistribute without permission
- ❌ Cannot use for commercial purposes without permission

**Contact:** grouplens-info@cs.umn.edu

**More info:** https://grouplens.org/datasets/movielens/100k/

---

## Notes

1. **No pre-split files needed:** The raw dataset includes pre-split files (u1.base, u1.test, etc.) but we create our own custom split with cold-start challenge.

2. **Graph structure:** The bipartite graph has edges only between users and movies (no user-user or movie-movie edges by default).

3. **Bidirectional edges:** For GNN message passing, edges are added in both directions (user→movie and movie→user).

4. **Feature engineering:** User features include age (normalized), gender (one-hot), and occupation (one-hot). Movie features are genre one-hot encodings.