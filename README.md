# BRIGHT: Bipartite Rating Inference on Heterogeneous Graphs with Cold-Start

**Predict movie ratings on a sparse bipartite graph with cold-start users and movies.**

---

## 📋 Overview

Bright challenges you to build a model that predicts movie ratings in a graph neural network setting. The dataset contains user-movie interactions represented as a bipartite graph, where some users and movies have very few ratings (the "cold start" problem).

**Your task:** Given a user and a movie, predict the rating (1-5 stars) the user would give.

**Evaluation metric:** Root Mean Squared Error (RMSE) — lower is better

---

## 🎯 Challenge

The key difficulty is **generalization to sparse data**:
- 33.5% of movies have ≤10 ratings
- Test set is 70.9% cold-start cases
- Can your model handle both popular and niche items?

This mirrors real-world recommender systems where new items constantly enter the catalog with minimal interaction history.

---

## 📊 Dataset

### Files

| File | Description |
|------|-------------|
| `graph_data.pt` | PyTorch graph object with node features, edges, and ratings |
| `train_ratings.csv` | Training labels: user_id, movie_id, rating |
| `test_nodes.csv` | Test pairs to predict: id, user_id, movie_id |
| `test_id_mapping.csv` | Helper mapping IDs to graph indices |
| `sample_submission.csv` | Example submission format |

### Graph Structure

**Bipartite graph:** 943 users ↔ 1,682 movies

- **Nodes:** 2,625 total (users indexed 0-942, movies 943-2624)
- **Edges:** ~95,000 training ratings (bidirectional for message passing)
- **Edge attributes:** Rating values (1.0 to 5.0)

### Node Features

**Users (23 dimensions):**
- Age (standardized)
- Gender (binary)
- Occupation (21 categories, one-hot)

**Movies (23 dimensions):**
- Genre indicators (19 genres)
- Padding to match user dimension

### Data Split

| Split | Ratings | Description |
|-------|---------|-------------|
| Train | 96,647 | Visible to participants |
| Test | 3353 | Hidden labels, scored in CI |

**Cold-start composition:**
- 0% cold-start users (all have >10 ratings)
- 33.5% cold-start movies (≤10 ratings)
- 70.9% of test set involves cold-start movies

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/bright.git
cd bright
```

### 2. Explore the data

```python
from starter_code.load_data import load_competition_data

graph_data, train_df, test_df = load_competition_data('./data/public')

print(f"Nodes: {graph_data['num_nodes']}")
print(f"Training ratings: {len(train_df)}")
print(f"Test pairs: {len(test_df)}")
```

### 3. Run the baseline

```bash
cd starter_code
python baseline.py
```

This creates a Random Forest baseline submission (~1.05 RMSE).

### 4. Build your model

Use any Graph Neural Networks model (GCN, GraphSAGE, GAT)

Train on `train_ratings.csv` and the graph structure.

### 5. Generate predictions

Your submission must be a CSV file:

```csv
id,y_pred
0,4.23
1,3.87
2,4.51
...
```

- `id`: From `test_nodes.csv`
- `y_pred`: Predicted rating [1.0, 5.0]

Use the helper function:

```python
from starter_code.load_data import create_submission

predictions = model.predict(test_df)
submission = create_submission(test_df, predictions)
submission.to_csv('predictions.csv', index=False)
```

---

## 📤 How to Submit

### Step 1: Fork this repository

### Step 2: Create submission folder

```bash
mkdir -p submissions/inbox/your_team_name/run_001
```

### Step 3: Add your files

```
submissions/inbox/your_team_name/run_001/
├── predictions.csv          # Required
└── metadata.json           # Optional
```

**metadata.json** (optional):
```json
{
  "team": "your_team_name",
  "model": "human",
  "notes": "GraphSAGE with 3 layers, dropout=0.3"
}
```

Valid `model` values: `human`, `llm-only`, `human+llm`

### Step 4: Open a Pull Request

1. Commit your submission:
```bash
git add submissions/inbox/your_team_name/
git commit -m "Add submission: your_team_name/run_001"
git push origin your-branch
```

2. Open a PR to `main` branch

3. **Automatic scoring** will run within 2-3 minutes

4. Check the PR comment for your score:

```
🎯 Submission Score

| Metric | Score |
|--------|-------|
| RMSE (Overall) | 0.9234 |
| RMSE (Cold Start) | 1.1456 |
| RMSE (Warm Start) | 0.8891 |

✅ Submission is valid and scored!
```

### Step 5: Add to leaderboard

Your score appears on the public leaderboard!

---

## 🏆 Leaderboard

View current standings:
- **Markdown:** [leaderboard/leaderboard.md](leaderboard/leaderboard.md)
- **Interactive:** Enable GitHub Pages for sortable, filterable view

Rankings are sorted by **overall RMSE** (lower is better), with breakdowns for cold-start and warm-start performance.

---

## 📏 Evaluation

Submissions are evaluated using **Root Mean Squared Error (RMSE)**:

```
RMSE = sqrt(mean((y_true - y_pred)²))
```

We report three metrics:
- **Overall RMSE** (primary ranking metric)
- **Cold-start RMSE** (movies with ≤10 ratings)
- **Warm-start RMSE** (movies with >10 ratings)

**Lower scores are better.** A naive baseline (predict global mean) achieves ~1.12 RMSE.

---

## 📜 Rules

### Allowed ✅
- Any GNN model architecture 
- Unlimited offline training time
- Feature engineering on provided data
- Multiple submissions per team

### Not Allowed ❌
- External datasets or pretrained embeddings
- Manual labeling of test data
- Modifying evaluation scripts
- Accessing test labels 

**Violations will result in disqualification.**

---

## 🔬 For Researchers: Human vs LLM Studies

Bright supports controlled human-vs-LLM comparisons:

### Experimental Setup
1. **Fix constraints:** Same time budget (e.g., 2 hours), submission limit (e.g., 5 runs)
2. **Track metadata:** Set `model` field in `metadata.json`
3. **Measure:**
   - Best RMSE within K submissions
   - Validity rate (% submissions passing validation)
   - Cold-start vs warm-start performance
   - Learning curve (score vs submission index)

### Metadata Fields
```json
{
  "model": "human" | "llm-only" | "human+llm",
  "llm_name": "gpt-4" | "claude-3" | null,
  "time_spent_minutes": 120,
  "notes": "Free-form description"
}
```

---

## 💡 Tips

- **Start simple:** Beat the baseline before trying complex GNNs
- **Handle cold-start:** Consider user/movie mean ratings as features
- **Use the graph:** Message passing can help, but isn't required
- **Validate locally:** Use `competition/validate_submission.py` before submitting
- **Watch for overfitting:** Test set is only 5% of data

---

## 🛠️ Code Structure

```
.
├── data/public/              # Training data and graph
├── starter_code/             # Baseline and utilities
├── competition/              # Evaluation scripts (do not modify)
├── submissions/inbox/        # Submit your predictions here
├── leaderboard/              # Auto-updated rankings
└── .github/workflows/        # Auto-scoring pipeline
```

---

## 📞 Support

- **Issues:** Open a GitHub issue for bugs or questions
- **Discussions:** Use GitHub Discussions for modeling questions
- **Updates:** Watch the repository for announcements

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🎓 Citation

If you use Bright in research, please cite:

```bibtex
@misc{bright_2025,
  title={BRIGHT: Bipartite Rating Inference on Heterogeneous Graphs with Cold-Start},
  author={BOULGHERAIF Fatma Zohra},
  year={2025},
  publisher={GitHub},
  url={https://github.com/b-fatma/bright}
}
```

---

**Good luck! 🚀**
