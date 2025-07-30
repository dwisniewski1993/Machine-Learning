use rand::Rng;
use linfa::dataset::DatasetBase;
use linfa_trees::{DecisionTree, SplitQuality};
use linfa::prelude::*;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use std::error::Error;

// Przestrzeń i historia
#[derive(Debug)]
pub struct BayesSearchEngine {
    pub history: Vec<(usize, f32, f64)>, // (depth, min_split, score)
    pub iterations: usize,
    pub k_folds: usize,
    pub verbose: bool,
}

impl BayesSearchEngine {
    pub fn new(k_folds: usize, iterations: usize, verbose: bool) -> Self {
        BayesSearchEngine {
            history: Vec::new(),
            iterations,
            k_folds,
            verbose,
        }
    }

    pub fn optimize(&mut self, records: &Array2<f64>, targets: &Array1<usize>) {
        let mut rng = rand::thread_rng();

        for i in 0..self.iterations {
            let (depth, min_split) = if self.history.is_empty() {
                (
                    rng.gen_range(5..=100),
                    rng.gen_range(0.5..=10.0),
                )
            } else {
                self.propose_next_point()
            };

            let score = match cross_validate_model(records, targets, depth, min_split, self.k_folds) {
                Ok(f1) => f1,
                Err(_) => 0.0,
            };

            if self.verbose{
                println!(
                "🔎 Iteracja {}: depth = {}, split = {:.2}, F1-score = {:.4}",
                i + 1,
                depth,
                min_split,
                score
            );}


            self.history.push((depth, min_split, score));
        }

        if let Some((d, s, f)) = self
            .history
            .iter()
            .copied()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        {
            println!(
                "\n✅ Najlepsze parametry:\n   depth = {}, min_split = {:.2}, F1-score = {:.4}",
                d, s, f
            );
        }
    }

    fn propose_next_point(&self) -> (usize, f32) {
        let mut rng = rand::thread_rng();
        let best = self.history.iter().copied().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
        let depth = ((best.0 as i32 + rng.gen_range(-10..10)).clamp(5, 100)) as usize;
        let split = (best.1 + rng.gen_range(-1.0..1.0)).clamp(0.5, 10.0);

        (depth, split)
    }
}

// F1-score
pub fn f1_score(preds: &Array1<usize>, targets: &[usize]) -> f64 {
    let (mut tp, mut fp, mut fn_) = (0, 0, 0);
    for (&p, &t) in preds.iter().zip(targets.iter()) {
        if p == t {
            tp += 1;
        } else if p == 1 && t == 0 {
            fp += 1;
        } else {
            fn_ += 1;
        }
    }

    let prec = tp as f64 / (tp + fp).max(1) as f64;
    let rec = tp as f64 / (tp + fn_).max(1) as f64;
    if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}

// Cross-Validation
pub fn cross_validate_model(
    records: &Array2<f64>,
    targets: &Array1<usize>,
    depth: usize,
    min_split: f32,
    k: usize,
) -> Result<f64, Box<dyn Error>> {
    let n = records.nrows();
    let fold = n / k;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rand::thread_rng());

    let mut scores = vec![];

    for i in 0..k {
        let start = i * fold;
        let end = if i == k - 1 { n } else { (i + 1) * fold };
        let test_idx = &idx[start..end];
        let train_idx = [&idx[..start], &idx[end..]].concat();

        let train_x = records.select(Axis(0), &train_idx).to_owned();
        let train_y = targets.select(Axis(0), &train_idx).to_owned();
        let test_x = records.select(Axis(0), test_idx).to_owned();
        let test_y = targets.select(Axis(0), test_idx).to_owned();

        let train_ds = DatasetBase::new(train_x, train_y);
        let test_ds = DatasetBase::new(test_x, test_y.clone());

        let model = DecisionTree::params()
            .split_quality(SplitQuality::Entropy)
            .max_depth(Some(depth))
            .min_weight_split(min_split)
            .fit(&train_ds)?;

        let preds = model.predict(&test_ds);
        let f1 = f1_score(&preds, &test_y.to_vec());
        scores.push(f1);
    }

    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}
