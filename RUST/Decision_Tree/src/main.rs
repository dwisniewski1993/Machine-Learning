mod bayes_search;

use bayes_search::BayesSearchEngine;
use ndarray::{Array1, Array2};
use csv::{ReaderBuilder, Trim};
use std::error::Error;
use std::fs::File;
use std::fs;
use std::path::PathBuf;

// Czyta CSV i konwertuje do Array2 i Array1
fn read_csv(file_path: PathBuf) -> Result<(Array2<f64>, Array1<usize>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().trim(Trim::All).from_reader(file);

    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let data: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap()).collect();
        targets.push(*data.last().unwrap() as usize);
        records.push(data[..data.len() - 1].to_vec());
    }

    let num_records = records.len();
    let num_features = records[0].len();
    let mut records_array = Array2::zeros((num_records, num_features));
    for (i, row) in records.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            records_array[[i, j]] = val;
        }
    }

    Ok((records_array, Array1::from(targets)))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Decision Tree with Bayesian Search");

    let folder_path = "../../_datasets_classification";
    let entries = fs::read_dir(folder_path)?;

    for entry in entries {
        println!("---------------------------------");
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "csv") {
            println!("File: {}", path.file_name().unwrap().to_string_lossy());
            let (records, targets) = read_csv(path)?;

            // Start Searching
            let mut engine = BayesSearchEngine::new(5, 30, true); // 5-fold, 30 iter
            engine.optimize(&records, &targets);
        }
    }

    Ok(())
}