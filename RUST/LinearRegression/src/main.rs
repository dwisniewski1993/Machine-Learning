use std::fs;
use std::error::Error;
use std::path::PathBuf;
use std::collections::HashMap;
use csv::ReaderBuilder;
use ndarray::{Array2, Array1};
use linfa::prelude::*;
use linfa_linear::LinearRegression;

fn read_csv(path: &PathBuf) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut features = vec![];
    let mut targets = vec![];

    let mut label_map: HashMap<String, f64> = HashMap::new();
    let mut next_label: f64 = 0.0;

    for result in rdr.records() {
        let record = result?;
        let len = record.len();

        let row: Vec<f64> = record.iter()
            .map(|s| {
                s.parse::<f64>().unwrap_or_else(|_| {
                    *label_map.entry(s.to_string()).or_insert_with(|| {
                        let label = next_label;
                        next_label += 1.0;
                        label
                    })
                })
            })
            .collect();

        // Last column = Target
        features.push(row[..len - 1].to_vec());
        targets.push(row[len - 1]);
    }

    let x = Array2::from_shape_vec((features.len(), features[0].len()), features.concat())?;
    let y = Array1::from(targets);
    Ok((x, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    let folder_path = "../../_datasets_regression";
    let entries = fs::read_dir(folder_path)?;

    for entry in entries {
        println!("---------------------------------");
        let entry = entry?;
        let path = entry.path();

        if path.extension().map_or(false, |e| e == "csv") {
            println!("📄 File: {}", path.file_name().unwrap().to_string_lossy());

            let (records, targets) = read_csv(&path)?;
            let dataset = Dataset::new(records, targets);

            let model = LinearRegression::default().fit(&dataset)?;
            let preds = model.predict(&dataset);
            let r2 = preds.r2(&dataset)?;
            println!("📊 R² Score: {:.4}", r2);
        }
    }

    Ok(())
}
