import requests
import pandas as pd
import json
import os
from time import sleep

def fetch_abr_data(start=0, rows=1000):
    base_url = "https://www.ebi.ac.uk/mi/impc/solr/experiment/select"
    params = {
        "q": "procedure_stable_id:IMPC_ABR_002 AND (parameter_stable_id:IMPC_ABR_003_001 OR parameter_stable_id:IMPC_ABR_005_001 OR parameter_stable_id:IMPC_ABR_007_001 OR parameter_stable_id:IMPC_ABR_009_001 OR parameter_stable_id:IMPC_ABR_011_001 OR parameter_stable_id:IMPC_ABR_013_001)",
        "fl": "parameter_stable_id,parameter_name,gene_symbol,allele_symbol,zygosity,sex,data_point,metadata,experiment_id",
        "rows": rows,
        "start": start,
        "wt": "json"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def process_results(results):
    docs = results['response']['docs']
    processed_data = []
    for doc in docs:
        metadata_dict = {item.split(' = ')[0]: item.split(' = ')[1] for item in doc.get('metadata', [])}
        waveform_data = doc.get('data_point', '')
        if isinstance(waveform_data, list):
            waveform_data = ','.join(map(str, waveform_data))
        processed_data.append({
            'experiment_id': doc.get('experiment_id', ''),
            'parameter_stable_id': doc.get('parameter_stable_id', ''),
            'parameter_name': doc.get('parameter_name', ''),
            'gene_symbol': doc.get('gene_symbol', ''),
            'allele_symbol': doc.get('allele_symbol', ''),
            'zygosity': doc.get('zygosity', ''),
            'sex': doc.get('sex', ''),
            'equipment_id': metadata_dict.get('Equipment ID', ''),
            'equipment_manufacturer': metadata_dict.get('Equipment manufacturer', ''),
            'equipment_model': metadata_dict.get('Equipment model', ''),
            'stimulus_level_range': metadata_dict.get('Range of stimulus levels used - Click', ''),
            'waveform_data': waveform_data
        })
    return processed_data

def main():
    all_data = []
    start = 0
    rows = 100
    max_records = 1000

    while start < max_records:
        print(f"Fetching records {start} to {min(start+rows, max_records)}...")
        results = fetch_abr_data(start, min(rows, max_records-start))
        if not results:
            break
        
        processed_data = process_results(results)
        all_data.extend(processed_data)
        
        if len(all_data) >= max_records:
            all_data = all_data[:max_records]
            break
        
        start += rows
        sleep(1)  # To avoid overwhelming the server

    df = pd.DataFrame(all_data)
    
    # Define the path to save the CSV file
    save_path = '/Volumes/IMPC/impc_abr_data_with_waveforms_10k.csv'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the DataFrame to CSV
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}. Total records: {len(df)}")

if __name__ == "__main__":
    main()