#!/usr/bin/env python3
"""
Script to query IMPC server for ABR related data.

We utilize the ABR procedure ID IMPC_ABR_002 to query and retrieve the server.
Parameter ID's were attempted for the raw ABR data such as,
6kHz-evoked ABR waveforms (numerical format) with ID IMPC_ABR_005_001 but this failed to
retrieve any hits.

The ID were accessed from the following website:
https://www.mousephenotype.org/impress/ProcedureInfo?action=list&procID=149

Author: L. Barrett
"""

import requests
import json

# Define the SOLR server URL
BASE_URL = "https://www.ebi.ac.uk/mi/impc/solr/experiment/select"

# Define the query parameters
PARAMS = {
    "q": "procedure_stable_id:IMPC_ABR_002",  # this is the ID for all ABR procedures
    "rows": 10,
    "wt": "json"
}

# Send the request
response = requests.get(BASE_URL, params=PARAMS)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = json.loads(response.text)
    
    # Get the number of records
    print(data)
    num_records = data['response']['numFound']
    
    print(f"Number of records with ABR data: {num_records}")
else:
    print(f"Error: Unable to retrieve data. Status code: {response.status_code}")
    print(f"Response content: {response.text}")  # Print the response content for debugging

# Print example data sample
if response.status_code == 200:
    print('Example data sample')
    print(data['response']['docs'][0])