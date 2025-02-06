# Copyright 2025 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests


WIKIDATA_SPARQL_ENTRY = "https://query.wikidata.org/sparql"


def get_wikidata_count(entity_id: str):
    """Query Wikidata API for references count of a given entity as subject or object."""

    url = WIKIDATA_SPARQL_ENTRY
    query = f"""
    SELECT (COUNT(DISTINCT ?subject) AS ?subject_count) (COUNT(DISTINCT ?object) AS ?object_count) WHERE {{
      {{
        ?subject ?p wd:{entity_id} .
      }} UNION {{
        wd:{entity_id} ?p ?object .
      }}
    }}
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(
        url, headers=headers, params={"query": query, "format": "json"}
    )
    data = response.json()

    try:
        subject_count = int(data["results"]["bindings"][0]["subject_count"]["value"])
    except (IndexError, KeyError):
        subject_count = 0  # Default to 0 if no data found or error in response

    try:
        object_count = int(data["results"]["bindings"][0]["object_count"]["value"])
    except (IndexError, KeyError):
        object_count = 0  # Default to 0 if no data found or error in response

    total_count = subject_count + object_count
    return total_count
