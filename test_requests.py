import argparse
import json
import sys
import numpy as np
np.random.seed(42)

import requests

columns_payload = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]

DEFAULT_URL = "http://localhost:8000/predict"


def send_request():
    new_data = [
        np.random.normal(0, 1) for _ in range(len(columns_payload))
    ]
    print("Sending data:", new_data)
    payload = {}
    for col, val in zip(columns_payload, new_data):
        payload[col] = val
        print(f"  {col}: {val}")
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(DEFAULT_URL, headers=headers, data=json.dumps(payload), timeout=10)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(2)

    print(f"Status code: {resp.status_code}")
    try:
        print("Response JSON:", resp.json())
    except ValueError:
        print("Response text:", resp.text)

    if resp.status_code != 200:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-requests", default=None, help="Number of requests to send", type=int)
    args = parser.parse_args()
    for _ in range(args.nb_requests):
        send_request()
