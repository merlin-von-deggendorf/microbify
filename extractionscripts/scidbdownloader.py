import os
import requests
import threading
import time
import sys
import concurrent.futures

target = '/Users/apple/Documents/Git_grapeDisease_test/microbify'

# read text file
with open('grapes.txt', 'r', encoding='utf-8-sig') as data:
    lines = data.readlines()

# Global counter and error tracker with lock for thread-safe operations.
downloaded_count = 0
lock = threading.Lock()
thread_count = 10


        
def process_line(line):
    global downloaded_count
    max_retries = 100
    attempts = 0
    while attempts < max_retries:
        try:
            url = line.strip()
            # extract query string
            query = url.split('?')[1]
            # extract parameters
            params = query.split('&')
            # extract the value of the parameter
            parameter_dict = {}
            for param in params:
                key, value = param.split('=')
                parameter_dict[key] = value
            # get path value
            path = parameter_dict['path']
            # check if path starts with /V2/
            if not path.startswith('/V2/'):
                raise ValueError('Path does not start with /V2/')
            # extract the folder name of the last part of the path
            folder = path.split('/')[-2]
            # get file name Parameter
            filename = parameter_dict['fileName']
            # if folder does not exist, create it
            target_folder = os.path.join(target, folder)
            filepath = os.path.join(target_folder, filename)
            if os.path.exists(filepath):
                return
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            # download the file using http request
            response = requests.get(url, verify=False)
            # check if the response is successful
            if not response.ok:
                if response.status_code == 429:
                    print(f"Rate limit exceeded. Waiting for 5 minutes.")
                    time.sleep(300)
                print(f"HTTP error: {response.status_code} occurred at timestamp")
                raise ValueError(f"HTTP error: {response.status_code}")
                
            # write the file
            with open(filepath, 'wb') as file:
                file.write(response.content)
            # Increase counter in a thread-safe way.
            with lock:
                downloaded_count += 1
                print(f"Downloaded file count: {downloaded_count}")
            break  # Successful execution, exit the loop.
        except Exception as e:
            attempts += 1
            print(f"Error processing line (attempt {attempts}/{max_retries}): {e}")
            if attempts == max_retries:
                print("Max retries reached. Skipping this line.")

with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
    executor.map(process_line, lines)


