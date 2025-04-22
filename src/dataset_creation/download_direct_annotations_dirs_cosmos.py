
#!/usr/bin/env python
# coding: utf-8

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import time
from collections import defaultdict

from PIL import Image
import imghdr
import requests
import tqdm

from googleapiclient.discovery import build
from utils import get_captions_from_page, save_html, merge_search_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Download dataset for direct search queries')
    parser.add_argument('--cosmos_data_path', type=str, default='test_dataset_cosmos/public_test_acm.json',
                        help='path to the visual news data')
    parser.add_argument('--save_folder_path', type=str, default='queries_dataset_cosmos',
                        help='location where to download data')
    
    parser.add_argument('--google_api_key', type=str, required=True,
                        help='api_key for the custom search engine')
    parser.add_argument('--google_cse_id', type=str, required=True,
                        help='custom search engine id')                    

    parser.add_argument('--sub_split', type=str, default='test',
                        help='which split to use from train,val,test splits')
                        
    parser.add_argument('--how_many_queries', type=int, default=1,
                        help='how many query to issue for each item - each query is 10 images')
    parser.add_argument('--continue_download', type=int, default=1,
                        help='whether to continue download or start from 0 - should be 0 or 1')

    parser.add_argument('--how_many', type=int, default=-1,
                        help='how many items to query and download, 0 means download untill the end')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='where to end, if not specified, will be inferred from how_many')    
    parser.add_argument('--start_idx', type=int, default=-1,
                        help='where to start, if not specified will be inferred from the current saved json or 0 otherwise')
    parser.add_argument('--random_index_path', type=str, default=None,
                        help='path to the file containing the random indices')

    parser.add_argument('--hashing_cutoff', type=int, default=15,
                        help='threshold used in hashing')
    parser.add_argument('--skip_existing', action="store_true")
    
    args = parser.parse_args()
    return args

def google_search(search_term, api_key, cse_id, how_many_queries, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res_list = []
    for i in range(0,how_many_queries):
        start = i*10 + 1
        
        exact_res = service.cse().list(
            q=f"\"{search_term}\"", 
            searchType='image', 
            lr='lang_en', 
            num=10, 
            start=start, 
            cx=cse_id, 
            **kwargs
        ).execute()
        
        # Second query: search without quotes
        broad_res = service.cse().list(
            q=search_term, 
            searchType='image', 
            lr='lang_en', 
            num=10, 
            start=start, 
            cx=cse_id, 
            **kwargs
        ).execute()
        
        # Merge the results
        combined_results = merge_search_results(exact_res, broad_res)
        res_list.append(combined_results)
        
    return res_list

def init_files_and_paths(args):
    """Initialize files and paths needed for the script"""
    full_save_path = os.path.join(args.save_folder_path, 'direct_search', args.sub_split)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Initialize files
    json_download_file_name = os.path.join(full_save_path, args.sub_split + '.json')
    
    # Initialize or load existing annotations
    if os.path.isfile(json_download_file_name) and os.access(json_download_file_name, os.R_OK) and args.continue_download:
        with open(json_download_file_name, 'r') as fp:
            all_direct_annotations_idx = json.load(fp)
    else:
        all_direct_annotations_idx = {}
        with open(json_download_file_name, 'w') as db_file:
            json.dump({}, db_file)
    
    return full_save_path, json_download_file_name, all_direct_annotations_idx

def download_and_save_image(image_url, save_folder_path, file_name):
    try:
        response = requests.get(image_url,stream = True,timeout=(60,60))
        if response.status_code == 200:
            response.raw.decode_content = True
            image_path = os.path.join(save_folder_path,file_name+'.jpg')
            with open(image_path,'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if imghdr.what(image_path).lower() == 'png':
                img_fix = Image.open(image_path)
                img_fix.convert('RGB').save(image_path)
            return 1 
        else:
            return 0
    except:
        return 0 

def process_single_item(item_data):
    """Process a single search result item"""
    item, counter, save_folder_path = item_data
    image = {}
    
    # Basic information extraction
    for key, target in [('link', 'img_link'), ('displayLink', 'domain')]:
        if key in item:
            image[target] = item[key]
    
    if 'image' in item and 'contextLink' in item['image']:
        image['page_link'] = item['image']['contextLink']
    if 'snippet' in item:
        image['snippet'] = item['snippet']

    # Download image
    if not download_and_save_image(item['link'], save_folder_path, str(counter)):
        return None

    image['image_path'] = os.path.join(save_folder_path, f"{counter}.jpg")

    try:
        caption, title, code, req = get_captions_from_page(
            item['link'], 
            item['image']['contextLink']
        )
    except Exception as e:
        print(f'Error in getting captions for item {counter}: {str(e)}')
        return None

    # Save HTML
    if save_html(req, os.path.join(save_folder_path, f"{counter}.txt")):
        image['html_path'] = os.path.join(save_folder_path, f"{counter}.txt")

    if code and code[0] in ['4', '5']:
        image['is_request_error'] = True

    # Process title
    item_title = item.get('title', '') or ''
    title = title if title is not None else ''
    image['page_title'] = title if len(title) > len(item_title.strip()) else item_title

    # Process caption
    if caption:
        image['caption'] = caption
        return ('with_captions', image)
    
    try:
        caption, title, code, req = get_captions_from_page(
            item['link'],
            item['image']['contextLink'],
            req,
            # args.hashing_cutoff
        )
    except Exception as e:
        print(f'Error in getting captions for item {counter} (second attempt): {str(e)}')
        return None

    if caption:
        image['caption'] = caption
        return ('matched_tags', image)
    
    return ('no_captions', image)

def get_direct_search_annotation(search_results_lists, save_folder_path):
    """Process search results in parallel"""
    items_to_process = []
    counter = 0
    
    for result_list in search_results_lists:
        if 'items' in result_list:
            for item in result_list['items']:
                items_to_process.append((item, counter, save_folder_path))
                counter += 1

    if not items_to_process:
        return {}

    results = defaultdict(list)
    
    # Use a context manager for ProcessPoolExecutor
    with cf.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_item, item_data): item_data
            for item_data in items_to_process
        }
        
        try:
            for future in cf.as_completed(futures, timeout=60):  # Global timeout
                try:
                    result = future.result(timeout=30)  # Timeout per task
                    if result:
                        category, image = result
                        results[category].append(image)
                except Exception as e:
                    item_data = futures[future]
                    print(f'Failed to process item {item_data[1]}: {str(e)}')
        
        except KeyboardInterrupt:
            print("🛑 User interrupted! Shutting down all processes...")
            executor.shutdown(wait=False, cancel_futures=True)  # 🚀 Force stop all workers
            raise  # Re-raise KeyboardInterrupt
        
        except Exception as e:
            print(f"🔥 Critical error: {str(e)}. Forcing shutdown.")
            executor.shutdown(wait=False, cancel_futures=True)  # 🚀 Force stop all workers

    if not results:
        return {}

    return {
        'images_with_captions': results['with_captions'],
        'images_with_no_captions': results['no_captions'],
        'images_with_caption_matched_tags': results['matched_tags']
    }

def main():
    args = parse_arguments()
    
    # Initialize environment and paths
    full_save_path, json_download_file_name, all_direct_annotations_idx = init_files_and_paths(args)
    
    # Load datasets
    cosmos_data = json.load(open(args.cosmos_data_path))
    
    # Determine start and end indices
    start_counter = (args.start_idx if args.start_idx != -1 
                    else (int(list(all_direct_annotations_idx.keys())[-1])+2 
                          if all_direct_annotations_idx else 0))
    
    end_counter = (args.end_idx if args.end_idx > 0 
                  else (start_counter + 2*args.how_many if args.how_many > 0 
                        else len(cosmos_data)))
    
    if args.random_index_path:
        try:
            with open(args.random_index_path, 'r') as f:
                random_indices = [int(line.strip()) for line in f.readlines()]
        except Exception as e:
            print(f"Error in reading random indices file: {str(e)}")
    else:
        random_indices = list(range(start_counter, end_counter))
    
    print(f"Processing items from {random_indices[0]} to {random_indices[-1]}")
    
    # Main processing loop
    for i in tqdm.tqdm(random_indices):
        if args.skip_existing:
            if os.path.exists(os.path.join(full_save_path, str(i))):
                # If the folder exists, and the direct_annotation.json file exists, skip the item
                if os.path.exists(os.path.join(full_save_path, str(i), 'direct_annotation.json')):
                    continue
        
        start_time = time.time()
        
        try:
            ann = cosmos_data[i]
            text_query = ann["caption1"]
        except Exception as e:
            print(f"Skipping item {i} due to error: {str(e)}")
            continue
            
        new_folder_path = os.path.join(full_save_path, str(i))
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Process single query
        result = google_search(text_query, args.google_api_key, args.google_cse_id, 
                             how_many_queries=args.how_many_queries)
        
        direct_search_results = get_direct_search_annotation(result, new_folder_path)
        
        # Save results
        if direct_search_results:
            new_entry = {
                str(i): {
                    'img_local_path': ann["img_local_path"],
                    'folder_path': new_folder_path
                }
            }
            
            try:
                # WINDOWS
                from filelock import FileLock
                lock_file = f"{json_download_file_name}.lock"
                with FileLock(lock_file):
                    with open(json_download_file_name, 'r') as f:
                        current_data = json.load(f)
                    current_data.update(new_entry)
                    with open(json_download_file_name, 'w') as f:
                        json.dump(current_data, f)
                
                with open(os.path.join(new_folder_path, 'direct_annotation.json'), 'w') as f:
                    json.dump(direct_search_results, f)
            except Exception as e:
                print(f"Error saving results for item {i}: {str(e)}")
        
        print(f"Processed item {i} in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()