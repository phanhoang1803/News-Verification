# inference_use_retrieved_evidences.py

from typing import Optional, Union
import openai
from modules import EntitiesModule, GPTConnector, GeminiConnector, TextEvidencesModule, ImageEvidencesModule
from mdatasets.newsclipping_datasets import MergedBalancedNewsClippingDataset
from templates import get_visual_prompt, get_final_prompt
from templates_for_generating_context import CAPTION_CONTEXT_CHECKING_RESPONSE_SCHEMA, CONTEXT_RESPONSE_SCHEMA, get_context_prompt, get_caption_context_checking_prompt, SYSTEM_PROMPT_FOR_VLM_GENERATED_CONTEXT, SYSTEM_PROMPT_FOR_CAPTION_CONTEXT_CHECKING
import os
from dotenv import load_dotenv
import argparse
import torch
import json
import time
from utils.utils import process_results
from modules.reasoning_module.connector.gpt import VISUAL_RESPONSE_SCHEMA, FINAL_RESPONSE_SCHEMA

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_dataset", 
                       help="")
    parser.add_argument("--entities_path", type=str, default="test_dataset/links_test.json")
    parser.add_argument("--image_evidences_path", type=str, default="queries_dataset/merged_balanced/inverse_search/test/test.json", 
                        help="")
    parser.add_argument("--text_evidences_path", type=str, default="queries_dataset/merged_balanced/direct_search/test/test.json", 
                        help="")
    parser.add_argument("--context_dir_path", type=str, default="queries_dataset/merged_balanced/context/test")
    parser.add_argument("--random_index_path", type=str, default=None)
    parser.add_argument("--vlm_model", type=str, default="gemini", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--gemini_api_key", type=str, default=None)
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output_dir_path", type=str, default="./result/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors/")
    
    # Dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    
    # Model configs
    parser.add_argument("--ner_model", type=str, default="dslim/bert-large-NER")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip2-opt-2.7b")
    
    return parser.parse_args()

def inference(entities_module: EntitiesModule,
             image_evidences_module: ImageEvidencesModule, 
             text_evidences_module: TextEvidencesModule,
             vlm_connector: Optional[Union[GPTConnector, GeminiConnector]],
             data: dict,
             idx: int,
             context_dir_path: str):

    start_time = time.time()
    
    image_base64 = data["image_base64"]
    
    visual_entities = image_evidences_module.get_entities_by_index(idx)
    # Use threshold=0.0 to get all evidences sorted by similarity
    image_evidences = image_evidences_module.get_evidence_by_index(idx, query=data["caption"], threshold=0.0, reference_image=image_base64, image_similarity_threshold=0.9, max_results=2, min_results=0, use_filter_by_domain=True)
    text_evidences = text_evidences_module.get_evidence_by_index(idx, query= data["caption"], threshold=0.85, reference_image=image_base64, image_similarity_threshold=0.9, max_results=2, min_results=0)
    evidences = image_evidences + text_evidences
    print(len(evidences))
    
    check_info = {
        "entities": visual_entities,
    }
    if len(evidences) > 0:
        # 1: Internal Checking (Image Checking - Image Search)
        visual_prompt = get_visual_prompt(
            caption=data["caption"],
            content=data["content"],
            visual_entities=visual_entities,
            visual_candidates=evidences,
            pre_check=False
        )
        visual_check_result = vlm_connector.call_with_structured_output(
            prompt=visual_prompt,
            schema=VISUAL_RESPONSE_SCHEMA,
            # image_base64=image_base64  # Uncomment if needed
        )
        check_info['evidences'] = [ev.to_dict() for ev in evidences]        
        check_info['result'] = visual_check_result
        check_info["check_type"] = "high quality evidences"
    else:
        image_evidences = image_evidences_module.get_evidence_by_index(idx, 
                                                                       query=data["caption"], 
                                                                       threshold=0.0,
                                                                       reference_image=image_base64, 
                                                                       image_similarity_threshold=0.7, 
                                                                       min_results=0, 
                                                                       max_results=2, 
                                                                       use_filter_by_domain=True,
                                                                       use_filter_by_excluding_domains=True,
                                                                       use_filter_by_unique_domain_title=True) # Sorted by text similarity score
        text_evidences = text_evidences_module.get_evidence_by_index(idx, 
                                                                     query=data["caption"],
                                                                     threshold=0.6,
                                                                     reference_image=image_base64,
                                                                     image_similarity_threshold=0.0,
                                                                     min_results=0, 
                                                                     max_results=2, 
                                                                     use_filter_by_domain=False,
                                                                     use_filter_by_excluding_domains=False,
                                                                     use_filter_by_unique_domain_title=False,
                                                                     sort_by_text_score=False) # Sorted by image similarity score
        
        evidences = image_evidences + text_evidences
        
        if len(evidences) > 0:
            visual_prompt = get_visual_prompt(
                caption=data["caption"],
                content=data["content"],
                visual_entities=visual_entities,
                visual_candidates=evidences,
                pre_check=True
            )
            visual_check_result = vlm_connector.call_with_structured_output(
                prompt=visual_prompt,
                schema=VISUAL_RESPONSE_SCHEMA,
                # image_base64=image_base64  # Uncomment if needed
            )
            check_info['evidences'] = [ev.to_dict() for ev in evidences]
            check_info['result'] = visual_check_result
            check_info["check_type"] = "low quality evidences"
        else:
            if os.path.exists(os.path.join(context_dir_path, f"{idx}.json")):
                with open(os.path.join(context_dir_path, f"{idx}.json"), "r") as f:
                    context_result = json.load(f)
            else:
                # Generate context from image
                print("Generating Context")
                context_prompt = get_context_prompt(entities=visual_entities, caption=data["caption"], news_content=data["content"])
                context_result = vlm_connector.call_with_structured_output(
                    prompt=context_prompt,
                    schema=CONTEXT_RESPONSE_SCHEMA,
                    images=[image_base64],
                    system_prompt=SYSTEM_PROMPT_FOR_VLM_GENERATED_CONTEXT
                )
                print("Generated Context")
                # Save the context result to a file
                os.makedirs(context_dir_path, exist_ok=True)
                file_path = os.path.join(context_dir_path, f"{idx}.json")
                with open(file_path, "w") as f:
                    json.dump(context_result, f, indent=2, ensure_ascii=False)
                    
            # print(context_result)
            caption_context_checking_prompt = get_caption_context_checking_prompt(
                caption=data["caption"],
                context=context_result
            )
            
            visual_check_result = vlm_connector.call_with_structured_output(
                prompt=caption_context_checking_prompt,
                schema=CAPTION_CONTEXT_CHECKING_RESPONSE_SCHEMA, 
                system_prompt=SYSTEM_PROMPT_FOR_CAPTION_CONTEXT_CHECKING
            )
            
            check_info["evidences"] = []
            check_info["result"] = visual_check_result
            check_info["context"] = context_result
            check_info["check_type"] = "context"
    
    # 2: Final Checking (using internal results + direct image analysis)
    final_prompt = get_final_prompt(
        caption=data["caption"],
        content=data["content"],
        visual_check_result=visual_check_result,
    )
    final_result = vlm_connector.call_with_structured_output(
        prompt=final_prompt,
        schema=FINAL_RESPONSE_SCHEMA, 
        images=[image_base64]  # Providing the image for direct analysis
    )
    
    inference_time = time.time() - start_time
    
    result = {
        "caption": data["caption"],
        "ground_truth": data["label"],
        "check_result": check_info,
        "final_result": final_result,
        "inference_time": float(inference_time)
    }
    
    return process_results(result)

def get_transform():
    return None

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    # login(token=os.environ["HF_TOKEN"])
    
    # Make res folder
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path)
    if not os.path.exists(args.errors_dir_path):
        os.makedirs(args.errors_dir_path)
    
    print("Connecting to VLM Model...")
    if args.vlm_model == "gpt":
        vlm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.vlm_model == "gemini":
        vlm_connector = GeminiConnector(
            api_key=args.gemini_api_key if args.gemini_api_key is not None else os.environ["GEMINI_API_KEY"],
            # model_name="gemini-2.0-flash-001"
            model_name="gemini-2.5-flash-preview-04-17"
        )
    else:
        raise ValueError(f"Invalid VLM model: {args.vlm_model}")
    print("VLM Model Connected")
        
        
    # Initialize modules
    print("Connecting to External Retrieval Module...")
    entities_module = EntitiesModule(args.entities_path)
    image_evidences_module = ImageEvidencesModule(args.image_evidences_path)
    text_evidences_module = TextEvidencesModule(args.text_evidences_path)
    
    # Process data and save results
    results = []
    error_items = []
    total_start_time = time.time()

    dataset = MergedBalancedNewsClippingDataset(args.data_path)
    
    start_idx = args.start_idx if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else len(dataset) - 1
    
    # Load random 1000 index from file
    if args.random_index_path != None:    
        try:
            with open(args.random_index_path, "r") as f:
                random_index = [int(line.strip()) for line in f.readlines()]
        except e:
            raise e
    else:
        random_index = list(range(start_idx, end_idx))
        
    # Select indices in random_index which are between start_idx and end_idx
    indices = [idx for idx in random_index if start_idx <= idx <= end_idx]
    
    # Validate indices
    if start_idx >= len(dataset):
        raise ValueError(f"Start index {start_idx} is out of range for dataset of length {len(dataset)}")
    if end_idx >= len(dataset):
        end_idx = len(dataset) - 1
    if start_idx > end_idx:
        raise ValueError(f"Start index {start_idx} is greater than end index {end_idx}")
    
    print(f"Processing items from index {start_idx} to {end_idx}")

    for idx in indices:
        try:
            print(f"Processing item {idx}")
            res_path = os.path.join(args.output_dir_path, f"result_{idx}.json")
            print(f"Path {res_path}")
            item = dataset[idx]
        
            if args.skip_existing and os.path.exists(os.path.join(args.output_dir_path, f"result_{idx}.json")):
                continue
        
            result = inference(
                entities_module=entities_module,
                image_evidences_module=image_evidences_module,
                text_evidences_module=text_evidences_module,
                vlm_connector=vlm_connector,
                data=item,
                idx=idx,
                context_dir_path=args.context_dir_path
            )
            
            with open(os.path.join(args.output_dir_path, f"result_{idx}.json"), "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            results.append(result)
        except KeyError as e:
            print(f"KeyError processing item {idx}: {e}")
            continue
        except openai.BadRequestError as e:
            print(f"BadRequestError processing item {idx}: {e}")
            continue
        except json.decoder.JSONDecodeError as e:
            print(f"e")
            continue
        except UnicodeEncodeError as e:
            print(f"e")
            continue
        except Exception as e:
            with open(os.path.join(args.errors_dir_path, f"error_{idx}.json"), "w") as f:
                error_item = {
                    "error": str(e),
                }
                json.dump(error_item, f, indent=2, ensure_ascii=False)
            error_items.append(error_item)
            print(f"Error processing item {idx}: {e}")
            # raise e
                
    total_time = time.time() - total_start_time
    
if __name__ == "__main__":
    main()