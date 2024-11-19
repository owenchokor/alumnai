from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from utils import Utils
import numpy as np
import io
from PIL import Image
import fitz
import boto3
import textwrap
from tqdm import tqdm
import yaml
import json


def create_page_embeddings(pdf_path, show, progress, query):
    data_load = PyPDFLoader(pdf_path)
    doc = fitz.open(pdf_path)

    image_to_text = {}
    bedrock_client = boto3.client(
        service_name='bedrock-runtime', 
        region_name='us-west-2'
    )
    
    IMG_MODEL_ID = "us.meta.llama3-2-11b-instruct-v1:0"
    if progress:
        loop = tqdm(range(len(doc)), desc= 'total pages', total=len(doc))
    else:
        loop = range(len(doc))
    for page_num in loop:
        page = doc.load_page(page_num)  # Load the current page
        
        # Extract images and generate descriptions using LLaMA
        image_list = page.get_images(full=True)
        image_descriptions = []
        
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get the image description using the LLaMA model
            try:
                image_description = process_image(image_bytes, bedrock_client, IMG_MODEL_ID, show, query)
                if image_description:
                    image_descriptions.append(image_description)
            except Exception as e:
                print(f"Error processing image: {e}")
        image_to_text.update({page_num:image_descriptions})
    pdftostring = []
    for i, page in enumerate(data_load.load()):
        read_image = image_to_text[i]
        for desc in read_image:
            page.page_content += ' ' + desc
        if not page.page_content:
            page.page_content = '.'
        pdftostring.append(page.page_content)
    return get_embeddings(pdftostring, progress)


def get_embeddings(strings_list, progress):
    embeddings = []
    runtime = boto3.client(
        service_name='bedrock-runtime', 
        region_name='us-west-2'
    )
    if progress:
        loop = tqdm(strings_list, desc = 'embedding strings', total = len(strings_list))
    else:
        loop = strings_list
    for text_input in loop:
        # Prepare request body for the Titan model
        request_body = json.dumps({
            'inputText': text_input
        })
        
        # Invoke the Titan embedding model (via Bedrock Runtime)
        response = runtime.invoke_model(
            modelId='amazon.titan-embed-text-v1',  # Example Titan embedding model ID; replace with the correct one
            accept='application/json',
            contentType='application/json',
            body=request_body
        )
        
        # Extract the embedding vector from the response
        response_body = json.loads(response['body'].read())
        embedding_vector = response_body['embedding']
        embeddings.append(embedding_vector)
    
    return embeddings

def stringToVec(sentences: list[dict], progress, length_cutoff=10) -> list[list]:
    strings = [x["String"] for x in sentences if len(x["String"]) > length_cutoff]
    return get_embeddings(strings, progress)

def process_image(image_data, bedrock_client, model_id, show, query):
    img = Image.open(io.BytesIO(image_data))  # Open image from binary data
    max_size = (1000, 1000)
    img.thumbnail(max_size)
    if is_informative_color(img):
        # Display image (optional)
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')  # Convert image to PNG
        image_bytes = img_byte_array.getvalue()
        if query == 'none':
            return None
        else:
            # Instruction for the LLaMA model to describe the image
            with open('./embed/queries.yaml', 'r') as file:
                queries = yaml.safe_load(file)
            

            user_message = queries['queries'][query]
            
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                            {"text": user_message},
                        ],
                    }
                ]

                # Call Bedrock client with LLaMA model to describe the image
                response = bedrock_client.converse(
                    modelId=model_id,
                    messages=messages,
                )
                response_out = response["output"]["message"]["content"][0]["text"]
            except Exception as e:
                print(e)
                return None
            if show:
                import matplotlib.pyplot as plt
                import time
                plt.imshow(img)
                plt.axis('off')
                wrapped_response = "\n".join(textwrap.wrap(response_out, width=50))
                plt.figtext(0.5, 0.01, wrapped_response, ha='center', fontsize=10)
                plt.show()
                time.sleep(1)
                plt.close()
            
            return response_out
    else:
        return None

def create_sent_embeddings(text_path, progress, length_cutoff = 10):

    strings_list, filtered_strings_list, sig_idx = Utils.search_format(text_path, length_cutoff)
    embeddings = get_embeddings(filtered_strings_list, progress)
    
    return embeddings, strings_list, sig_idx

def is_informative_color(image: Image.Image, threshold: float = 0.001) -> bool:
    """
    Determine if a color image is informative based on pixel variance across RGB channels,
    normalized by the number of pixels.

    Parameters:
    image (PIL.Image.Image): The image to be analyzed.
    threshold (float): The normalized variance threshold below which the image is considered uninformative. Default is 0.01.

    Returns:
    bool: Returns True if the image is likely to contain information, False otherwise.
    """
    # Convert the image to RGB (if not already)
    rgb_image = image.convert("RGB")
    
    # Convert the image to a NumPy array (with 3 channels: R, G, B)
    pixel_values = np.array(rgb_image)
    
    # Split the image into three channels (R, G, B)
    red_channel = pixel_values[:, :, 0]
    green_channel = pixel_values[:, :, 1]
    blue_channel = pixel_values[:, :, 2]
    
    # Calculate the variance for each channel
    red_variance = np.var(red_channel)
    green_variance = np.var(green_channel)
    blue_variance = np.var(blue_channel)
    
    # Calculate the total number of pixels
    total_pixels = rgb_image.size[0] * rgb_image.size[1]
    
    # Normalize the variances by the total number of pixels
    normalized_red_variance = red_variance / total_pixels
    normalized_green_variance = green_variance / total_pixels
    normalized_blue_variance = blue_variance / total_pixels
    
    # Calculate the combined normalized variance (average normalized variance across all channels)
    combined_normalized_variance = (normalized_red_variance + normalized_green_variance + normalized_blue_variance) / 3
    
    # If the combined normalized variance is below the threshold, return False, otherwise True
    return combined_normalized_variance > threshold