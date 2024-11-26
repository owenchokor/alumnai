from langchain_community.document_loaders import PyPDFLoader
from src.utils import Utils
from src.llm import EmbeddingModel, ImageModel
import numpy as np
import io
from PIL import Image
import fitz
import textwrap
from tqdm import tqdm
import yaml
import json

class Embedder:

    @staticmethod
    def create_page_embeddings(pdf_path, show, query):
        data_load = PyPDFLoader(pdf_path)
        doc = fitz.open(pdf_path)
        image_to_text = {}
        loop = tqdm(range(len(doc)), desc= 'total pages', total=len(doc))

        for page_num in loop:
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            image_descriptions = []
            
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                try:
                    image_description = Embedder.process_image(image_bytes, show, query)
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

        embeddings = Embedder.get_embeddings(pdftostring)
        return embeddings

    @staticmethod
    def get_embeddings(strings_list):
        embeddings = []
        titan = EmbeddingModel()
        loop = tqdm(strings_list, desc = 'embedding strings', total = len(strings_list))
        for text_input in loop:
            request_body = json.dumps({
                'inputText': text_input
            })
            embedding_vector = titan.getRequest(request_body)
            embeddings.append(embedding_vector)
        
        return embeddings

    @staticmethod
    def stringToVec(sentences: list[dict], progress, length_cutoff=10) -> list[list]:
        strings = [x["String"] for x in sentences if len(x["String"]) > length_cutoff]
        return Embedder.get_embeddings(strings, progress)

    @staticmethod
    def process_image(image_data, show, query):
        img = Image.open(io.BytesIO(image_data))
        max_size = (1000, 1000)
        img.thumbnail(max_size)
        if Embedder.__is_informative_color(img):

            if query == 'none':
                return None
            else:
                with open('./src/queries.yaml', 'r') as file:
                    queries = yaml.safe_load(file)
                

                user_message = queries['queries'][query]
                imgmodel = ImageModel()
                response_out = imgmodel.getRequest(img, user_message)

                if show:
                    import matplotlib.pyplot as plt
                    import time
                    plt.imshow(img)
                    plt.axis('off')
                    wrapped_response = "\n".join(textwrap.wrap(response_out, width=50))
                    plt.figtext(0.5, 0.01, wrapped_response, ha='center', fontsize=10)
                    plt.show(block = False)
                    time.sleep(1)
                    plt.close()
                
                return response_out
        else:
            return None

    @staticmethod
    def create_sent_embeddings(text_path, length_cutoff = 10):
        strings_list, filtered_strings_list, sig_idx = Utils.search_format(text_path, length_cutoff)
        embeddings = Embedder.get_embeddings(filtered_strings_list)
        
        return embeddings, strings_list, sig_idx

    @staticmethod
    def __is_informative_color(image: Image.Image, threshold: float = 0.001) -> bool:
        rgb_image = image.convert("RGB")
        pixel_values = np.array(rgb_image)
        red_channel = pixel_values[:, :, 0]
        green_channel = pixel_values[:, :, 1]
        blue_channel = pixel_values[:, :, 2]
        red_variance = np.var(red_channel)
        green_variance = np.var(green_channel)
        blue_variance = np.var(blue_channel)
        total_pixels = rgb_image.size[0] * rgb_image.size[1]
        normalized_red_variance = red_variance / total_pixels
        normalized_green_variance = green_variance / total_pixels
        normalized_blue_variance = blue_variance / total_pixels
        combined_normalized_variance = (normalized_red_variance + normalized_green_variance + normalized_blue_variance) / 3
        
        return combined_normalized_variance > threshold