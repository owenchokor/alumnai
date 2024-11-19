from embed.embed import create_page_embeddings, create_sent_embeddings
from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Parse arguments for PDF to Text task")

    parser.add_argument('--pdf_path', type=str, required=True, help="Path to the PDF file")
    parser.add_argument('--text_path', type=str, required=True, help="Path for the STT result file")
    parser.add_argument('--save_path', type=str, required = True, help="Path to save the align result")
    parser.add_argument('--show', type=bool, default=False, help="Flag to show the embedded images (default: False)")
    parser.add_argument('--progress', type=bool, default=True, help="Flag to display progress (default: True)")
    parser.add_argument('--query', type=str, default='long', help="length of image description query (default: long)")

    args = parser.parse_args()

    print('Creating page indexes...')
    page_embeddings = create_page_embeddings(args.pdf_path, show = args.show, progress = args.progress, query=args.query)
    print('Creating sentence indexes...')
    sent_embeddings, strings_list, sig_idx = create_sent_embeddings(args.text_path, args.progress)

    aligned = Utils.embedNalign(sig_idx, page_embeddings, sent_embeddings, length_cutoff=10)
    
    result_json = Utils.get_result(strings_list, aligned)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        f.write(result_json)
        print(f'result sucessfully saved to {args.save_path}')

    plt.scatter(sig_idx, [a[1] for a in aligned], c='blue')
    plt.show()

if __name__ == '__main__':
    main()