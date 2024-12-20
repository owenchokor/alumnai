from src.embed import Embedder
from src.utils import Utils
from src.annot import PDFAnnotator
from src.stt import transcribe_audio, save_segments_to_txt
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import pickle
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Parse arguments for PDF to Text task")
    parser.add_argument('--pdf_path', type=str, required=True, help="Path to the PDF file")
    #parser.add_argument('--text_path', type=str, required=True, help="Path for the STT result file")
    parser.add_argument('--audio_path', type=str, required=True, help="Path to the Audio file")
    parser.add_argument('--model_path', type=str, default='large-v3', help="Directory to the Model") # ../model에 가중치 추가해야 함
    parser.add_argument('--show', type=bool, default=False, help="Flag to show the embedded images (default: False)")
    parser.add_argument('--query', type=str, default='long', help="length of image description query (default: long)")
    parser.add_argument('--align_test', type=str, default='', help="path to parent folder of data.pkl, if only align needed (default: Empty String)")
    args = parser.parse_args()

    if args.align_test:
        with open(os.path.join(args.align_test, 'data.pkl'), 'rb') as file:
            saved = pickle.load(file)
        sig_idx = saved[0]
        page_embeddings = saved[1]
        sent_embeddings = saved[2]
        strings_list = saved[3]
        log_dir = saved[4]
        output_filename = saved[5]
    else:
        # Speech To Text
        print('Transcribing audio to text...')
        script = transcribe_audio(args.audio_path, args.model_path, args.device) # script는 문장들의 리스트 
        save_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        text_path = os.path.join('logs', save_name + '.txt')
        save_segments_to_txt(script, text_path)

        os.mkdir(f"./logs/{os.path.basename(args.pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}")
        log_dir = f"./logs/{os.path.basename(args.pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}"
        output_filename = os.path.join('results', f"{os.path.basename(args.pdf_path).split('.')[0]}_annotated_{datetime.now().strftime('%y%m%d%H%M%S')}.pdf")
        
        #embed
        os.mkdir(f"./logs/{os.path.basename(args.pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}")
        log_dir = f"./logs/{os.path.basename(args.pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}"
        output_filename = os.path.join('results', f"{os.path.basename(args.pdf_path).split('.')[0]}_annotated_{datetime.now().strftime('%y%m%d%H%M%S')}.pdf")

        print('Creating page indexes...')
        page_embeddings = Embedder.create_page_embeddings(args.pdf_path, show = args.show, query=args.query)
        print('Creating sentence indexes...')
        sent_embeddings, strings_list, sig_idx = Embedder.create_sent_embeddings(text_path)

        saved = [sig_idx, page_embeddings, sent_embeddings, strings_list, log_dir, output_filename]
        with open(os.path.join(log_dir, 'data.pkl'), 'wb') as file:
            pickle.dump(saved, file)

    aligned = Utils.embedNalign(sig_idx, page_embeddings, sent_embeddings, length_cutoff=10)
    result_json = Utils.get_result(strings_list, aligned)
    with open(os.path.join(log_dir, f'alignment_{args.query}.json'), 'w', encoding='utf-8') as f:
        f.write(result_json)
        print(f'json file sucessfully saved to {os.path.join(log_dir, f"alignment_{args.query}.json")}')

    df = pd.read_csv("./GT.csv")
    x_data = df.iloc[:, 0]  # First column (x-axis)
    y_data = df.iloc[:, 1]  # Second column (y-axis)
    y_data = y_data - np.ones_like(y_data)
    x_data = x_data - np.ones_like(x_data)
    
    plt.scatter(x_data, y_data, c='blue', label='GT')
    plt.scatter(sig_idx, [a[1] for a in aligned], c='green', label='Prediction')
    plt.savefig(os.path.join(log_dir, f'alignment_results.png'))

    #annot
    PDFAnnotator.add_summary(args.pdf_path, os.path.join(log_dir, f"alignment_{args.query}.json"), output_filename, log_dir, t = 'path')

if __name__ == '__main__':
    main()