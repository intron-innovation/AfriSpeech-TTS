import argparse
import os
import multiprocessing
from multiprocessing import Process
import text
from utils import load_filepaths_and_text
import logging
logging.disable(logging.CRITICAL)

def clean_text(filepaths_and_text_, filelist, text_index, 
               out_extension, text_cleaners, cpu_proc):
    
    sentence_list = [filepaths_and_text_[i][text_index] for i in range(len(filepaths_and_text_))]
    
    cleaned_texts = text._clean_text(sentence_list, text_cleaners)
    
    # print(cpu_proc, ": ", len(cleaned_texts), len(sentence_list))
    assert len(cleaned_texts) == len(sentence_list), f"proc {cpu_proc} had a length error."
    
    for i in range(len(filepaths_and_text_)):
        filepaths_and_text_[i][text_index] = cleaned_texts[i]

    new_filelist = filelist + "." + out_extension + "." + str(cpu_proc)
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in filepaths_and_text_])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "filelists/ljs_audio_text_val_filelist.txt",
            "filelists/ljs_audio_text_test_filelist.txt",
        ],
    )
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    j = 0
    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)

        # doing this to avoid running out of memory
        
        num_cpu = multiprocessing.cpu_count()
        print(f"{num_cpu} cpu(s) available for parallel computation")

        # create shared variable
        # speaker_index = Value('L', 9999)
        # avg_dist = Value('d', -999.0)
        
        # start mp computation
        print(f"Preprocessing a total of {len(filepaths_and_text)} files...")
        processes = []
        total_splits = len(filepaths_and_text)//num_cpu

        for i in range(num_cpu-1):
            temp_filepaths_and_text = filepaths_and_text[total_splits*i: total_splits*i+total_splits]
            
            # create a child process process
            process = Process(target=clean_text, args=(temp_filepaths_and_text, filelist, args.text_index, 
                                                       args.out_extension, args.text_cleaners, i))
            
            processes.append(process)
            process.start()
        
        final_split = total_splits*(num_cpu-1)
        temp_filepaths_and_text = filepaths_and_text[final_split: ]
        process = Process(target=clean_text, args=(temp_filepaths_and_text, filelist, args.text_index, 
                                                   args.out_extension, args.text_cleaners, num_cpu-1))
                
        processes.append(process)
        process.start()
        
        # complete the processes
        for proc in processes:
            proc.join()
        
        cleaned_filepaths_and_text = []
        new_filelist = filelist + "." + args.out_extension
        for i in range(num_cpu):
            temp_filelist = filelist + "." + args.out_extension + "." + str(i)
            if os.path.isfile(temp_filelist):
                cleaned_filepaths_and_text += load_filepaths_and_text(temp_filelist)
                os.remove(temp_filelist)
            else:
                print(f"file for process {i} was not found")
            
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in cleaned_filepaths_and_text])
