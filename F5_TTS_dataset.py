from os import listdir, makedirs
from os.path import exists

import f5_tts


def get_f5_tts_dataset(words: str):
    """
    Fetches the categories of words in new_dataset and returns the cloned audio files.
    """
    # Initialize the F5 TTS dataset
    f5_tts_dataset = f5_tts.F5TTS()
    if not exists(f5_tts_dataset + words):
        makedirs(f5_tts_dataset + words)

    # Load the dataset
    dataset_path = "../new_dataset/"
    model = "F5TTS_v1_Base"
    dataset_path_words = dataset_path + "/" + words
    dataset_words = listdir(dataset_path_words)

    for audio in dataset_words:
        ref_audio = audio
        gen_audio = f5_tts_dataset.generate_audio(ref_audio, model=model)
        output_path = "../f5_tts_dataset/" + words + "/" + "gen_" + audio
        f5_tts_dataset.save_audio(gen_audio, output_path)
        print(f"Generated audio saved to {output_path}")
    
