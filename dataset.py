from os import listdir, makedirs
from os.path import exists, join
from typing import List

from scipy.io import wavfile


def fetch(word1: str, num: int = 100):
    """
    Function to fetch audio files from the dataset and save them to a new directory
        Parameters:
                - word1: First word category to fetch
                - word2: Second word category to fetch
                - word3: Third word category to fetch
                - word4: Fourth word category to fetch
                - num: Number of audio files to fetch from each category (default is 100)
    """
    dataset_path = "../dataset/"

    # Create new dataset directories if they do not exist
    new_dataset_path = "../new_dataset/"
    if not exists(new_dataset_path + word1):
        makedirs(new_dataset_path + word1)

    # Fetch audio files from a certain word category
    for i in range(num):
        # -------------------  word  -------------------
        # dataset_path = "../dataset/word1"
        path1 = dataset_path + word1
        audio_file = listdir(path1)

        sample_rate, data = wavfile.read(path1 + "/" + audio_file[i])
        output_path1 = join("../new_dataset/" + word1 + "/", "new_" + audio_file[i])

        # Save the audio file to the new dataset directory
        wavfile.write(output_path1, sample_rate, data)

fetch("dog", 10) # Example usage to fetch audio files for the word "dog"


def fetch_many(words: List[str], num: int = 100):
    """
    Function to fetch audio files from multiple word categories and save them to a new directory
        Parameters:
                - words: List of word categories to fetch
                - num: Number of audio files to fetch from each category (default is 100)
    """
    dataset_path = "../dataset/"
    new_dataset_path = "../new_dataset/"

    for word in words:
        # Create new dataset directory if it does not exist
        if not exists(new_dataset_path + word):
            makedirs(new_dataset_path + word)

        # Fetch audio files from the word category
        path = dataset_path + word
        audio_files = listdir(path)

        for i in range(min(num, len(audio_files))):
            sample_rate, data = wavfile.read(join(path, audio_files[i]))
            output_path = join(new_dataset_path + word, "new_" + audio_files[i])

            # Save the audio file to the new dataset directory
            wavfile.write(output_path, sample_rate, data)


# Example usage to fetch audio files for multiple words
List_words = ["down", "dog", "tree", "tree"]
fetch_many(List_words, 100)  # Fetch audio files for "cat", "dog", and "bird"
