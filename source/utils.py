import time
import platform
import os
import subprocess
import tensorflow as tf

def print_gpu_info(run_on_gpu=True):
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        print("No GPUs available.")
    else:
        if not run_on_gpu:
            print("GPU is available, you are running on cpu instead of gpu.")
        print("Num GPUs Available: ", len(gpus))
        for i, gpu in enumerate(gpus):
            print(f"GPU {i + 1}: {gpu.name}, Memory: {gpu.memory_limit // (1024 * 1024)} MB")


def print_progress_bar(text, iteration, total, start_time=None, length=50):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + '-' * (length - filled_length)

    if start_time is not None:
        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)
        timer = f"{mins:02d}:{secs:02d}"
        progress_bar = f"{bar} | {percent}% Complete {iteration}/{total} images | Time: {timer}"
    else:
        progress_bar = f"{bar} | {percent}% Complete {iteration}/{total} images."

    print(f'\r{text}: {progress_bar}', end='', flush=True)
def open_folder(destination):
        print(f'Result saved to: {os.path.abspath(destination)}')
        if platform.system() == 'Windows':
            os.startfile(os.path.join(os.getcwd(), destination))
        elif platform.system() == 'Darwin' or platform.system() == 'Linux':
            subprocess.run(['open', os.path.join(os.getcwd(), destination)])
        else:
            print(f"Unsupported OS: {platform.system()}")
