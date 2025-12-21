import os
import time
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import Queue, Manager
import src.ui_manager.PySimpleGUI as sg

from src.logic.image_logic.image_manager import manipulate_image

from src.logic.vid_logic import ffmpeg_manager
from src.path_manager.pather import resource_path
import logging

logger = logging.getLogger(__name__)

assets_path = resource_path(f"./assets/cache/")

# The cache folders
vid_cache_folder_jpg = os.path.normpath(os.path.join(assets_path, "./img_cache_jpg/"))
vid_cache_folder_png = os.path.normpath(os.path.join(assets_path, "./img_cache_png/"))
vid_cache_folder_m4a = os.path.normpath(os.path.join(assets_path, "./audio_cache/"))

vid_processed_folder = os.path.normpath(os.path.join(assets_path, "./img_process_cache/"))

# Create the paths if they don't exist (Git does not recognize empty folders)
if not os.path.exists(vid_cache_folder_jpg):
    os.makedirs(vid_cache_folder_jpg)

if not os.path.exists(vid_cache_folder_png):
    os.makedirs(vid_cache_folder_png)

if not os.path.exists(vid_cache_folder_m4a):
    os.makedirs(vid_cache_folder_m4a)

if not os.path.exists(vid_processed_folder):
    os.makedirs(vid_processed_folder)

THREAD_KEY = '-Vid_Thread-'


def vid_manager(window: sg.Window, filepath: str, output: str, manipulation: str, scale: str, details: dict):
    # Cleaning cache folders, incase a previous run failed to do so
    cleanup_folders()
    logger.info("Cleaned Previous Cache")
    logger.debug("Video Details:")
    logger.debug("FilePath: " + filepath)
    logger.debug("Output Path: " + output)
    logger.debug("Manupilations: " + manipulation)
    logger.debug("Scale: " + scale)
    logger.debug(f"Details: {details}")

    # Instantiating the process/thread related objects
    process_count = details['process_count']
    process_pool = Pool(processes=process_count)
    ff_pool = ThreadPool(processes=2)
    manager = Manager()
    event_queue = manager.Queue()

    frame_count = ffmpeg_manager.get_frame_count(filepath, details['frame_rate'])
    # Close enough frame count
    window.write_event_value((THREAD_KEY, '-Image_Count-'), frame_count)

    # Selecting the correct cache folder
    cache_folder = vid_cache_folder_png if details['quality'] else vid_cache_folder_jpg

    # Running the encoder?(In another thread)
    ff_pool.apply_async(
        ffmpeg_manager.vid_to_img, (filepath, details['frame_rate'], cache_folder),
        callback=lambda x: print(f"Test {x}")
    )
    ff_pool.apply_async(
        ffmpeg_manager.vid_to_audio, (filepath,)
    )
    ff_pool.close()
    # Starting the image conversion, till at least 95% the images are there (Not all, because we don't get exact frame
    # count with the get_frame_count
    current_image = 0
    image_processes = []
    img_count = 0
    while True:
        # Checking the number of files that have been outputted by ffmpeg
        files = os.listdir(cache_folder)
        file_count = len(files)
        if file_count >= (frame_count * 0.95):
            break
        # Updating the progress meter
        window.write_event_value((THREAD_KEY, '-Img_Conversion-'), file_count / frame_count)

        # Starting the processing of images to blocks, while ffmpeg is generating those images, to save time
        if file_count > current_image:
            img_file_path = os.path.join(cache_folder, files[current_image])
            # Remove the first 4 and last 4 characters, to get just the number
            file_number = os.path.split(img_file_path)[1][4:-4]
            processed_path = os.path.join(vid_processed_folder, file_number + ".png")
            image_processes.append(process_pool.apply_async(
                manage_single_image,
                (event_queue, img_file_path, processed_path, manipulation, scale, details)
            ))
            current_image += 1

        # Item count checking frequency
        time.sleep(0.1)
    ff_pool.join()
    logger.info("Converted Video to images")

    # Run the already processing images and show their status
    for process in image_processes:
        process.wait()
        window.write_event_value((THREAD_KEY, '-Image_Done-'), None)
        img_count += 1

    # Update images done
    window.write_event_value((THREAD_KEY, '-Set_Images_Done-'), img_count)
    # Video has now been converted to images and audio
    window.write_event_value((THREAD_KEY, '-Img_Conversion-'), 1)
    logger.debug("Completed partial processing")

    # Getting the correct item count and finding which files we have to process now
    files = os.listdir(cache_folder)
    file_count = len(files)
    completed_files = os.listdir(vid_processed_folder)
    for completed_file in completed_files:
        file_no = completed_file.replace(".png", "")
        name_of_file = "file" + file_no + (".jpg" if cache_folder == vid_cache_folder_jpg else ".png")
        files.remove(name_of_file)

    window.write_event_value((THREAD_KEY, '-Image_Count-'), file_count)

    # Processing the rest of the images
    for unprocessed_file in files:
        img_file_path = os.path.join(cache_folder, unprocessed_file)
        file_number = os.path.split(img_file_path)[1][4:-4]
        processed_path = os.path.join(vid_processed_folder, file_number + ".png")
        image_processes.append(process_pool.apply_async(
            manage_single_image,
            (event_queue, img_file_path, processed_path, manipulation, scale, details)
        ))
        current_image += 1

    process_pool.close()
    iter_count = 0
    while True:
        # Updating the progress meters, till the all the processing has been done
        try:
            # Use timeout to prevent indefinite blocking
            data = event_queue.get(timeout=0.5)
            if data[0] == "-Single_Frame-":
                window['-Single_Frame-'](data[1])
        except Exception as queue_err:
            # Queue timeout or error — check if all processes are done
            logger.debug(f"Queue timeout/error: {queue_err}")
            data = None

        # Check if all processes are ready (finished)
        all_ready = all([proc.ready() for proc in image_processes])
        
        if all_ready:
            logger.info(f"All {len(image_processes)} processes are ready/complete")
            break

        iter_count += 1

        # Update the images done progress bar, as event-based is too slow
        if iter_count % 10 == 0:
            processed_file_count = len(os.listdir(vid_processed_folder))
            progress = processed_file_count / file_count * 100
            window['-Number_Of_Frames-'](progress)
            window['-Number_Of_Frames_Text-'](f"{processed_file_count}/{file_count}")

        # Check for worker failures
        if any([not proc.successful() for proc in image_processes if proc.ready()]):
            logger.error("ERROR: One or more worker processes failed!")
            proc_list = [not proc.successful() for proc in image_processes if proc.ready()]
            indices = [i for i, x in enumerate(proc_list) if x]
            for index in indices:
                try:
                    logger.error(image_processes[index].get())
                except Exception as e:
                    logger.exception(f"Failed to get result from process {index}: {e}")
            break

    process_pool.join()
    logger.info("Process pool joined successfully")

    # If the processing was very fast, and happened before we caught up with the queue
    # updating the progress meters indicating processing done
    window['-Single_Frame-'](100)
    for i in range(img_count, frame_count):
        window.write_event_value((THREAD_KEY, '-Image_Done-'), None)

    window.write_event_value((THREAD_KEY, '-Img_Conversion-'), 1.4)
    logger.info("Completed Image Processing")
    
    # List processed frames for debugging
    processed_files = os.listdir(vid_processed_folder)
    logger.info(f"Total processed frames: {len(processed_files)}")
    
    # Creating the ffmpeg run command
    if len(os.listdir(vid_cache_folder_m4a)) > 0:
        re_options = (
            f"-i {os.path.join(vid_cache_folder_m4a, 'audio.m4a')} "
            f"-r {details['frame_rate']} "
            f"-i {os.path.join(vid_processed_folder, '%01d.png')} "
            f"-crf 20 -pix_fmt yuv420p {output}"
        )
    else:
        re_options = (
            f"-r {details['frame_rate']} -i {os.path.join(vid_processed_folder, '%01d.png')} "
            f"-crf 20 -pix_fmt yuv420p {output}"
        )
    logger.debug("The ffmpeg converted join command: " + re_options)
    logger.info("Starting FFmpeg re-encoding...")
    try:
        ffmpeg_manager.ffmpeg_runner(re_options)
        logger.info("FFmpeg re-encoding completed successfully")
    except Exception as e:
        logger.exception(f"FFmpeg re-encoding failed: {e}")
        raise
    window.write_event_value((THREAD_KEY, '-Img_Conversion-'), 1.8)
    logger.info("Video Created")
    # Cleaning up the cache folders
    cleanup_folders()
    logger.info("Cleaning Cache")
    window.write_event_value((THREAD_KEY, '-Img_Conversion-'), 2)
    logger.info("Video Conversion Completed!")


# Cleans up cache
def cleanup_folders():
    cache_png_files = os.listdir(vid_cache_folder_png)
    for file in cache_png_files:
        os.remove(os.path.join(vid_cache_folder_png, file))

    cache_jpg_files = os.listdir(vid_cache_folder_jpg)
    for file in cache_jpg_files:
        os.remove(os.path.join(vid_cache_folder_jpg, file))

    cache_processed_files = os.listdir(vid_processed_folder)
    for file in cache_processed_files:
        os.remove(os.path.join(vid_processed_folder, file))

    audio_processed_files = os.listdir(vid_cache_folder_m4a)
    for file in audio_processed_files:
        os.remove(os.path.join(vid_cache_folder_m4a, file))


# Running this for every single frame
def manage_single_image(event_queue: Queue, filename: str, output: str, manipulation: str, scale: str, details: dict):
    manipulation = manipulation.replace("Video", "Image")
    iteration = 0
    image_size = 1
    updates = 30
    # Using the same "image to blocks" conversion as in the image tab
    for values in manipulate_image(filename, output, manipulation, None, scale, details):
        if iteration == 0:
            image_size = values
        iteration += 1
        # Updating the progress every certain amount of iterations
        if iteration % round(image_size / updates) == 0:
            if isinstance(values, str):
                pass
            else:
                event_queue.put(['-Single_Frame-', values / image_size * 100])
    event_queue.put(['-Image_Done-', output])
