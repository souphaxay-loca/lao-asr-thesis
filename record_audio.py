#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Recording Script for ASR Dataset Creation

Purpose:
Facilitates recording audio samples for an Automatic Speech Recognition (ASR)
dataset. Reads sentences from a CSV file (with 'sentence_id' and 'transcription'
headers), displays them one by one, and records audio from a selected
microphone, saving it with a structured filename.

Usage:
1. Install required libraries: pip install sounddevice soundfile numpy
2. Prepare your sentences CSV file (e.g., 'datatext.csv') with headers
   'sentence_id' and 'transcription'.
3. Place this CSV file in a 'data_source' sub-directory relative to the script.
4. Create a 'recordings' directory (the script will create subfolders inside).
5. Run the script from the terminal, providing speaker and accent info:
   python record_audio.py --speaker_id <YourSpeakerID> --accent <AccentName> [--device <ID>]
   Example: python record_audio.py --speaker_id C_SPK01 --accent central --sentences_file data_source/datatext.csv

Arguments:
  --speaker_id : Unique identifier for the speaker (e.g., C_SPK01). (Required)
  --accent     : Accent category name (e.g., central, northern, southern). (Required)
  --sentences_file : Path to the CSV file containing sentences to read.
                 (Default: data_source/datatext.csv)
  --list_devices : List available audio input devices and exit.
  --device       : Numeric ID of the audio input device to use (optional).
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import argparse
import sys
import queue
import time
import csv

# --- Configuration ---
SAMPLE_RATE = 16000  # Hertz (samples per second), standard for ASR
CHANNELS = 1         # Mono audio
AUDIO_DTYPE = 'int16' # Data type for 16-bit audio
RECORDINGS_BASE_DIR = 'recordings' # Base directory to save recordings
# Default assumes your CSV is named 'datatext.csv' inside a 'data_source' folder
DEFAULT_SENTENCES_FILE = os.path.join('data_source', 'datatext.csv')

# --- Audio Queue (used by sounddevice callback) ---
# Using a queue is the standard way to pass data from the audio callback thread
audio_queue = queue.Queue()

# --- Audio Callback Function ---
def audio_callback(indata, frames, time, status):
    """
    This function is called by sounddevice from a separate thread for each
    incoming audio block ('indata').
    """
    if status:
        # Report any issues encountered by the audio stream
        print(f"Audio Stream Status Warning: {status}", file=sys.stderr)
    # Add a copy of the audio data block to the queue
    audio_queue.put(indata.copy())

# --- Helper Functions ---
def load_sentences_from_csv(filepath):
    """
    Loads sentences to be read from a CSV file.
    Expects columns named 'sentence_id' and 'transcription'.
    Returns a dictionary mapping {sentence_id: sentence_text}.
    """
    sentences = {}
    required_headers = ['sentence_id', 'transcription']
    print(f"Attempting to load sentences from: {filepath}")
    try:
        # Use 'utf-8-sig' encoding to handle potential BOM (Byte Order Mark) from Excel CSVs
        with open(filepath, mode='r', encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # print(reader)

            # Verify required headers are present
            if not all(header in reader.fieldnames for header in required_headers):
                print(f"Error: CSV file '{filepath}' is missing required headers.", file=sys.stderr)
                print(f"Required headers: {required_headers}", file=sys.stderr)
                print(f"Found headers: {reader.fieldnames}", file=sys.stderr)
                sys.exit(1)

            # Read each row
            for i, row in enumerate(reader):
                # Use .get() for safety and .strip() to remove leading/trailing whitespace
                sentence_id = row.get('sentence_id', '').strip()
                text = row.get('transcription', '').strip()

                # Ensure both ID and text are present before adding
                if sentence_id and text:
                    if sentence_id in sentences:
                         print(f"Warning: Duplicate sentence_id '{sentence_id}' found at row {i+2}. Keeping first occurrence.", file=sys.stderr)
                    else:
                        sentences[sentence_id] = text
                elif not sentence_id and not text:
                    # Skip completely empty rows silently
                    pass
                else:
                    # Warn about rows with missing ID or text
                    print(f"Warning: Skipping row {i+2} due to missing sentence-id or transcription in CSV: {row}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Sentences CSV file not found at '{filepath}'", file=sys.stderr)
        print("Please ensure the file exists or provide the correct path using --sentences_file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or parsing sentences CSV file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    if not sentences:
        print(f"Error: No valid sentences loaded from '{filepath}'.", file=sys.stderr)
        print("Please check the file content, headers ('sentence_id', 'transcription'), and encoding (should be UTF-8).")
        sys.exit(1)

    print(f"Successfully loaded {len(sentences)} sentences.")
    return sentences

def list_audio_devices():
    """Lists available audio input devices with their IDs."""
    print("\nAvailable Audio Input Devices:")
    try:
        devices = sd.query_devices()
        input_devices_found = False
        for i, device in enumerate(devices):
            # Check if the device has input channels
            if device.get('max_input_channels', 0) > 0:
                try:
                    # Attempt to decode device name robustly
                    device_name = device.get('name', f'Device {i} (Unnamed)')
                except Exception:
                    device_name = f"Device {i} (Could not decode name)"

                print(f"  ID {i}: {device_name} (Max Input Channels: {device.get('max_input_channels', 'N/A')})")
                input_devices_found = True
        if not input_devices_found:
            print("  No audio input devices found or accessible.")
        print("-" * 30)
    except Exception as e:
        print(f"Error querying audio devices: {e}", file=sys.stderr)


# --- Main Execution Block ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Record audio sentences for an ASR dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter # Preserve formatting in help text
    )
    parser.add_argument('--speaker_id', required=True,
                        help="Unique identifier for the speaker (e.g., C_SPK01).")
    parser.add_argument('--accent', required=True,
                        help="Accent category name (e.g., central, northern, southern).")
    parser.add_argument('--sentences_file', default=DEFAULT_SENTENCES_FILE,
                        help=f"Path to the CSV file containing sentences to read (default: {DEFAULT_SENTENCES_FILE}).")
    parser.add_argument('--list_devices', action='store_true',
                        help="List available audio input devices and exit.")
    parser.add_argument('--device', type=int,
                        help="Numeric ID of the audio input device to use (optional, uses default if omitted).")

    args = parser.parse_args()

    # --- Handle --list_devices Action ---
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # --- Audio Device Setup ---
    selected_device_info = "Default System Input Device"
    if args.device is not None:
        try:
            device_info = sd.query_devices(args.device)
            # Check if selected device actually has input channels
            if device_info.get('max_input_channels', 0) > 0:
                 selected_device_info = f"ID {args.device}: {device_info.get('name', 'Unnamed')}"
                 # Set the default device for sounddevice operations
                 sd.default.device = args.device
            else:
                 print(f"Error: Device ID {args.device} ({device_info.get('name', 'Unnamed')}) does not appear to be an input device.", file=sys.stderr)
                 sys.exit(1)
        except Exception as e:
            print(f"Error accessing audio device ID {args.device}: {e}", file=sys.stderr)
            print("Please use --list_devices to see available IDs and ensure the device is connected.")
            sys.exit(1)
    print(f"Using audio input device: {selected_device_info}")

    # --- Directory Setup ---
    # Construct the path for saving recordings: recordings/accent/speaker_id/
    speaker_recording_dir = os.path.join(RECORDINGS_BASE_DIR, args.accent, args.speaker_id)
    try:
        # Create directories if they don't exist. exist_ok=True prevents errors if they already exist.
        os.makedirs(speaker_recording_dir, exist_ok=True)
        print(f"Recordings will be saved to: {speaker_recording_dir}")
    except OSError as e:
        print(f"Error creating directory '{speaker_recording_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load Sentences ---
    sentences = load_sentences_from_csv(args.sentences_file)
    # Get sentence IDs from the dictionary keys and sort them for consistent processing order
    # sentence_ids = sorted(sentences.keys())
    sentence_ids = sorted(sentences.keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(sentence_ids)
    total_sentences = len(sentence_ids)
    session_recorded_count = 0
    session_skipped_count = 0

    # --- Display Instructions ---
    print("\n--- Recording Session Start ---")
    print(f"Speaker: {args.speaker_id} | Accent: {args.accent}")
    print(f"Total sentences loaded: {total_sentences}")
    print("\nInstructions for the OPERATOR:")
    print("  1. The sentence to be read will be displayed.")
    print("  2. Ensure the speaker is ready.")
    print("  3. Press ENTER to start recording.")
    print("  4. After the speaker finishes reading, press ENTER again to stop.")
    print("  5. To quit the session early, type 'q' and press Enter instead of starting a recording.")
    print("\nInstructions for the SPEAKER:")
    print("  1. Wait for the 'RECORDING NOW' message.")
    print("  2. Read the displayed sentence clearly and at a natural pace.")
    print("  3. Wait for the 'Recording stopped' message before the next sentence.")
    print("-" * 40)
    time.sleep(3) # Pause for instructions to be read

    # --- Main Recording Loop ---
    current_stream = None # Keep track of the current audio stream
    try:
        for i, sentence_id in enumerate(sentence_ids):
            # Get the sentence text from the dictionary using the ID
            sentence_text = sentences[sentence_id]
            # Define the output filename: SpeakerID_SentenceID.wav
            output_filename = f"{args.speaker_id}_{sentence_id}.wav"
            output_filepath = os.path.join(speaker_recording_dir, output_filename)

            print(f"\n[{i+1}/{total_sentences}] Sentence ID: {sentence_id}")
            print(f"  Text to Read: '{sentence_text}'")

            # Check if this sentence has already been recorded in a previous session
            if os.path.exists(output_filepath):
                print(f"  --> Already recorded as '{output_filename}'. Skipping.")
                session_skipped_count += 1
                continue

            # Prompt operator to start or quit
            operator_action = input("Press ENTER to start recording, or type 'q' to quit session: ")
            if operator_action.lower().strip() == 'q':
                print("Quit command received. Ending recording session.")
                break

            # --- Start Recording Process ---
            # Clear the audio queue before starting a new recording
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Create and start the audio input stream
            current_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=AUDIO_DTYPE,
                callback=audio_callback,
                # device=args.device # sd.default should be set already if --device was used
            )
            current_stream.start()
            print("\n--- 🟡RECORDING NOW 🟡 ---")
            print("   (Operator: Press ENTER when speaker has finished reading)")

            # Wait for operator to press Enter to signal end of speaking
            input() # This blocks until Enter is pressed

            # --- Stop Recording Process ---
            print("--- Recording stopped. Processing audio... ---")
            current_stream.stop()
            current_stream.close()
            current_stream = None # Clear the stream variable

            # Retrieve all recorded audio data chunks from the queue
            recording_chunks = []
            while not audio_queue.empty():
                try:
                    recording_chunks.append(audio_queue.get_nowait())
                except queue.Empty:
                    break

            if not recording_chunks:
                 print("Warning: No audio data was captured for this sentence. Please check microphone.", file=sys.stderr)
                 print("Skipping save for this sentence.")
                 continue # Go to the next sentence

            # Combine the chunks into a single NumPy array
            audio_data = np.concatenate(recording_chunks, axis=0)

            # --- Save the Audio File ---
            try:
                # Save as WAV file, explicitly specifying 16-bit PCM subtype
                sf.write(output_filepath, audio_data, SAMPLE_RATE, subtype='PCM_16')
                duration_seconds = len(audio_data) / SAMPLE_RATE
                print(f"--> Successfully saved: '{output_filename}' ({duration_seconds:.2f} seconds)")
                session_recorded_count += 1
                time.sleep(0.5) # Brief pause before the next prompt

            except Exception as e:
                 print(f"Error saving audio file '{output_filepath}': {e}", file=sys.stderr)
                 print("Please check file permissions and disk space.")
                 # Decide if you want to stop the whole session on a save error
                 break # Stop the session for safety

    except sd.PortAudioError as e:
        print(f"\nFatal Audio Device Error: {e}", file=sys.stderr)
        print("Please check microphone connection, system audio settings, and permissions.")
        if current_stream:
            try: current_stream.close() # Attempt to close stream if open
            except Exception: pass
        # Offer to list devices again might be helpful here
        list_audio_devices()
        print("Exiting script due to audio error.")
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nSession interrupted by user (Ctrl+C). Exiting.")
        if current_stream:
            try: current_stream.close()
            except Exception: pass
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        if current_stream:
            try: current_stream.close()
            except Exception: pass
        # Consider adding more detailed error reporting here if needed
    finally:
        # --- Session Summary ---
        # This block executes whether the loop finished normally or was interrupted
        print("\n--- Recording Session Summary ---")
        print(f"Speaker: {args.speaker_id} | Accent: {args.accent}")
        print(f"Total Sentences in File: {total_sentences}")
        print(f"Sentences Recorded in this Session: {session_recorded_count}")
        print(f"Sentences Skipped (already existed): {session_skipped_count}")
        # Calculate remaining accurately, considering possibility session ended early
        recorded_ids = set(os.path.splitext(f)[0] for f in os.listdir(speaker_recording_dir) if f.startswith(args.speaker_id) and f.endswith('.wav'))
        expected_ids = set(f"{args.speaker_id}_{sid}" for sid in sentence_ids)
        actually_recorded_count = len(recorded_ids.intersection(expected_ids))
        remaining = total_sentences - actually_recorded_count
        print(f"Total Sentences Recorded for this Speaker (Cumulative): {actually_recorded_count}")
        print(f"Sentences Remaining in File: {remaining}")
        print("-" * 40)

# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure the main function is called only when the script is executed directly
    main()
