#!/bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: $0 <book_title> <page_number> <text> <language>"
  exit 1
fi

# Create a directory for audio files if it doesn't exist
mkdir -p audio_files

# Capture the arguments
book_title="$1"
page_number="$2"
text="$3"
language="$4"

# If page number contains 'n/a', append timestamp to filename
if [[ "$page_number" == *"nochapter"* ]]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  page_number="${page_number}_${timestamp}"
  echo "Page number contains 'n/a'. Creating new audio file with timestamp."
fi

# Format the filename
filename="${book_title}_${page_number}_${language}.wav"

# Count the number of characters in the text
char_count=$(echo -n "$text" | wc -m)
echo "Character count: $char_count"

# Capture start time
start=$(date +%s.%N)

# Select the TTS model based on language
if [ "$language" == "swh" ]; then
  tts_model="sw_CD-lanfrica-medium.onnx"
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
elif [ "$langauge" == "spa" ]; then
  tts_model="es_MX-claude-high.onnx"
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
elif [ "$language" == "arz" ]; then
  tts_model="ar_JO-kareem-medium.onnx "
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
elif [ "$language" == "fra" ]; then
  tts_model="fr_FR-tom-medium.onnx"
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
elif [ "$language" == "yue" ]; then
  tts_model="zh_CN-huayan-medium.onnx"
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
elif [ "$language" == "deu" ]; then
  tts_model ="de_DE-thorsten-high.onnx"
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
else
  tts_model="en_US-lessac-medium"
  tts_command="piper --model $tts_model --output_file audio_files/$filename --cuda"
fi

# Generate speech using Piper
echo "$text" | $tts_command

# Capture end time
end=$(date +%s.%N)

# Calculate execution time
duration=$(echo "$end - $start" | bc)

# Print execution time
echo "Execution time: $duration seconds"
