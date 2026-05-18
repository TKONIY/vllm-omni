#!/usr/bin/env bash
# Streaming chat-completions client for the PD-disaggregated Qwen3-Omni
# server (paired with run_server_disagg.sh / run_all_stages_disagg.sh).
#
# Sends `"stream": true` and prints SSE deltas as they arrive. Use this to
# verify end-to-end token streaming through the four-stage PD pipeline:
# tokens should start appearing immediately after the decode stage emits
# the first text chunk — without waiting for talker / code2wav to finish.
#
# Usage:
#   ./run_curl_streaming_disagg.sh [text|use_audio|use_image|use_video] [modalities-json]
#
# Examples:
#   ./run_curl_streaming_disagg.sh                       # text-only (default)
#   ./run_curl_streaming_disagg.sh text '["text"]'       # text modality only
#   ./run_curl_streaming_disagg.sh use_image
#   ./run_curl_streaming_disagg.sh use_video '["text","audio"]'
#
# Env vars:
#   HOST          API server host (default: localhost)
#   PORT          API server port (default: 8091)
#   MODEL         Model id (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   RAW           If set to 1, print raw SSE frames instead of parsing deltas

set -euo pipefail

QUERY_TYPE="${1:-text}"
MODALITIES="${2:-null}"

if [[ ! "$QUERY_TYPE" =~ ^(text|use_audio|use_image|use_video)$ ]]; then
    echo "Error: invalid query type '$QUERY_TYPE'" >&2
    echo "Usage: $0 [text|use_audio|use_image|use_video] [modalities-json]" >&2
    exit 1
fi

HOST="${HOST:-localhost}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
RAW="${RAW:-0}"

MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
CHERRY_BLOSSOM_IMAGE_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
SAMPLE_VIDEO_URL="https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

thinker_sampling_params='{
  "temperature": 0.4,
  "top_p": 0.9,
  "top_k": 1,
  "max_tokens": 1024,
  "seed": 42,
  "repetition_penalty": 1.05,
  "stop_token_ids": [151645]
}'
talker_sampling_params='{
  "temperature": 0.9,
  "top_k": 50,
  "max_tokens": 4096,
  "seed": 42,
  "detokenize": false,
  "repetition_penalty": 1.05,
  "stop_token_ids": [2150]
}'
code2wav_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 65536,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.1
}'

case "$QUERY_TYPE" in
  text)
    user_content='[{"type":"text","text":"Explain the system architecture for a scalable audio generation pipeline. Answer in 30 words."}]'
    ;;
  use_audio)
    user_content='[
      {"type":"audio_url","audio_url":{"url":"'"$MARY_HAD_LAMB_AUDIO_URL"'"}},
      {"type":"text","text":"What is the content of this audio?"}
    ]'
    ;;
  use_image)
    user_content='[
      {"type":"image_url","image_url":{"url":"'"$CHERRY_BLOSSOM_IMAGE_URL"'"}},
      {"type":"text","text":"What is the content of this image?"}
    ]'
    ;;
  use_video)
    user_content='[
      {"type":"video_url","video_url":{"url":"'"$SAMPLE_VIDEO_URL"'"}},
      {"type":"text","text":"Why is this video interesting?"}
    ]'
    ;;
esac

sampling_params_list='['"$thinker_sampling_params"','"$talker_sampling_params"','"$code2wav_sampling_params"']'

request_body=$(cat <<EOF
{
  "model": "${MODEL}",
  "stream": true,
  "stream_options": {"include_usage": true},
  "sampling_params_list": ${sampling_params_list},
  "mm_processor_kwargs": {},
  "modalities": ${MODALITIES},
  "messages": [
    {
      "role": "system",
      "content": [{"type":"text","text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
    },
    {
      "role": "user",
      "content": ${user_content}
    }
  ]
}
EOF
)

URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "=========================================="
echo "PD-Disagg streaming client"
echo "=========================================="
echo "URL:        ${URL}"
echo "Query type: ${QUERY_TYPE}"
echo "Modalities: ${MODALITIES}"
echo "Model:      ${MODEL}"
echo "=========================================="
echo ""

# -N disables curl output buffering so each SSE frame is flushed promptly.
# --no-buffer is the curl alias for -N (works on older curl versions too).
if [[ "$RAW" == "1" ]]; then
    exec curl -sS -N --no-buffer --retry 3 --retry-delay 2 --retry-connrefused \
        -X POST "$URL" \
        -H "Content-Type: application/json" \
        -H "Accept: text/event-stream" \
        -d "$request_body"
fi

# Pretty path: parse `data: {json}` SSE frames, print text deltas inline,
# annotate role / finish_reason / usage events.
curl -sS -N --no-buffer --retry 3 --retry-delay 2 --retry-connrefused \
    -X POST "$URL" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d "$request_body" \
| awk '
    /^data: \[DONE\]/ { print "\n\n[done]"; next }
    /^data: /        { sub(/^data: /, ""); print }
    /^$/             { next }
' | while IFS= read -r line; do
    # Try a few common delta shapes. jq exits non-zero on a missing path
    # so we use `// empty` fallbacks and discard empties.
    content=$(echo "$line" | jq -r '.choices[0].delta.content // empty' 2>/dev/null || true)
    audio=$(echo "$line"   | jq -r '.choices[0].delta.audio.transcript // empty' 2>/dev/null || true)
    role=$(echo "$line"    | jq -r '.choices[0].delta.role // empty' 2>/dev/null || true)
    finish=$(echo "$line"  | jq -r '.choices[0].finish_reason // empty' 2>/dev/null || true)
    usage=$(echo "$line"   | jq -c '.usage // empty' 2>/dev/null || true)

    if [[ -n "$role" ]]; then
        printf "\n[role: %s] " "$role"
    fi
    if [[ -n "$content" ]]; then
        printf "%s" "$content"
    fi
    if [[ -n "$audio" ]]; then
        printf "%s" "$audio"
    fi
    if [[ -n "$finish" ]]; then
        printf "\n[finish_reason: %s]" "$finish"
    fi
    if [[ -n "$usage" ]]; then
        printf "\n[usage: %s]" "$usage"
    fi
    if [[ "$line" == "[done]" ]]; then
        echo ""
    fi
done
echo ""
