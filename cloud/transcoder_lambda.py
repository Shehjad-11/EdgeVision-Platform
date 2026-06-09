# cloud/transcoder_lambda.py
import os
import subprocess
import urllib.parse
from typing import Dict, Any

# Safe boto3 import checks
BOTO3_AVAILABLE = False
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    pass

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler triggered by S3 bucket ObjectCreated events.
    Downloads the raw edge-recorded MP4 segment, transcodes the stream
    into browser-compatible H.264 video / AAC audio, chunks it into
    HLS segments (.ts files), and uploads the output to a CDN-backed S3 bucket.
    """
    if not BOTO3_AVAILABLE:
        print("[Lambda Mock] Running in simulated environment (boto3 missing).")
        return {
            "status": "MOCK_COMPLETED",
            "hls_stream_url": "s3://edgevision-prod-clips/streams/cam-wh-east-04/mock_stream.m3u8"
        }

    s3_client = boto3.client('s3')

    # Parse bucket location and key details from triggering event
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    raw_key = event['Records'][0]['s3']['object']['key']
    key = urllib.parse.unquote_plus(raw_key)

    # Establish clean local scratch spaces inside Lambda container
    filename = os.path.basename(key)
    temp_input_path = f"/tmp/{filename}"
    temp_output_dir = "/tmp/hls_transcode_out"
    os.makedirs(temp_output_dir, exist_ok=True)

    print(f"[Lambda] Downloading raw clip from S3: s3://{bucket_name}/{key}")
    s3_client.download_file(bucket_name, key, temp_input_path)

    # Execute FFmpeg binary via subprocess (usually loaded via AWS Lambda layer)
    # Re-encodes H.264/AAC to ensure absolute browser compatibility (e.g. Safari / Chrome mobile)
    # Segments video into rolling 4-second .ts chunks for low-latency player loads
    hls_output_playlist = os.path.join(temp_output_dir, "playlist.m3u8")
    ffmpeg_cmd = [
        "/opt/ffmpeg/ffmpeg", "-y",
        "-i", temp_input_path,
        "-c:v", "libx264", "-profile:v", "main", "-crf", "23",
        "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
        "-f", "hls", "-hls_time", "4", "-hls_playlist_type", "vod",
        "-hls_segment_filename", f"{temp_output_dir}/segment_%03d.ts",
        hls_output_playlist
    ]

    print(f"[Lambda] Executing FFmpeg HLS transcoding command: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as err:
        error_msg = err.stderr.decode('utf-8') if err.stderr else str(err)
        print(f"[Lambda FFmpeg Error] Transcoding failed: {error_msg}")
        raise RuntimeError(f"FFmpeg transcode failed: {error_msg}")

    # Upload HLS stream artifacts back to the cloud storage bucket
    out_prefix = f"streams/{os.path.splitext(filename)[0]}"
    print(f"[Lambda] Uploading HLS artifacts to S3 folder: s3://{bucket_name}/{out_prefix}/")
    
    for root, dirs, files in os.walk(temp_output_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = f"{out_prefix}/{file}"
            
            # Map correct mime types to prevent browser rendering download issues
            content_type = "application/x-mpegURL" if file.endswith('.m3u8') else "video/MP2T"
            
            s3_client.upload_file(
                local_file_path,
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )

    # Clean up local scratch files from Lambda storage allocation
    try:
        os.remove(temp_input_path)
        for file in os.listdir(temp_output_dir):
            os.remove(os.path.join(temp_output_dir, file))
        os.rmdir(temp_output_dir)
    except Exception as cleanup_err:
        print(f"[Lambda Warning] Clean temporary space files failed: {cleanup_err}")

    hls_url = f"s3://{bucket_name}/{out_prefix}/playlist.m3u8"
    print(f"[Lambda] Transcoding stream completed. Target playlist: {hls_url}")
    
    return {
        "status": "COMPLETED",
        "hls_stream_url": hls_url
    }
