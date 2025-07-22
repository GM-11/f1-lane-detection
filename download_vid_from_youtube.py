import os
import yt_dlp


def download_youtube_video(url: str, custom_name: str, output_folder: str = "./data"):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{custom_name}.mp4")
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Downloaded: {output_path}")


url = "https://www.youtube.com/watch?v=Zjt72i2PEEE"
file_name = "spa_onboard"
download_youtube_video(url, file_name)
