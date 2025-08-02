import os
import yt_dlp


def download_youtube_video(url: str, custom_name: str, output_folder: str = "./data"):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{custom_name}.mp4")

    ydl_opts = {
        # Strict format filter: H.264 video (avc1) + AAC audio (m4a)
        "format": (
            "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
            "best[ext=mp4][vcodec^=avc1]"
        ),
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"✅ Downloaded (H.264 only): {output_path}")


# 🧪 Example batch usage:
if __name__ == "__main__":
    video_list = [
        ("https://www.youtube.com/watch?v=_5Lr6fDIZG8", "lando_monaco_lap"),
    ]

    for url, name in video_list:
        download_youtube_video(url, name)
