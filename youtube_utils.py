import yt_dlp

def get_playlist_info_md(playlist_url: str):
    """
    Scraps youtube playlist video titles
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
    }

    md_lines = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            playlist_title = info.get('title', 'Unknown Playlist')
            md_lines.append(f"# {playlist_title}")

            for entry in info.get('entries', []):
                title = entry.get('title', 'Untitled')
                vid = entry.get('id')
                if vid:
                    url = f"https://www.youtube.com/watch?v={vid}"
                    md_lines.append(f"- [{title}]({url})")

            # print markdown lines
            print("\n".join(md_lines))
            return md_lines

        except Exception as e:
            print(f"An error occurred: {e}")
            return []
