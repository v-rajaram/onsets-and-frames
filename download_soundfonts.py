import requests
from pathlib import Path


def download_soundfont(url):
    name = url.split('/')[-1]
    save_dir = Path('data/soundfonts')
    save_path = save_dir / name

    if save_path.exists():
        print(f"Soundfont {save_path} are already downloaded.")
        return

    print(f"Downloading sondfount from url: {url}")
    r = requests.get(url, allow_redirects=True)

    save_dir.mkdir(parents= True, exist_ok=True)
    open(save_path, 'wb').write(r.content)


download_soundfont(url = 'https://musical-artifacts.com/artifacts/841/GMGSx.SF2')
download_soundfont(url = 'https://gifx.co/soundfonts/masquerade55v006.sf2')
download_soundfont(url = 'https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3')
download_soundfont(url = 'https://github.com/craffel/pretty-midi/raw/master/pretty_midi/TimGM6mb.sf2')
