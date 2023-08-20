import argparse
import json
import os
import subprocess
import sys
import zipfile
from collections import namedtuple
from datetime import datetime, timedelta
from functools import reduce
from random import uniform

import chardet
#import numpy as np
import requests
import srt
from environs import Env
from tqdm import tqdm
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)


def create_config():
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, '.conf')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as file:
            file.write('\n'.join([
                '#Uncomment one line to specify the language you want\n',
                'MODEL= vosk-model-small-en-us-0.15 # English',
                '# MODEL= vosk-model-ar-mgb2-0.4 # Arabic',
                '# MODEL= vosk-model-small-ca-0.4 # Catalan',
                '# MODEL= vosk-model-small-cn-0.22 # Chinese',
                '# MODEL= vosk-model-small-cs-0.4-rhasspy # Czech',
                '# MODEL= vosk-model-small-nl-0.22 # Dutch',
                '# MODEL= vosk-model-small-fa-0.4 # Farsi',
                '# MODEL= vosk-model-small-fr-0.22 # French',
                '# MODEL= vosk-model-small-de-0.15 # German',
                '# MODEL= vosk-model-small-hi-0.22 # Hindi',
                '# MODEL= vosk-model-small-it-0.22 # Italian',
                '# MODEL= vosk-model-small-ja-0.22 # Japanese',
                '# MODEL= vosk-model-small-kz-0.15 # Kazakh',
                '# MODEL= vosk-model-small-ko-0.22 # Korean',
                '# MODEL= vosk-model-small-pl-0.22 # Polish',
                '# MODEL= vosk-model-small-pt-0.3 # Portuguese',
                '# MODEL= vosk-model-small-ru-0.22 # Russian',
                '# MODEL= vosk-model-small-es-0.42 # Spanish',
                '# MODEL= vosk-model-small-sv-rhasspy-0.15 # Swedish',
                '# MODEL= vosk-model-small-tr-0.3 # Turkish',
                '# MODEL= vosk-model-small-uk-v3-nano # Ukrainian',
                '# MODEL= vosk-model-small-uz-0.22 # Uzbek',
                '# MODEL= vosk-model-small-vn-0.3 # Vietnamese',
            ]))


def download_model(model):
    url = f'https://alphacephei.com/vosk/models/{model}.zip'
    model_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        response = requests.get(url, stream=True)
        if requests.status_codes != 200:
            response.raise_for_status()
        with open('model.zip', 'wb') as f:
            total_length = int(response.headers.get('content-length'))
            pbar = tqdm(total=total_length, unit='B', unit_scale=True, desc='Downloading neural net for recognize')
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(len(data))
            pbar.close()
    except Exception as e:
        print('Ошибка при загрузке страницы: ' + str(e))
        return False
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(os.path.realpath(__file__)))
    return True


def is_right_language(sign, model_name):
    possible_signs = {'en': ['eng'],
                      'ar': ['ara'],
                      'ca': ['cat'],
                      'cn': ['chi', 'zho'],
                      'cs': ['ces', 'cze'],
                      'nl': ['dut', 'nld'],
                      'fa': ['fas', 'per'],
                      'fr': ['fra', 'fre'],
                      'de': ['deu', 'ger'],
                      'hi': ['hin'],
                      'it': ['ita'],
                      'ja': ['jpn'],
                      'kz': ['kaz'],
                      'ko': ['kor'],
                      'pl': ['pol'],
                      'pt': ['por'],
                      'ru': ['rus'],
                      'es': ['esl', 'spa'],
                      'sv': ['sve', 'swe'],
                      'tr': ['tur', 'tuk'],
                      'uk': ['ukr'],
                      'uz': ['uzb'],
                      'vn': ['vi', 'vie', ],
                      }
    # print(model_name)
    model_sign = [n for n in model_name.split('-') if len(n) == 2][0]
    return sign in possible_signs[model_sign]


def is_installed_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def is_internet_on():
    try:
        requests.get('http://www.google.com', timeout=1)
        return True
    except requests.ConnectionError:
        return False


def get_model_name():
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, '.conf')
    if not os.path.exists(config_path):
        create_config()
    env = Env()
    env.read_env(config_path)
    model = env.str('MODEL')
    model_path = os.path.join(script_path, model)
    if not os.path.exists(model_path):
        download_model(model)
    return model


def get_subtitles(filename):
    with open(filename, 'rb') as file:
        rawdata = file.read()
    encoding = chardet.detect(rawdata)['encoding']
    # print(encoding)
    subtitles = list(srt.parse(rawdata.decode(encoding)))
    if subtitles[-1].index - subtitles[-2].index > 1:
        subtitles.pop(-1)
    if 'www.' in subtitles[-1].content:
        subtitles.pop(-1)
    #enumerate for right index (if wrong source file)
    for i,s in  enumerate(subtitles):
        s.index=i+1

    return [{'index': s.index,
             'start': s.start.total_seconds(),
             'end': s.end.total_seconds(),
             'content': s.content,
             'correct': None}
            for s in subtitles]


def put_subtitles(filename, subt):
    if not os.path.isfile(f'{filename}.back'):
        os.rename(filename, f'{filename}.back')
    subtitles = []
    for i, s in enumerate(subt):
        start = timedelta(seconds=s['start'] - s['correct'])
        end = timedelta(seconds=s['end'] - s['correct'])
        subtitles.append(srt.Subtitle(index=i + 1, start=start, end=end, content=s['content']))
    with open(filename, 'wb') as file:
        file.write(srt.compose(subtitles).encode('utf-8'))


def get_subwords1(subs):
    Subs = namedtuple('Subs', 'num word subtitr start end pos')
    all_words = []
    n = 0
    for s in subs:
        normal_str = s['content'].lower().translate({ord(c): None for c in '.,?!-"'})
        normal_str = normal_str.replace('\n', ' ').strip().replace('  ', ' ')
        words = normal_str.split()
        for i, w in enumerate(words):
            all_words.append(Subs(n, w, s['index'] - 1, s['start'], s['end'], i))
            n += 1
    return all_words


def recognize(filename, model_path, metadata, audio_num=0, start=None, duration=None):
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.Reset()
    rec.SetWords(True)
    input_file = os.path.abspath(filename)
    command = ["ffmpeg", "-loglevel", "quiet",
               "-i", input_file,
               "-map", f"0:a:{audio_num}",
               "-ar", str(16000),
               "-ac", "1",
               "-ab", "2",
               # "-af", "pan=mono|c0=0.5*FL+0.5*FR",
               "-f", "s16le",
               "-acodec", "pcm_s16le",
               "-", ]
    if duration:
        command.insert(5, "-t")
        command.insert(6, str(duration))
        size_stream = 16000 * 2 * duration * 1.1
    else:
        size_stream = 16000 * 2 * metadata['duration'] * 1.1
    if start:
        command.insert(5, "-ss")
        command.insert(6, str(start))
    xxx = 0
    pbar = tqdm(total=size_stream, unit='B', unit_scale=True,
                desc=f'Recognize {"fragment of" if duration else ""} audiostream {audio_num}',
                bar_format='{desc}: {percentage:3.0f}% {bar:20}'
                )
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        while True:
            data = process.stdout.read(1310716)
            xxx += len(data)
            if len(data) == 0:
                break
            pbar.update(len(data))
            rec.AcceptWaveform(data)
    res = json.loads(rec.FinalResult())
    pbar.update(size_stream - pbar.n)
    pbar.close()
    Recs = namedtuple('Recs', 'num word start end conf')
    rec_words = []
    for n, r in enumerate(res['result']):
        rec_words.append(Recs(n, r['word'], r['start'], r['end'], r['conf']))
    return rec_words


def get_metadata(filename):
    input_file = os.path.abspath(filename)
    lines = str(subprocess.run(['ffmpeg', '-i', input_file, '-']
                               , capture_output=True
                               , shell=True
                               , encoding='utf-8')
                ).split('\\n')
    zero_date = datetime.strptime('00:00', '%H:%M')
    duration_str = [s for s in lines if ' Duration: ' in s][0].split(',')[0].split(':', 1)[1].strip()
    duration_date = (datetime.strptime(duration_str, "%H:%M:%S.%f"))
    duration = float((duration_date - zero_date).total_seconds())
    streams = [s.strip() for s in lines if ' Audio: ' in s]
    return {'duration': duration, 'streams': streams}


def find_native_stream(filename, model_name, metadata):
    ''' search for an audio stream number that is better recognized '''
    fragment_duration = 150
    if len(metadata['streams']) == 1:
        return 0
    for i, s in enumerate(metadata['streams']):
        sl = s[:s.index(': Audio')]
        if '(' in sl:
            lang = sl[sl.index('(') + 1:sl.index(')')]
            if is_right_language(lang, model_name):
                return i
    best_stream_and_rating = (0, 0)
    start_time = uniform(120, metadata['duration'] - fragment_duration)
    for i, s in enumerate(metadata['streams']):
        rec = recognize(filename, model_name, metadata, i, start_time, fragment_duration)
        exactly_recognuzed = reduce(lambda n, w: n if w.conf < 1 else n + 1, rec, 0.0)
        k = exactly_recognuzed / len(rec)
        # print(f'stream:{i}  words:{len(rec)}  K={k}')
        if k > best_stream_and_rating[1]:
            best_stream_and_rating = (i, k)
    print('best_stream=', best_stream_and_rating)
    return best_stream_and_rating[0]


def mkdict_recs(subs, recs):
    d = {w: [] for w in set(sw.word for sw in subs)}
    [d[w.word].append(i) for i, w in enumerate(recs) if w.word in d.keys()]
    return d


def calc_timeframe(subs, recs):
    return abs(subs[-1].end - recs[-1].end) + 30


def find_longest_chain(subs, recs, seek):
    # Initialize the longest chain to be empty
    words_pos = {w: [] for w in set(sw.word for sw in subs)}
    [words_pos[w.word].append(i) for i, w in enumerate(recs) if w.word in words_pos.keys()]
    time_frame = calc_timeframe(subs, recs)
    longest_chain = {'indsub': -1, 'indrec': -1, 'length': 0, 'chain': []}
    # Loop through the words in the first list
    for i, word1 in enumerate(subs):
        if not seek.sub_from < i: continue
        if i >= seek.sub_to: break
        if word1.word in words_pos.keys():
            for n in words_pos[word1.word]:
                if n < seek.rec_from: continue
                if n >= seek.rec_to: break
                if not (subs[i].start - time_frame) < recs[n].start < (subs[i].end + time_frame): continue
                y = 0
                while (i + y) < seek.sub_to and (n + y) < seek.rec_to and subs[i + y].word == recs[n + y].word:
                    y += 1
                if y > longest_chain['length']:
                    longest_chain.update(
                        {'indsub': i, 'indrec': n, 'length': y, 'chain': [subs[i + z].word for z in range(y)]})
    return longest_chain

def interp(x,x_values,y_values):
    """linear interpolation istead call from numpy"""
    if x <= x_values[0]:
        return y_values[0]
    if x >= x_values[-1]:
        return y_values[-1]
    i = 0
    while x_values[i] < x:
        i += 1
    tilt = (y_values[i] - y_values[i - 1]) / (x_values[i] - x_values[i - 1])
    y = y_values[i - 1] + tilt * (x - x_values[i - 1])
    return y

def correct_srt(video_filename, model_name, metadata, audio_stream_num=None):
    subtitle_filename = os.path.abspath(os.path.splitext(video_filename)[0] + '.srt')
    subt = get_subtitles(subtitle_filename)
    subs = get_subwords1(subt)
    audio_stream = find_native_stream(video_filename, model_name,
                                      metadata) if audio_stream_num == None else audio_stream_num
    recs = recognize(video_filename, model_name, metadata, audio_stream)
    l1 = len(subs)
    l2 = len(recs)
    subs_link = [None] * l1
    #### find long chains
    Seek = namedtuple('Seek', 'sub_from sub_to rec_from rec_to')
    seeks = [Seek(0, l1, 0, l2)]
    cw = 0
    while len(seeks) > 0:
        seek = seeks[0]
        longest_chain = find_longest_chain(subs, recs, seeks[0])
        l = longest_chain['length']
        #### find chains longest than 2 words
        if l > 2:
            cw = cw + l
            i = longest_chain['indsub']
            r = longest_chain['indrec']
            for y in range(longest_chain['length']):
                if subs_link[i + y] != None: print('-----Alarm!!!')
                subs_link[i + y] = r + y
            seeks.append(Seek(seek.sub_from, i, seek.rec_from, r))
            seeks.append(Seek(i + l, seek.sub_to, r + l, seek.rec_to))
        seeks.pop(0)
    # print(cw,l1,l2,cw/l1)
    if cw / l1 < 0.25:
        return False
    # correct subt by founded chains
    a, j, k, m = 0, 0, 0, 0
    for n, s in enumerate(subs):
        if subs_link[n] == None:
            continue
        if s.pos == 0:
            j += 1
            cor = s.start - recs[subs_link[n]].start
            subt[s.subtitr]['correct'] = cor
        if n == l1 - 1 or subs[n + 1].pos == 0:
            k = k + 1
            cor = s.end - recs[subs_link[n]].end
            if subt[s.subtitr]['correct'] == None:
                subt[s.subtitr]['correct'] = cor
    f = list(zip(*[(s['start'], s['correct']) for s in subt if s['correct'] != None]))
    for s in subt:
        if s['correct'] == None:
            s['correct'] = interp(s['start'], f[0],f[1])
            # s['correct'] = np.interp(s['start'], np.array(f[0]), np.array(f[1]))
    put_subtitles(subtitle_filename, subt)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('movie', type=str, nargs='+', help='path to movie file')
    parser.add_argument('-s', '--stream', type=int, help='audio stream number to recognize')
    if not is_installed_ffmpeg():
        print('not installed ffmpeg')
        sys.exit(1)
    model_name = get_model_name()
    args = parser.parse_args(sys.argv[1:])
    movie = os.path.abspath(' '.join(args.movie))
    subtitle_filename = os.path.splitext(movie)[0] + '.srt'
    if not model_name:
        print('neural network not found or not specifed')
        sys.exit(1)
    if not os.path.exists(movie) or not os.path.isfile(movie):
        print(f'not found video file: {movie}')
        sys.exit(1)
    if not os.path.exists(subtitle_filename) or not os.path.isfile(subtitle_filename):
        print(f'not found subtiles file: {subtitle_filename}')
        sys.exit(1)
    metadata = get_metadata(movie)
    if args.stream != None and not (0 <= args.stream < len(metadata['streams'])):
        print('wrong number audiostream')
        sys.exit(1)
    if not correct_srt(movie, model_name, metadata, args.stream):
        print('unable to recognize speech, process stopped')
        sys.exit(1)
    else:
        print('successfully fixed time in  SRT file')


if __name__ == '__main__':
    main()
