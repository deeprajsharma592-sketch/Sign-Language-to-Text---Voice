# Sign Language to Text and Voice (Real Time)

A computer-vision app that reads sign language gestures from an ordinary webcam and turns them into written text and spoken audio as they happen. It is built to help close the communication gap between people who sign and people who do not.

It recognizes a small vocabulary of isolated signs (5 to 10 words plus an `idle` class). It is an MVP, not a full sign language translator.

## How it works

```
Webcam -> MediaPipe Hands -> normalized landmarks -> 30-frame window
       -> GRU classifier -> confidence + debounce -> text on screen + speech
```

1. **Hand tracking.** MediaPipe extracts 21 landmarks per hand (x, y, z), up to 2 hands, giving 126 numbers per frame.
2. **Normalization.** Landmarks are made wrist-relative and scaled by hand size, so the model does not care where your hand is in the frame or how far it is from the camera.
3. **Movement, not single frames.** A sliding window of the last 30 frames (about 1 second) goes into a small GRU. Many signs differ only in motion.
4. **Stable output.** A word is emitted only when the same label wins with high confidence several times in a row, then a cooldown stops repeats. `idle` never produces output.
5. **Voice.** Text-to-speech runs off the video loop, so speech never stalls the camera.

## Two ways to run it

| Mode | Command | Camera and speech |
|---|---|---|
| **Desktop** (OpenCV window) | `python sign2text.py run` | Python webcam, pyttsx3 voice |
| **Web** (browser frontend) | `uvicorn server:app --port 8000` | Browser webcam, browser voice |

In web mode, the browser runs MediaPipe (JavaScript) and sends only hand landmarks over a WebSocket. The Python server normalizes them, runs the model, and sends back predictions. Video never leaves the browser.

```
Browser: webcam -> MediaPipe JS -> raw landmarks
                                      | WebSocket (about 1 KB per frame)
Server:  normalize -> window -> GRU -> debounce -> word
                                      | WebSocket
Browser: shows prediction + sentence, speaks via Web Speech API
```

## Project files

| File | Purpose |
|---|---|
| `sign2text.py` | Data collection, training and the desktop live demo (`collect`, `train`, `run`) |
| `server.py` | FastAPI WebSocket server that serves the model to the browser |
| `index.html` | Browser frontend (webcam, landmarks overlay, live text, voice) |
| `data/<word>/*.npy` | Recorded landmark sequences, created by `collect` |
| `sign_model.pt` | Trained model, created by `train` |

## Requirements

- Python 3.10 or 3.11
- A webcam
- Runs on CPU. No GPU needed.
- For web mode: a modern browser (Chrome or Edge recommended) and an internet connection on first load, because MediaPipe's browser files load from a CDN.

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install mediapipe opencv-python numpy torch pyttsx3
pip install fastapi "uvicorn[standard]"      # only needed for web mode
```

## Quick start

### 1. Pick your signs

Choose 5 signs that look clearly different from each other, for example `hello`, `thank_you`, `yes`, `no`, `help`. Learn each sign from a video dictionary of one sign language (ASL or ISL), and do not mix languages. You also need an `idle` class (see below).

### 2. Record data

```bash
python sign2text.py collect hello
python sign2text.py collect thank_you
python sign2text.py collect yes
python sign2text.py collect no
python sign2text.py collect help
python sign2text.py collect idle
```

Each command records 40 samples. A countdown appears, then it records 1 second of movement. Tips for good data:

- Record 40 to 60 samples per sign. You can run the command again to add more, and it appends.
- **Vary the conditions between batches:** lighting, distance from the camera, sitting versus standing, clothing, and which hand leads.
- **Record `idle` properly:** hands resting, hands entering and leaving the frame, random fidgeting. Without it, the model will force every movement into one of your words.
- If you can, get 2 or 3 other people to record samples. This is the biggest boost to real-world accuracy.

### 3. Train

```bash
python sign2text.py train
```

This trains on everything in `data/`, prints validation accuracy, and saves `sign_model.pt`. Training takes minutes on CPU.

### 4. Run

**Desktop:**
```bash
python sign2text.py run
```
Press `q` to quit.

**Web:**
```bash
uvicorn server:app --port 8000
```
Open `http://localhost:8000` and allow camera access. `server.py`, `index.html`, `sign2text.py` and `sign_model.pt` must be in the same folder.

## Adding a new sign

1. `python sign2text.py collect new_sign`
2. `python sign2text.py train`
3. Restart the app (or the server).

If the new sign gets confused with an existing one, record more varied samples of both, or choose a more distinct sign.

## Configuration

The tunable values are at the top of `sign2text.py`:

| Setting | Default | What it does |
|---|---|---|
| `SEQ_LEN` | 30 | Frames per gesture window |
| `CONF_THRESH` | 0.85 | Minimum confidence to accept a prediction |
| `STABLE_N` | 3 | Same label this many times in a row before it is emitted |
| `COOLDOWN` | 1.5 | Seconds before the same word can be spoken again |

If you change `SEQ_LEN`, re-record data and retrain. Old samples will not match the new window length.

## Troubleshooting

**Nothing is recognized, or it keeps predicting the wrong word**
Check that you have an `idle` class, enough varied samples, and that signs are distinct. Lower `CONF_THRESH` slightly if nothing ever fires, or raise it if you get false triggers.

**Works for me, fails for someone else**
Your training data is too uniform. Add samples from other people, lighting conditions and distances.

**Browser predictions look swapped compared to the desktop demo**
Set `SWAP_HANDS = True` in `server.py`. Training data and live input must use the same left/right convention. Recording data through the browser page keeps them consistent.

**Camera does not start in the browser**
Use `http://localhost:8000`, not an IP address. Browsers block camera access on non-localhost HTTP. Also check the site's camera permission.

**Web page shows "server: reconnecting"**
Make sure `uvicorn` is running and that `sign_model.pt` exists. Run `python sign2text.py train` first.

**GPU error in the browser**
Open `index.html` and change `delegate: "GPU"` to `delegate: "CPU"`.

**First page load is slow or fails offline**
The browser downloads MediaPipe's wasm and hand model from a CDN on first load. Download them and serve them locally for an offline demo.

**No voice in desktop mode**
`pyttsx3` needs a system speech engine. On Linux, install `espeak`.

**Using it from a phone or another computer**
The camera needs HTTPS outside localhost. Use a tunnel such as ngrok or Cloudflare Tunnel, and change the WebSocket URL in `index.html` from `ws://` to `wss://`.

## Known limitations

- Recognizes **isolated words only** from a small vocabulary. It does not do continuous sentences.
- Uses **hand landmarks only.** Real sign languages also rely on facial expressions and body position.
- Accuracy depends heavily on the variety of your training data. Validation accuracy from a single session is optimistic. Test with a person who was not in your training data.
- Does not handle fingerspelling.

## Roadmap ideas

- Add face and pose landmarks (MediaPipe Holistic) by changing the feature extractor.
- Split train and validation by person or session for honest accuracy numbers.
- Train on a public dataset such as WLASL for a larger vocabulary.
- Add a language layer that turns a word sequence into a grammatical sentence.
- Measure and optimize per-stage latency (landmarks, model, network).

## Credits

Built with [MediaPipe](https://developers.google.com/mediapipe), [OpenCV](https://opencv.org), [PyTorch](https://pytorch.org), [FastAPI](https://fastapi.tiangolo.com) and the Web Speech API.
