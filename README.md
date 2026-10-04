# Wuanii Voice

Free, human-style text-to-speech for YouTube Shorts. Type a script, pick a voice and style, download the audio.

- Voice engine: [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (open source, 82M params), runs on a Vercel Python function
- The 325 MB model is downloaded on the server at first use, never on your phone
- Front end: single static page in `public/`

## API
`POST /api/tts` with JSON `{ "text": "...", "voice": "am_puck", "style": "hype|normal|dramatic", "speed": 1.0 }` returns a WAV file.
